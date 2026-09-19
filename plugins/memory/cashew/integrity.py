"""Read-only integrity inspection and explicit upstream repair delegation.

Inspection remains query-only.  The selected immutable Cashew composite exposes
the stable connection-aware contract, so ``inspect_integrity`` and the
explicitly confirmed ``apply_integrity_repairs`` delegate to it without taking
connection or transaction ownership.  Older installations continue to return
the structured unavailable result; no local repair algorithm is maintained
here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import shutil
import sqlite3
import struct
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, cast

from .admission import admit_operation
from .locking import (
    is_supported_vec_ddl,
    open_readonly_verified,
    verify_readonly_profile,
)

logger = logging.getLogger(__name__)

_REQUIRED_TABLES = {
    "thought_nodes",
    "embeddings",
    "derivation_edges",
    "hermes_provider_meta",
}
_REQUIRED_NODE_COLUMNS = {
    "id",
    "content",
    "node_type",
    "domain",
    "timestamp",
    "access_count",
    "last_accessed",
    "source_file",
    "decayed",
    "metadata",
    "last_updated",
    "mood_state",
    "permanent",
    "tags",
    "referent_time",
}
_REQUIRED_EMBEDDING_COLUMNS = {"node_id", "vector", "model", "updated_at"}
_REQUIRED_EDGE_COLUMNS = {"parent_id", "child_id", "weight", "reasoning", "timestamp"}
_REQUIRED_META = {"embedding_model", "embedding_dim", "vec_dim", "maintenance_epoch"}
_AUDIT_DEADLINE_SECONDS = 5.0
_MAX_AUDIT_ROWS = 100_000
_MAX_AUDIT_BYTES = 64 * 1024 * 1024
_MAX_VECTOR_BYTES = 4 * 1024 * 1024
_AUDIT_BATCH_SIZE = 256


class _AuditBudgetError(RuntimeError):
    """The audit cannot safely continue within its fixed work budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclasses.dataclass
class _AuditBudget:
    """Small, explicit work budget shared by every potentially large scan."""

    deadline: float
    rows: int = 0
    bytes: int = 0
    incomplete_reasons: set[str] = dataclasses.field(default_factory=set)
    scans: dict[str, bool] = dataclasses.field(default_factory=dict)

    @property
    def incomplete(self) -> bool:
        return bool(self.incomplete_reasons)

    def check(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.incomplete_reasons.add("audit_deadline")
            return False
        return True

    def ready(self) -> bool:
        return self.check() and not self.incomplete_reasons

    def reserve_bytes(self, byte_count: int) -> bool:
        if not self.ready():
            return False
        if byte_count < 0 or self.bytes + byte_count > _MAX_AUDIT_BYTES:
            self.incomplete_reasons.add("audit_byte_cap")
            return False
        self.bytes += byte_count
        return True

    def consume(self, byte_count: int = 0) -> bool:
        if not self.ready():
            return False
        if self.rows >= _MAX_AUDIT_ROWS:
            self.incomplete_reasons.add("audit_row_cap")
            return False
        if not self.reserve_bytes(byte_count):
            return False
        self.rows += 1
        return True

    def progress(self) -> int:
        if time.monotonic() >= self.deadline:
            self.incomplete_reasons.add("audit_deadline")
            return 1
        return 0


def _table_exists(
    conn: sqlite3.Connection, table: str, budget: _AuditBudget | None = None
) -> bool:
    if budget is not None and not budget.ready():
        return False
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _table_inventory(
    conn: sqlite3.Connection, budget: _AuditBudget
) -> tuple[set[str], dict[str, Any]]:
    """Inspect known tables while hashing, never returning arbitrary names."""
    known = {name: _table_exists(conn, name, budget) for name in _REQUIRED_TABLES}
    unknown_count = 0
    digest = hashlib.sha256()
    complete = budget.ready()
    if complete:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' "
            "AND name NOT IN (?, ?, ?, ?) "
            "ORDER BY name "
            f"LIMIT {_MAX_AUDIT_ROWS + 1}",
            tuple(sorted(_REQUIRED_TABLES)),
        )
        while budget.ready():
            batch = cursor.fetchmany(_AUDIT_BATCH_SIZE)
            if not batch:
                break
            for row in batch:
                name = str(row[0])
                if not budget.consume(len(name.encode("utf-8")) + 1):
                    complete = False
                    break
                unknown_count += 1
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
            if budget.incomplete:
                complete = False
                break
        if unknown_count > _MAX_AUDIT_ROWS:
            complete = False
    fingerprint = digest.hexdigest()[:16] if unknown_count else None
    budget.scans["table_inventory"] = complete and not budget.incomplete
    return (
        {name for name, present in known.items() if present},
        {
            "required_tables": known,
            "unexpected_table_count": unknown_count if complete else None,
            "unexpected_table_fingerprint": fingerprint,
            "inventory_complete": complete and not budget.incomplete,
        },
    )


def _columns(
    conn: sqlite3.Connection, table: str, budget: _AuditBudget | None = None
) -> set[str]:
    # Table names are selected from sqlite_master or fixed constants above.
    if budget is not None and not budget.ready():
        return set()
    cursor = conn.execute(f"PRAGMA table_info({table})")
    result: set[str] = set()
    while True:
        batch = cursor.fetchmany(_AUDIT_BATCH_SIZE)
        if not batch:
            break
        for row in batch:
            if budget is not None and not budget.consume():
                return result
            result.add(str(row[1]))
    return result


def _safe_source_id(conn: sqlite3.Connection) -> str | None:
    try:
        value = conn.execute("SELECT sqlite_source_id()").fetchone()
    except sqlite3.Error:
        return None
    return None if not value else str(value[0])


@contextmanager
def _source_snapshot(
    path: pathlib.Path, budget: _AuditBudget
) -> Iterator[pathlib.Path]:
    """Copy a profile and its live sidecars before opening any SQLite handle.

    SQLite may update reader marks in a WAL ``-shm`` file even for a query-only
    connection.  Auditing a private copy keeps the operator's profile, WAL,
    and SHM bytes untouched while preserving the WAL state for inspection.
    Cooperative writers are excluded by the caller's shared maintenance lease.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    source_files = [path]
    source_files.extend(
        pathlib.Path(f"{path}{suffix}")
        for suffix in ("-wal", "-shm")
        if pathlib.Path(f"{path}{suffix}").exists()
    )
    for source in source_files:
        try:
            size = source.stat().st_size
        except OSError:
            raise
        if not budget.reserve_bytes(size):
            raise _AuditBudgetError("audit_byte_cap")
    with tempfile.TemporaryDirectory(prefix="hermes-cashew-audit-") as directory:
        snapshot = pathlib.Path(directory) / "profile.db"
        if not budget.ready():
            raise _AuditBudgetError(
                next(iter(budget.incomplete_reasons), "audit_deadline")
            )
        shutil.copy2(path, snapshot)
        if not budget.ready():
            raise _AuditBudgetError(
                next(iter(budget.incomplete_reasons), "audit_deadline")
            )
        for source in source_files[1:]:
            suffix = source.name[len(path.name) :]
            shutil.copy2(source, pathlib.Path(f"{snapshot}{suffix}"))
            if not budget.ready():
                raise _AuditBudgetError(
                    next(iter(budget.incomplete_reasons), "audit_deadline")
                )
        yield snapshot


def _provenance(
    conn: sqlite3.Connection, journal_mode: str, budget: _AuditBudget | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_source_id": _safe_source_id(conn),
        "journal_mode": journal_mode,
        "user_version": None,
        "provider_model": None,
        "provider_embedding_dim": None,
        "provider_vec_dim": None,
        "provider_epoch": None,
    }
    try:
        result["user_version"] = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except (TypeError, ValueError, sqlite3.Error):
        pass
    if not _table_exists(conn, "hermes_provider_meta", budget):
        return result
    try:
        rows = conn.execute(
            "SELECT key, value FROM hermes_provider_meta WHERE key IN "
            "('embedding_model','embedding_dim','vec_dim','maintenance_epoch')"
        )
        bounded_rows = []
        while True:
            batch = rows.fetchmany(_AUDIT_BATCH_SIZE)
            if not batch:
                break
            for row in batch:
                if budget is not None and not budget.consume():
                    break
                bounded_rows.append(tuple(row))
            if budget is not None and budget.incomplete:
                break
    except sqlite3.Error:
        return result
    meta = {str(key): value for key, value in bounded_rows}
    result["provider_model"] = meta.get("embedding_model")
    result["provider_epoch"] = meta.get("maintenance_epoch")
    for key, output_key in (
        ("embedding_dim", "provider_embedding_dim"),
        ("vec_dim", "provider_vec_dim"),
    ):
        try:
            result[output_key] = int(meta[key])
        except (KeyError, TypeError, ValueError):
            result[output_key] = None
    return result


def _fingerprint(value: object) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _safe_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Project internal identity values into a payload-safe public report."""
    model = provenance.pop("provider_model", None)
    epoch = provenance.pop("provider_epoch", None)
    source_id = provenance.pop("sqlite_source_id", None)
    provenance["provider_model_fingerprint"] = _fingerprint(model)
    provenance["provider_epoch_present"] = epoch is not None
    provenance["provider_epoch_valid"] = _positive_int(epoch) is not None
    provenance["sqlite_source_id_fingerprint"] = _fingerprint(source_id)
    return provenance


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _add_reason(reasons: Counter[str], reason: str, count: int = 1) -> None:
    if reason and count > 0:
        reasons[reason] += count


def _finite_vector(
    blob: object, expected_dim: int | None
) -> tuple[str | None, int | None]:
    """Return a bounded reason and decoded dimension without raising."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return "embedding_blob_invalid", None
    raw = bytes(blob)
    if len(raw) == 0 or len(raw) % 4:
        return "embedding_blob_invalid", None
    dim = len(raw) // 4
    if dim > 16384:
        return "embedding_dimension_invalid", dim
    if expected_dim is not None and dim != expected_dim:
        return "embedding_dimension_mismatch", dim
    try:
        values = struct.unpack(f"<{dim}f", raw)
    except (struct.error, ValueError):
        return "embedding_blob_invalid", dim
    if not all(math.isfinite(value) for value in values):
        return "embedding_nonfinite", dim
    if math.sqrt(sum(value * value for value in values)) <= 1e-12:
        return "embedding_zero_norm", dim
    return None, dim


def _load_vec_readonly(conn: sqlite3.Connection) -> bool:
    """Load only the virtual-table module; the database remains query-only."""
    try:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec

            sqlite_vec.load(conn)
        except (ImportError, AttributeError):
            conn.load_extension("vec0")
        return True
    except Exception:
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def _inspect_vec(
    conn: sqlite3.Connection,
    ordinary_ids: set[str],
    expected_dim: int | None,
    reasons: Counter[str],
    budget: _AuditBudget,
) -> dict[str, Any]:
    vec = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
    ).fetchone()
    if vec is None:
        budget.scans["vec_index"] = True
        _add_reason(reasons, "vec_index_missing", len(ordinary_ids))
        return {
            "available": False,
            "entries": 0,
            "missing_entries": len(ordinary_ids),
            "stale_entries": 0,
            "declared_dimension": None,
            "scan_complete": True,
        }
    sql = str(vec[1] or "")
    if not is_supported_vec_ddl(sql):
        budget.scans["vec_index"] = False
        _add_reason(reasons, "vec_index_unverifiable")
        return {
            "available": False,
            "entries": None,
            "missing_entries": None,
            "stale_entries": None,
            "declared_dimension": None,
            "scan_complete": False,
        }
    if not _load_vec_readonly(conn):
        budget.scans["vec_index"] = False
        _add_reason(reasons, "vec_index_unverifiable")
        return {
            "available": False,
            "entries": None,
            "missing_entries": None,
            "stale_entries": None,
            "declared_dimension": None,
            "scan_complete": False,
        }
    match = re.search(r"(?:float|int8)\s*\[\s*(\d+)\s*\]", sql, re.IGNORECASE)
    declared_dim = int(match.group(1)) if match else None
    if declared_dim is None:
        _add_reason(reasons, "vec_dimension_unavailable")
    elif expected_dim is not None and declared_dim != expected_dim:
        _add_reason(reasons, "vec_dimension_mismatch")
    try:
        rows: list[tuple[object, object]] = []
        cursor = conn.execute(
            "SELECT node_id, LENGTH(embedding) FROM vec_embeddings "
            f"LIMIT {_MAX_AUDIT_ROWS + 1}"
        )
        while budget.ready():
            batch = cursor.fetchmany(_AUDIT_BATCH_SIZE)
            if not batch:
                break
            for row in batch:
                length = 0 if row[1] is None else int(row[1])
                if not budget.consume(length):
                    break
                rows.append((row[0], row[1]))
            if budget.incomplete:
                break
        budget.scans["vec_index"] = not budget.incomplete
    except (sqlite3.Error, TypeError, ValueError):
        budget.scans["vec_index"] = False
        _add_reason(reasons, "vec_index_unverifiable")
        return {
            "available": False,
            "entries": None,
            "missing_entries": None,
            "stale_entries": None,
            "declared_dimension": declared_dim,
            "scan_complete": False,
        }
    vec_ids = {str(row[0]) for row in rows if row[0] is not None}
    if budget.incomplete:
        return {
            "available": True,
            "entries": len(rows),
            "missing_entries": None,
            "stale_entries": None,
            "declared_dimension": declared_dim,
            "scan_complete": False,
        }
    missing = ordinary_ids - vec_ids
    stale = vec_ids - ordinary_ids
    _add_reason(reasons, "vec_entry_missing", len(missing))
    _add_reason(reasons, "vec_entry_stale", len(stale))
    invalid_lengths = 0
    for vec_row in rows:
        vec_length = vec_row[1]
        if vec_length is None or (
            expected_dim is not None and int(str(vec_length)) != expected_dim * 4
        ):
            invalid_lengths += 1
    _add_reason(reasons, "vec_blob_dimension_mismatch", invalid_lengths)
    budget.scans["vec_index"] = not budget.incomplete
    return {
        "available": True,
        "entries": len(rows),
        "missing_entries": len(missing),
        "stale_entries": len(stale),
        "declared_dimension": declared_dim,
        "scan_complete": not budget.incomplete,
    }


def _graph_findings(
    conn: sqlite3.Connection, reasons: Counter[str], budget: _AuditBudget
) -> dict[str, int]:
    """Report referential graph defects without invoking upstream mutators."""
    if not budget.ready():
        budget.scans["graph"] = False
        return {"orphan_edges": 0, "self_edges": 0}
    try:
        orphan_edges = _bounded_count(
            conn,
            "SELECT 1 FROM derivation_edges e "
            "LEFT JOIN thought_nodes p ON p.id=e.parent_id "
            "LEFT JOIN thought_nodes c ON c.id=e.child_id "
            "WHERE p.id IS NULL OR c.id IS NULL",
            budget,
            label="orphan_edges",
        )
        self_edges = _bounded_count(
            conn,
            "SELECT 1 FROM derivation_edges WHERE parent_id=child_id",
            budget,
            label="self_edges",
        )
    except sqlite3.Error:
        budget.scans["graph"] = False
        _add_reason(reasons, "graph_unverifiable")
        return {"orphan_edges": 0, "self_edges": 0}
    _add_reason(reasons, "orphan_edge", orphan_edges)
    _add_reason(reasons, "self_edge", self_edges)
    budget.scans["graph"] = not budget.incomplete
    return {"orphan_edges": orphan_edges, "self_edges": self_edges}


def _bounded_count(
    conn: sqlite3.Connection, sql: str, budget: _AuditBudget, *, label: str
) -> int:
    """Count matches through a capped row stream, preserving completeness state."""
    if not budget.ready():
        budget.scans[label] = False
        return 0
    count = 0
    cursor = conn.execute(sql + f" LIMIT {_MAX_AUDIT_ROWS + 1}")
    while budget.ready():
        batch = cursor.fetchmany(_AUDIT_BATCH_SIZE)
        if not batch:
            budget.scans[label] = True
            return count
        for _row in batch:
            if not budget.consume():
                budget.scans[label] = False
                return count
            count += 1
    budget.scans[label] = False
    return count


def _inspect_profile(  # noqa: C901
    conn: sqlite3.Connection, journal_mode: str, budget: _AuditBudget
) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    provenance = _provenance(conn, journal_mode, budget)
    tables, table_report = _table_inventory(conn, budget)
    missing_tables = sorted(_REQUIRED_TABLES - tables)
    _add_reason(reasons, "schema_table_missing", len(missing_tables))
    if provenance["user_version"] != 3:
        _add_reason(reasons, "schema_version_unsupported")

    missing_columns: dict[str, list[str]] = {}
    for table, required in (
        ("thought_nodes", _REQUIRED_NODE_COLUMNS),
        ("embeddings", _REQUIRED_EMBEDDING_COLUMNS),
        ("derivation_edges", _REQUIRED_EDGE_COLUMNS),
    ):
        if table not in tables:
            continue
        missing = sorted(required - _columns(conn, table, budget))
        if missing:
            missing_columns[table] = missing
            _add_reason(reasons, "schema_column_missing", len(missing))
    try:
        meta_cursor = conn.execute(
            "SELECT key FROM hermes_provider_meta WHERE key IN "
            "('embedding_model','embedding_dim','vec_dim','maintenance_epoch') "
            "LIMIT 4"
        )
        meta_keys: set[str] = set()
        for row in meta_cursor.fetchmany(4):
            if not budget.consume():
                break
            meta_keys.add(str(row[0]))
    except sqlite3.Error:
        meta_keys = set()
    missing_meta = (
        sorted(_REQUIRED_META - meta_keys)
        if "hermes_provider_meta" in tables
        else sorted(_REQUIRED_META)
    )
    _add_reason(reasons, "provider_identity_missing", len(missing_meta))

    counts: Counter[str] = Counter()
    vec_report: dict[str, Any] = {
        "available": False,
        "entries": None,
        "missing_entries": None,
        "stale_entries": None,
        "declared_dimension": None,
    }
    graph_report = {"orphan_edges": 0, "self_edges": 0}
    if not missing_tables and not missing_columns and budget.ready():
        try:
            integrity = str(
                conn.execute("PRAGMA integrity_check").fetchone()[0]
            ).lower()
        except sqlite3.Error:
            integrity = "unavailable"
        if integrity != "ok":
            _add_reason(reasons, "sqlite_integrity_failed")

        expected_dim = provenance["provider_embedding_dim"]
        expected_model = provenance["provider_model"]
        ordinary_ids: set[str] = set()
        counts["embeddings"] = 0
        cursor = conn.execute(
            "SELECT e.node_id, e.vector, e.model FROM embeddings e "
            f"LIMIT {_MAX_AUDIT_ROWS + 1}"
        )
        while budget.ready():
            batch = cursor.fetchmany(_AUDIT_BATCH_SIZE)
            if not batch:
                break
            for node_id, blob, model in batch:
                byte_count = (
                    len(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else 0
                )
                if byte_count > _MAX_VECTOR_BYTES:
                    budget.incomplete_reasons.add("audit_byte_cap")
                    break
                if not budget.consume(byte_count):
                    break
                counts["embeddings"] += 1
                if node_id is not None:
                    ordinary_ids.add(str(node_id))
                reason, _ = _finite_vector(blob, expected_dim)
                _add_reason(reasons, reason or "", 1)
                if expected_model is not None and model != expected_model:
                    _add_reason(reasons, "embedding_model_mismatch")
                if node_id is None:
                    _add_reason(reasons, "embedding_node_id_invalid")
            if budget.incomplete:
                break
        budget.scans["embeddings"] = not budget.incomplete
        if budget.ready():
            counts["orphan_embeddings"] = _bounded_count(
                conn,
                "SELECT 1 FROM embeddings e "
                "LEFT JOIN thought_nodes n ON n.id=e.node_id WHERE n.id IS NULL",
                budget,
                label="orphan_embeddings",
            )
            counts["nodes_without_embeddings"] = _bounded_count(
                conn,
                "SELECT 1 FROM thought_nodes n "
                "LEFT JOIN embeddings e ON e.node_id=n.id WHERE e.node_id IS NULL",
                budget,
                label="nodes_without_embeddings",
            )
        else:
            counts["orphan_embeddings"] = 0
            counts["nodes_without_embeddings"] = 0
        _add_reason(reasons, "orphan_embedding", counts["orphan_embeddings"])
        _add_reason(
            reasons, "node_embedding_missing", counts["nodes_without_embeddings"]
        )
        try:
            counts["permanent_and_decayed"] = _bounded_count(
                conn,
                "SELECT 1 FROM thought_nodes WHERE COALESCE(permanent,0) != 0 "
                "AND COALESCE(decayed,0) != 0",
                budget,
                label="permanent_and_decayed",
            )
            counts["permanent_core_nodes"] = _bounded_count(
                conn,
                "SELECT 1 FROM thought_nodes WHERE node_type='core_memory' "
                "AND COALESCE(permanent,0) != 0",
                budget,
                label="permanent_core_nodes",
            )
        except sqlite3.Error:
            counts["permanent_and_decayed"] = 0
            counts["permanent_core_nodes"] = 0
            _add_reason(reasons, "permanence_unverifiable")
        _add_reason(reasons, "permanent_and_decayed", counts["permanent_and_decayed"])
        # A permanent core_memory node is expected state.  Contradictory
        # permanent+decayed state is still reported above.
        if budget.ready():
            vec_report = _inspect_vec(conn, ordinary_ids, expected_dim, reasons, budget)
        if budget.ready():
            graph_report = _graph_findings(conn, reasons, budget)
    else:
        counts["embeddings"] = 0
        counts["orphan_embeddings"] = 0
        counts["nodes_without_embeddings"] = 0

    # Historical merge intent cannot be reconstructed from the current schema.
    uncertainty = ["historical_consolidation"]
    for incomplete_reason in budget.incomplete_reasons:
        _add_reason(reasons, incomplete_reason)
    return {
        "provenance": _safe_provenance(provenance),
        "schema": {
            **table_report,
            "missing_tables": missing_tables,
            "missing_columns": missing_columns,
            "missing_provider_keys": missing_meta,
        },
        "counts": dict(counts),
        "vector_index": vec_report,
        "graph": graph_report,
        "completeness": dict(sorted(budget.scans.items())),
        "uncertainty": uncertainty,
        "reasons": dict(sorted(reasons.items())),
        "informational": {
            "permanent_core_nodes": counts.get("permanent_core_nodes", 0)
        },
        "limits": {
            "rows_scanned": budget.rows,
            "bytes_scanned": budget.bytes,
            "row_cap": _MAX_AUDIT_ROWS,
            "byte_cap": _MAX_AUDIT_BYTES,
            "complete": not budget.incomplete,
        },
    }


def _unavailable(path: pathlib.Path, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "unavailable",
        "read_only": True,
        "mutated": False,
        "profile_fingerprint": _fingerprint(path),
        "provenance": {},
        "schema": {},
        "counts": {},
        "vector_index": {},
        "graph": {},
        "completeness": {},
        "uncertainty": [],
        "reasons": {reason: 1},
        "repair": {
            "status": "unavailable",
            "reason": "stable_targeted_repair_api_unavailable",
        },
    }


def _incomplete(
    path: pathlib.Path, budget: _AuditBudget, reason: str
) -> dict[str, Any]:
    budget.incomplete_reasons.add(reason)
    return {
        "schema_version": 1,
        "status": "audit_incomplete",
        "read_only": True,
        "mutated": False,
        "profile_fingerprint": _fingerprint(path),
        "provenance": {},
        "schema": {},
        "counts": {},
        "vector_index": {},
        "graph": {},
        "completeness": {},
        "uncertainty": [],
        "reasons": {item: 1 for item in sorted(budget.incomplete_reasons)},
        "limits": {
            "rows_scanned": budget.rows,
            "bytes_scanned": budget.bytes,
            "row_cap": _MAX_AUDIT_ROWS,
            "byte_cap": _MAX_AUDIT_BYTES,
            "complete": False,
        },
        "repair": {
            "status": "unavailable",
            "reason": "stable_targeted_repair_api_unavailable",
            "explicit_apply_required": True,
        },
    }


def audit_integrity(
    db_path: str | pathlib.Path, *, deadline_seconds: float = _AUDIT_DEADLINE_SECONDS
) -> dict[str, Any]:
    """Return a bounded, query-only integrity report for an existing profile."""
    path = pathlib.Path(db_path).resolve(strict=False)
    budget = _AuditBudget(deadline=time.monotonic() + max(0.0, deadline_seconds))
    try:
        with admit_operation(graph_path=path, deadline=deadline_seconds):
            try:
                with _source_snapshot(path, budget) as snapshot:
                    try:
                        conn, journal_mode = open_readonly_verified(snapshot)
                    except Exception as exc:
                        del exc
                        return _unavailable(path, "readonly_open_failed")
                    try:
                        profile_error: str | None = None
                        try:
                            # Install the deadline callback before the verifier's
                            # integrity_check and bounded scans, so those SQLite
                            # operations cannot outrun the shared audit budget.
                            conn.set_progress_handler(budget.progress, 1000)
                            verify_readonly_profile(conn, budget=budget)
                        except _AuditBudgetError as exc:
                            return _incomplete(path, budget, exc.reason)
                        except Exception as exc:
                            del exc
                            if budget.incomplete_reasons:
                                reason = next(
                                    (
                                        reason
                                        for reason in (
                                            "audit_deadline",
                                            "audit_row_cap",
                                            "audit_byte_cap",
                                        )
                                        if reason in budget.incomplete_reasons
                                    ),
                                    sorted(budget.incomplete_reasons)[0],
                                )
                                return _incomplete(path, budget, reason)
                            profile_error = "profile_verification_failed"
                        report = _inspect_profile(conn, journal_mode, budget)
                        if profile_error is not None:
                            reasons = cast(dict[str, Any], report["reasons"])
                            reasons[profile_error] = 1
                            report["reasons"] = dict(reasons)
                    except Exception as exc:
                        logger.warning(
                            "Cashew integrity audit could not inspect profile"
                        )
                        del exc
                        return _unavailable(path, "audit_failed")
                    finally:
                        try:
                            conn.set_progress_handler(None, 0)
                        except Exception:
                            pass
                        conn.close()
            except _AuditBudgetError as exc:
                return _incomplete(path, budget, exc.reason)
            except Exception as exc:
                del exc
                return _unavailable(path, "readonly_snapshot_failed")
    except _AuditBudgetError as exc:
        return _incomplete(path, budget, exc.reason)
    except Exception as exc:
        del exc
        return _unavailable(path, "readonly_admission_failed")

    report["schema_version"] = 1
    incomplete_reasons = {
        reason
        for reason in report.get("reasons", {})
        if str(reason).startswith("audit_")
    }
    report["status"] = (
        "audit_incomplete"
        if incomplete_reasons
        else ("ok" if not report["reasons"] else "findings")
    )
    report["read_only"] = True
    report["mutated"] = False
    report["profile_fingerprint"] = _fingerprint(path)
    report["repair"] = {
        "status": "unavailable",
        "reason": "stable_targeted_repair_api_unavailable",
        "explicit_apply_required": True,
    }
    return report


def _upstream_integrity_api() -> tuple[Any, Any] | None:
    """Load the optional upstream API without making it a production import.

    The selected composite pin includes ``core.integrity``.  Keeping this
    lookup lazy preserves compatibility with older installations and keeps the
    provider importable when Cashew is absent from a test environment.
    """
    try:
        from core import integrity as upstream_integrity

        inspect_fn = getattr(upstream_integrity, "inspect_integrity", None)
        repair_fn = getattr(upstream_integrity, "repair_integrity", None)
    except (ImportError, AttributeError):
        return None
    if not callable(inspect_fn) or not callable(repair_fn):
        return None
    return inspect_fn, repair_fn


def _repair_unavailable(*, confirm: bool) -> dict[str, Any]:
    """Preserve the fail-closed response for installations without the API."""
    return {
        "schema_version": 1,
        "status": "unavailable",
        "mutated": False,
        "confirmed": bool(confirm),
        "reason": "stable_targeted_repair_api_unavailable",
        "message": (
            "Cashew repair remains unavailable because this installed Cashew "
            "does not provide the connection-aware targeted repair API."
        ),
        "repairs": [],
    }


def _profile_identity(
    conn: sqlite3.Connection,
) -> tuple[str, int] | None:
    """Return the repair identity only when both persisted values are usable."""
    try:
        rows = conn.execute(
            "SELECT key, value FROM hermes_provider_meta "
            "WHERE key IN ('embedding_model', 'embedding_dim')"
        ).fetchall()
        values = {str(key): value for key, value in rows}
        model = str(values["embedding_model"]).strip()
        dimension = int(str(values["embedding_dim"]))
    except (KeyError, TypeError, ValueError, sqlite3.Error):
        return None
    if not model or dimension <= 0:
        return None
    return model, dimension


def _create_verified_backup(
    source_path: pathlib.Path, backup_dir: pathlib.Path
) -> pathlib.Path:
    """Create and verify a snapshot while the caller excludes database writers."""
    backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".cashew-integrity-", suffix=".db", dir=backup_dir, delete=False
    )
    temporary = pathlib.Path(handle.name)
    handle.close()
    target = backup_dir / f"cashew-integrity-{time.time_ns()}.db"
    source_conn: sqlite3.Connection | None = None
    backup_conn: sqlite3.Connection | None = None
    try:
        source_conn = sqlite3.connect(f"{source_path.as_uri()}?mode=ro", uri=True)
        backup_conn = sqlite3.connect(temporary)
        source_conn.backup(backup_conn)
        if str(backup_conn.execute("PRAGMA integrity_check").fetchone()[0]) != "ok":
            raise sqlite3.DatabaseError("backup_integrity_check_failed")
        backup_conn.close()
        backup_conn = None
        source_conn.close()
        source_conn = None
        os.chmod(temporary, 0o600)
        temporary.replace(target)
        return target
    except BaseException:
        if backup_conn is not None:
            backup_conn.close()
        if source_conn is not None:
            source_conn.close()
        temporary.unlink(missing_ok=True)
        raise


def _repair_failure(
    reason: str,
    *,
    backup_path: pathlib.Path | None = None,
    rolled_back: bool = False,
    committed: bool | None = False,
    mutated: bool = False,
    mutation_uncertain: bool = False,
    recovery_required: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "mutated": mutated,
        "confirmed": True,
        "committed": committed,
        "rolled_back": rolled_back,
        "reason": reason,
    }
    if mutation_uncertain:
        report["mutation_uncertain"] = True
    if recovery_required:
        report["recovery_required"] = True
    if backup_path is not None:
        report["backup"] = {"path": str(backup_path), "verified": True}
    return report


def _actionable_count_keys(
    actions: Iterable[str] | None, permanence_policy: str
) -> set[str]:
    selected = (
        {
            "repair_vec",
            "repair_embeddings",
            "remove_orphan_embeddings",
            "remove_orphan_edges",
        }
        if actions is None
        else set(actions)
    )
    keys: set[str] = set()
    mapping = {
        "repair_vec": {"missing_vec", "stale_vec", "invalid_vec", "vec_mismatched"},
        "repair_embeddings": {"missing_embeddings", "invalid_embeddings"},
        "remove_orphan_embeddings": {"orphan_embeddings"},
        "remove_orphan_edges": {"orphan_edges"},
        "remove_self_edges": {"self_edges"},
        "promote_core_memories": {"core_memory_not_permanent"},
    }
    for action in selected:
        keys.update(mapping.get(action, set()))
    if permanence_policy != "report":
        keys.add("permanent_and_decayed")
    return keys


def _repair_verification_matches(
    repair: dict[str, Any],
    inspection: dict[str, Any],
    *,
    actions: Iterable[str] | None,
    permanence_policy: str,
) -> bool:
    """Confirm that upstream's declared outcome matches a fresh inspection."""
    if inspection.get("status") not in {"ok", "findings"}:
        return False
    keys = _actionable_count_keys(actions, permanence_policy)
    counts = inspection.get("counts")
    remaining = repair.get("remaining")
    if not isinstance(counts, dict) or not isinstance(remaining, dict):
        return False
    actual = {key: counts[key] for key in keys if counts.get(key, 0)}
    declared = {key: remaining[key] for key in keys if remaining.get(key, 0)}
    if actual != declared:
        return False
    status = repair.get("status")
    if status == "completed":
        return not actual
    if status == "partial":
        return bool(actual or repair.get("skipped") or repair.get("failures"))
    return False


def _profile_repair_request_error(
    *, deadline_seconds: float, batch_size: int, max_items: int, permanence_policy: str
) -> str | None:
    if not math.isfinite(deadline_seconds) or deadline_seconds < 0:
        return "invalid_deadline"
    if batch_size <= 0 or max_items <= 0:
        return "invalid_repair_bounds"
    if permanence_policy not in {"report", "preserve_permanent", "preserve_decay"}:
        return "invalid_permanence_policy"
    return None


def inspect_integrity(
    conn: sqlite3.Connection,
    *,
    expected_model: str | None = None,
    expected_dimension: int | None = None,
) -> dict[str, Any]:
    """Delegate a connection-owned audit to upstream when available.

    This function never opens, closes, commits, rolls back, or changes journal
    mode on ``conn``.  Model names are adapter-sensitive configuration and are
    removed from the returned upstream report before it crosses this boundary.
    """
    api = _upstream_integrity_api()
    if api is None:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "read_only": True,
            "mutated": False,
            "reason": "stable_targeted_integrity_api_unavailable",
        }
    inspect_fn, _repair_fn = api
    try:
        report = dict(
            inspect_fn(
                conn,
                expected_model=expected_model,
                expected_dimension=expected_dimension,
            )
        )
    except Exception:
        logger.warning("Cashew upstream integrity inspection failed")
        return {
            "schema_version": 1,
            "status": "unavailable",
            "read_only": True,
            "mutated": False,
            "reason": "upstream_integrity_inspection_failed",
        }
    report.pop("expected_model", None)
    report["read_only"] = True
    report["mutated"] = False
    report["adapter"] = {"delegated_to": "core.integrity.inspect_integrity"}
    return report


def apply_integrity_repairs(
    db_path: str | pathlib.Path | None = None,
    *,
    confirm: bool = False,
    backup_dir: str | pathlib.Path | None = None,
    conn: sqlite3.Connection | None = None,
    expected_model: str | None = None,
    expected_dimension: int | None = None,
    embedding_fn: Any = None,
    embedding_model: str | None = None,
    require_vec_parity: bool = True,
    actions: Iterable[str] | None = None,
    permanence_policy: str = "report",
    batch_size: int = 100,
    max_items: int = 1000,
    deadline_seconds: float = 5.0,
) -> dict[str, Any]:
    """Apply an explicit upstream repair through either supported ownership mode.

    With ``conn``, the caller retains connection, transaction, backup, lock,
    commit, and rollback ownership.  With ``db_path``, the adapter runs the
    complete operator workflow: exclusive maintenance admission, compatible
    persisted identity, verified SQLite backup, write-excluding transaction,
    upstream repair, pre-commit verification, commit, and a fresh post-commit
    inspection.  ``confirm`` is required in both modes.
    """
    if not confirm:
        return {
            "schema_version": 1,
            "status": "rejected",
            "mutated": False,
            "confirmed": False,
            "reason": "explicit_confirmation_required",
        }
    if conn is not None and db_path is not None:
        return {
            "schema_version": 1,
            "status": "rejected",
            "mutated": False,
            "confirmed": True,
            "reason": "choose_connection_or_path",
        }
    if conn is None:
        if db_path is None:
            return {
                "schema_version": 1,
                "status": "rejected",
                "mutated": False,
                "confirmed": True,
                "reason": "connection_or_path_required",
            }
        return _apply_profile_repairs(
            pathlib.Path(db_path),
            backup_dir=(None if backup_dir is None else pathlib.Path(backup_dir)),
            expected_model=expected_model,
            expected_dimension=expected_dimension,
            embedding_fn=embedding_fn,
            embedding_model=embedding_model,
            require_vec_parity=require_vec_parity,
            actions=actions,
            permanence_policy=permanence_policy,
            batch_size=batch_size,
            max_items=max_items,
            deadline_seconds=deadline_seconds,
        )
    if not isinstance(conn, sqlite3.Connection):
        return {
            "schema_version": 1,
            "status": "rejected",
            "mutated": False,
            "confirmed": True,
            "reason": "caller_connection_required",
        }
    if not conn.in_transaction:
        return {
            "schema_version": 1,
            "status": "rejected",
            "mutated": False,
            "confirmed": True,
            "reason": "outer_transaction_required",
        }
    api = _upstream_integrity_api()
    if api is None:
        return _repair_unavailable(confirm=True)
    if embedding_model is None:
        embedding_model = expected_model
    _inspect_fn, repair_fn = api
    try:
        result = dict(
            repair_fn(
                conn,
                embedding_fn=embedding_fn,
                embedding_model=embedding_model,
                expected_dimension=expected_dimension,
                require_vec_parity=require_vec_parity,
                actions=actions,
                permanence_policy=permanence_policy,
                batch_size=batch_size,
                max_items=max_items,
            )
        )
    except Exception:
        logger.warning("Cashew upstream integrity repair failed")
        return {
            "schema_version": 1,
            "status": "unavailable",
            "mutated": None,
            "mutation_uncertain": True,
            "confirmed": True,
            "reason": "upstream_integrity_repair_failed",
        }
    result["confirmed"] = True
    result.pop("expected_model", None)
    result["adapter"] = {"delegated_to": "core.integrity.repair_integrity"}
    return result


def _apply_profile_repairs(
    db_path: pathlib.Path,
    *,
    backup_dir: pathlib.Path | None,
    expected_model: str | None,
    expected_dimension: int | None,
    embedding_fn: Any,
    embedding_model: str | None,
    require_vec_parity: bool,
    actions: Iterable[str] | None,
    permanence_policy: str,
    batch_size: int,
    max_items: int,
    deadline_seconds: float,
) -> dict[str, Any]:
    """Own one complete, backup-backed repair transaction for an operator."""
    path = db_path.resolve(strict=False)
    request_error = _profile_repair_request_error(
        deadline_seconds=deadline_seconds,
        batch_size=batch_size,
        max_items=max_items,
        permanence_policy=permanence_policy,
    )
    if request_error is not None:
        return _repair_failure(request_error)
    if not path.is_file():
        return _repair_failure("profile_not_found")
    if _upstream_integrity_api() is None:
        return _repair_unavailable(confirm=True)
    destination = (
        path.parent / "backups"
        if backup_dir is None
        else backup_dir.resolve(strict=False)
    )
    backup_path: pathlib.Path | None = None
    connection: sqlite3.Connection | None = None
    repair_result: dict[str, Any] | None = None
    commit_attempted = False
    commit_succeeded = False
    selected_actions = None if actions is None else tuple(actions)
    try:
        with admit_operation(
            graph_path=path, exclusive=True, deadline=deadline_seconds
        ):
            connection = sqlite3.connect(path, timeout=max(0.0, deadline_seconds))
            identity = _profile_identity(connection)
            if identity is None:
                return _repair_failure("provider_identity_unavailable")
            stored_model, stored_dimension = identity
            if expected_model is not None and expected_model != stored_model:
                return _repair_failure("embedding_model_mismatch")
            if (
                expected_dimension is not None
                and expected_dimension != stored_dimension
            ):
                return _repair_failure("embedding_dimension_mismatch")
            expected_model = stored_model
            expected_dimension = stored_dimension
            if embedding_model is None:
                embedding_model = stored_model

            connection.execute("BEGIN IMMEDIATE")
            backup_path = _create_verified_backup(path, destination)
            repair_result = apply_integrity_repairs(
                conn=connection,
                confirm=True,
                expected_model=expected_model,
                expected_dimension=expected_dimension,
                embedding_fn=embedding_fn,
                embedding_model=embedding_model,
                require_vec_parity=require_vec_parity,
                actions=selected_actions,
                permanence_policy=permanence_policy,
                batch_size=batch_size,
                max_items=max_items,
            )
            if repair_result.get("status") not in {"completed", "partial"}:
                connection.rollback()
                failure = _repair_failure(
                    str(repair_result.get("reason", "repair_failed")),
                    backup_path=backup_path,
                    rolled_back=True,
                )
                failure["repair"] = repair_result
                return failure
            before_commit = inspect_integrity(
                connection,
                expected_model=expected_model,
                expected_dimension=expected_dimension,
            )
            if not _repair_verification_matches(
                repair_result,
                before_commit,
                actions=selected_actions,
                permanence_policy=permanence_policy,
            ):
                connection.rollback()
                return _repair_failure(
                    "precommit_verification_failed",
                    backup_path=backup_path,
                    rolled_back=True,
                )
            commit_attempted = True
            connection.commit()
            commit_succeeded = True
            after_commit = inspect_integrity(
                connection,
                expected_model=expected_model,
                expected_dimension=expected_dimension,
            )
            if not _repair_verification_matches(
                repair_result,
                after_commit,
                actions=selected_actions,
                permanence_policy=permanence_policy,
            ):
                return _repair_failure(
                    "postcommit_verification_failed",
                    backup_path=backup_path,
                    committed=True,
                    mutated=bool(repair_result.get("mutated")),
                    mutation_uncertain=True,
                    recovery_required=True,
                )
            repair_result["committed"] = True
            repair_result["rolled_back"] = False
            repair_result["backup"] = {"path": str(backup_path), "verified": True}
            repair_result["verification"] = {
                "before_commit": before_commit,
                "after_commit": after_commit,
            }
            repair_result["operator_workflow"] = {
                "exclusive_admission": True,
                "sqlite_write_exclusion": "BEGIN IMMEDIATE",
                "embedding_cache": "unchanged_same_model_content_cache",
                "maintenance_epoch": "preserved_same_identity",
            }
            return repair_result
    except Exception:
        if connection is not None and connection.in_transaction:
            connection.rollback()
            rolled_back = True
            committed: bool | None = False
            mutation_uncertain = False
        else:
            rolled_back = False
            committed = (
                True if commit_succeeded else (None if commit_attempted else False)
            )
            mutation_uncertain = bool(commit_attempted and not commit_succeeded)
        logger.warning("Cashew operator integrity repair failed")
        return _repair_failure(
            "operator_repair_failed",
            backup_path=backup_path,
            rolled_back=rolled_back,
            committed=committed,
            mutated=bool(
                committed is True
                and repair_result is not None
                and repair_result.get("mutated")
            ),
            mutation_uncertain=mutation_uncertain,
            recovery_required=bool(committed is not False),
        )
    finally:
        if connection is not None:
            connection.close()


# Short operator-friendly names; the explicit names remain canonical for callers.
audit = audit_integrity
apply = apply_integrity_repairs


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit an existing Cashew database safely."
    )
    parser.add_argument("db_path", type=pathlib.Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run the backup-backed operator repair workflow.",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Confirm an explicit repair request."
    )
    parser.add_argument(
        "--backup-dir",
        type=pathlib.Path,
        help="Backup destination (default: a backups directory beside the database).",
    )
    parser.add_argument(
        "--action",
        action="append",
        dest="actions",
        help="Upstream repair action; repeat to select multiple actions.",
    )
    parser.add_argument(
        "--permanence-policy",
        choices=("report", "preserve_permanent", "preserve_decay"),
        default="report",
        help="Explicit resolution for contradictory permanence state.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--deadline-seconds", type=float, default=5.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = (
        apply_integrity_repairs(
            args.db_path,
            confirm=args.confirm,
            backup_dir=args.backup_dir,
            actions=args.actions,
            permanence_policy=args.permanence_policy,
            batch_size=args.batch_size,
            max_items=args.max_items,
            deadline_seconds=args.deadline_seconds,
        )
        if args.apply
        else audit_integrity(args.db_path)
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    successful = (
        {"completed", "partial"}
        if args.apply
        else {"ok", "findings", "audit_incomplete", "unavailable"}
    )
    return 0 if report.get("status") in successful else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
