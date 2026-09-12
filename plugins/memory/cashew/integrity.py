"""Read-only integrity inspection for existing Cashew profiles.

This module deliberately stops at an audit boundary.  Cashew's currently
available repair helpers are broad, default-service based, and do not expose a
stable connection-aware transaction contract.  ``apply_integrity_repairs``
therefore returns a structured unavailable result until upstream provides that
contract; it never mutates a profile as a side effect of inspection.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
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


def apply_integrity_repairs(
    db_path: str | pathlib.Path,
    *,
    confirm: bool = False,
    backup_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Return a structured refusal until upstream exposes safe targeted repair.

    ``db_path`` and ``backup_dir`` are accepted to make the future operator
    contract explicit.  They are intentionally not opened or created here.
    """
    del db_path, backup_dir
    return {
        "schema_version": 1,
        "status": "unavailable",
        "mutated": False,
        "confirmed": bool(confirm),
        "reason": "stable_targeted_repair_api_unavailable",
        "message": (
            "Cashew repair remains unavailable until upstream provides a "
            "connection-aware targeted repair API with atomic ordinary/vec writes."
        ),
        "repairs": [],
    }


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
        help="Report repair availability (never implicit).",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Confirm an explicit repair request."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = (
        apply_integrity_repairs(args.db_path, confirm=args.confirm)
        if args.apply
        else audit_integrity(args.db_path)
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return (
        0
        if report.get("status") in {"ok", "findings", "audit_incomplete", "unavailable"}
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
