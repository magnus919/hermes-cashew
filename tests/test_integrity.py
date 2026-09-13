"""Contract tests for the query-only #206 integrity boundary."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path

import pytest

import plugins.memory.cashew.integrity as integrity
from plugins.memory.cashew.integrity import (
    apply_integrity_repairs,
    audit_integrity,
)
from plugins.memory.cashew.locking import (
    SQLiteWALUnsupportedError,
    open_readonly_verified,
    verify_readonly_profile,
)


def _create_profile(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA user_version = 3;
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY, content TEXT, node_type TEXT, domain TEXT,
            timestamp TEXT, access_count INTEGER, last_accessed TEXT,
            source_file TEXT, decayed INTEGER, metadata TEXT, last_updated TEXT,
            mood_state TEXT, permanent INTEGER, tags TEXT, referent_time TEXT
        );
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT
        );
        CREATE TABLE derivation_edges (
            parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, timestamp TEXT
        );
        CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO hermes_provider_meta VALUES
            ('embedding_model', 'model-a'),
            ('embedding_dim', '4'),
            ('vec_dim', '4'),
            ('maintenance_epoch', '7');
        """
    )
    conn.commit()
    conn.close()


def _add_node(
    conn: sqlite3.Connection, node_id: str, *, permanent: int = 0, decayed: int = 0
) -> None:
    conn.execute(
        "INSERT INTO thought_nodes VALUES (?, ?, 'fact', 'user', '2026-09-12', 0, "
        "NULL, NULL, ?, '{}', NULL, NULL, ?, NULL, NULL)",
        (node_id, f"content-{node_id}", decayed, permanent),
    )


def _snapshot(path: Path) -> dict[str, tuple[int, str]]:
    result = {}
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            result[str(candidate)] = (
                candidate.stat().st_size,
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
    return result


def test_audit_reports_findings_and_preserves_profile_files(tmp_path: Path) -> None:
    path = tmp_path / "brain.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    _add_node(conn, "valid")
    _add_node(conn, "missing")
    _add_node(conn, "permanent-decayed", permanent=1, decayed=1)
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
        ("valid", struct.pack("<4f", 1.0, 0.0, 0.0, 0.0), "model-a"),
    )
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
        ("orphan", struct.pack("<4f", 0.0, 0.0, 0.0, 0.0), "wrong-model"),
    )
    conn.execute(
        "INSERT INTO derivation_edges VALUES ('valid', 'valid', 1.0, 'self', 'now')"
    )
    conn.execute(
        "INSERT INTO derivation_edges VALUES ('absent', 'valid', 1.0, 'orphan', 'now')"
    )
    conn.commit()
    before = _snapshot(path)

    report = audit_integrity(path)

    assert report["read_only"] is True
    assert report["mutated"] is False
    assert report["status"] == "findings"
    assert report["provenance"]["provider_model_fingerprint"]
    assert "model-a" not in json.dumps(report)
    assert report["provenance"]["provider_epoch_present"] is True
    assert report["provenance"]["provider_embedding_dim"] == 4
    assert report["counts"]["orphan_embeddings"] == 1
    assert report["counts"]["nodes_without_embeddings"] == 2
    assert report["reasons"]["embedding_zero_norm"] == 1
    assert report["reasons"]["embedding_model_mismatch"] == 1
    assert report["reasons"]["permanent_and_decayed"] == 1
    assert report["reasons"]["vec_index_missing"] == 2
    assert report["reasons"]["orphan_edge"] == 1
    assert report["reasons"]["self_edge"] == 1
    assert report["uncertainty"] == ["historical_consolidation"]
    assert _snapshot(path) == before


def test_audit_privacy_canary_is_absent_from_report_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "SECRET-CANARY-profile.db"
    _create_profile(path)
    monkeypatch.setattr(
        integrity,
        "_inspect_profile",
        lambda *args: (_ for _ in ()).throw(RuntimeError("SECRET-CANARY-model")),
    )
    report = audit_integrity(path)
    assert "SECRET-CANARY" not in json.dumps(report, sort_keys=True)
    assert "SECRET-CANARY" not in caplog.text


def test_permanent_core_memory_is_informational(tmp_path: Path) -> None:
    path = tmp_path / "core.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    _add_node(conn, "core", permanent=1)
    conn.execute("UPDATE thought_nodes SET node_type='core_memory' WHERE id='core'")
    conn.commit()
    conn.close()
    report = audit_integrity(path)
    assert report["informational"]["permanent_core_nodes"] == 1
    assert "permanent_core_node" not in report["reasons"]


def test_audit_reports_fixed_row_and_byte_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "budget.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    for index in range(3):
        _add_node(conn, f"node-{index}")
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
            (f"node-{index}", struct.pack("<4f", 1.0, 0.0, 0.0, 0.0), "model-a"),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(integrity, "_MAX_AUDIT_ROWS", 1)
    report = audit_integrity(path)
    assert report["status"] == "audit_incomplete"
    assert report["reasons"]["audit_row_cap"] == 1
    assert "schema_table_missing" not in report["reasons"]
    assert "provider_identity_missing" not in report["reasons"]
    assert "profile_verification_failed" not in report["reasons"]
    assert report["limits"]["rows_scanned"] == 1
    assert report["limits"]["complete"] is False
    assert report["completeness"] == {}


def test_audit_marks_graph_scan_incomplete_without_leaking_schema_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "graph-budget.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    for index in range(4):
        _add_node(conn, f"n{index}")
        conn.execute(
            "INSERT INTO derivation_edges VALUES (?, ?, 1.0, 'edge', 'now')",
            (f"n{index}", "missing-node"),
        )
    conn.execute('CREATE TABLE "SECRET-CANARY-schema" (payload TEXT)')
    conn.commit()
    conn.close()
    monkeypatch.setattr(integrity, "_MAX_AUDIT_ROWS", 2)

    report = audit_integrity(path)

    encoded = json.dumps(report, sort_keys=True)
    assert report["status"] == "audit_incomplete"
    assert report["limits"]["complete"] is False
    assert "SECRET-CANARY-schema" not in encoded
    assert report["schema"] == {}


def test_audit_reports_aggregate_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "byte-budget.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    _add_node(conn, "node")
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
        ("node", struct.pack("<4f", 1.0, 0.0, 0.0, 0.0), "model-a"),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(integrity, "_MAX_AUDIT_BYTES", 8)
    report = audit_integrity(path)
    assert report["status"] == "audit_incomplete"
    assert report["reasons"]["audit_byte_cap"] == 1
    assert report["limits"]["bytes_scanned"] == 0


def test_audit_deadline_is_explicitly_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "deadline.db"
    _create_profile(path)
    report = audit_integrity(path, deadline_seconds=0)
    assert report["status"] == "audit_incomplete"
    assert report["reasons"]["audit_deadline"] >= 1
    assert "schema_table_missing" not in report["reasons"]
    assert "provider_identity_missing" not in report["reasons"]
    assert "profile_verification_failed" not in report["reasons"]


def test_audit_installs_deadline_before_profile_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "verify-deadline.db"
    _create_profile(path)
    observed: list[bool] = []
    real_open = integrity.open_readonly_verified

    class _TrackingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def set_progress_handler(self, callback, interval) -> None:
            observed.append(callback is not None and interval == 1000)
            self._connection.set_progress_handler(callback, interval)

        def close(self) -> None:
            self._connection.close()

    def open_tracking(snapshot: Path):
        connection, mode = real_open(snapshot)
        return _TrackingConnection(connection), mode

    def verify_tracking(connection, *, budget):
        assert any(observed)
        budget.incomplete_reasons.add("audit_deadline")

    monkeypatch.setattr(integrity, "open_readonly_verified", open_tracking)
    monkeypatch.setattr(integrity, "verify_readonly_profile", verify_tracking)
    monkeypatch.setattr(
        integrity,
        "_inspect_profile",
        lambda _conn, _mode, _budget: {"reasons": {"audit_deadline": 1}},
    )

    report = audit_integrity(path)

    assert report["status"] == "audit_incomplete"
    assert any(observed)


def test_profile_verification_preserves_non_deadline_budget_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "verify-row-cap.db"
    _create_profile(path)

    def fail_at_row_cap(_connection, *, budget):
        budget.incomplete_reasons.add("audit_row_cap")
        raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(integrity, "verify_readonly_profile", fail_at_row_cap)
    report = audit_integrity(path)

    assert report["reasons"] == {"audit_row_cap": 1}


def test_audit_loaded_vec_parity_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "vec.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding float[4])"
    )
    _add_node(conn, "vec-node")
    blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
        ("vec-node", blob, "model-a"),
    )
    conn.execute("INSERT INTO vec_embeddings VALUES (?, ?)", ("vec-node", blob))
    conn.commit()
    conn.close()
    before = _snapshot(path)
    report = audit_integrity(path)
    assert report["vector_index"]["available"] is True
    assert report["vector_index"]["entries"] == 1
    assert "vec_entry_missing" not in report["reasons"]
    assert _snapshot(path) == before


def test_incomplete_vec_scan_does_not_report_parity(tmp_path: Path) -> None:
    path = tmp_path / "partial-vec.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding float[4])"
    )
    blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    conn.execute("INSERT INTO vec_embeddings VALUES (?, ?)", ("one", blob))
    conn.execute("INSERT INTO vec_embeddings VALUES (?, ?)", ("two", blob))
    conn.commit()
    budget = integrity._AuditBudget(
        deadline=time.monotonic() + 5,
        rows=integrity._MAX_AUDIT_ROWS - 1,
    )
    reasons: Counter[str] = Counter()

    result = integrity._inspect_vec(conn, {"one", "two", "three"}, 4, reasons, budget)

    conn.close()
    assert result["scan_complete"] is False
    assert result["missing_entries"] is None
    assert result["stale_entries"] is None
    assert "vec_entry_missing" not in reasons


def test_missing_expected_model_is_unverifiable_not_a_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "missing-model.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    _add_node(conn, "node")
    conn.execute("DELETE FROM hermes_provider_meta WHERE key='embedding_model'")
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
        ("node", struct.pack("<4f", 1.0, 0.0, 0.0, 0.0), "model-a"),
    )
    conn.commit()
    conn.close()

    report = audit_integrity(path)

    assert "embedding_model_mismatch" not in report["reasons"]


def test_audit_loaded_vec_wal_sidecars_remain_byte_identical(tmp_path: Path) -> None:
    path = tmp_path / "vec-wal.db"
    _create_profile(path)
    writer = sqlite3.connect(path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        writer.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(writer)
        writer.enable_load_extension(False)
        writer.execute(
            "CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding float[4])"
        )
        _add_node(writer, "wal-vec")
        blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        writer.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
            ("wal-vec", blob, "model-a"),
        )
        writer.execute("INSERT INTO vec_embeddings VALUES (?, ?)", ("wal-vec", blob))
        writer.commit()
        before = _snapshot(path)

        report = audit_integrity(path)

        assert report["vector_index"]["available"] is True
        assert _snapshot(path) == before
        assert set(before) >= {str(path), f"{path}-wal", f"{path}-shm"}
    finally:
        writer.close()


def test_audit_vec_unavailable_fallback_preserves_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vec-unavailable.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE vec_embeddings (node_id TEXT, embedding BLOB)")
    conn.commit()
    conn.close()
    before = _snapshot(path)
    monkeypatch.setattr(integrity, "_load_vec_readonly", lambda conn: False)
    report = audit_integrity(path)
    assert report["vector_index"]["available"] is False
    assert report["reasons"]["vec_index_unverifiable"] == 1
    assert _snapshot(path) == before


def test_audit_plain_table_named_vec_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plain-vec.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE vec_embeddings (node_id TEXT, embedding BLOB)")
    conn.commit()
    conn.close()
    # A successful module load must not make an ordinary table look like vec0.
    monkeypatch.setattr(integrity, "_load_vec_readonly", lambda conn: True)

    report = audit_integrity(path)

    assert report["vector_index"]["available"] is False
    assert report["vector_index"]["scan_complete"] is False
    assert report["reasons"]["vec_index_unverifiable"] == 1


def test_audit_deceptive_vec_text_never_reaches_vec_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "deceptive-vec.db"
    _create_profile(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE vec_embeddings (node_id TEXT DEFAULT 'USING vec0', embedding BLOB)"
    )
    conn.commit()
    conn.close()

    def fail_loader(_conn: sqlite3.Connection) -> bool:
        raise AssertionError("deceptive ordinary table reached vec loader")

    monkeypatch.setattr(integrity, "_load_vec_readonly", fail_loader)

    report = audit_integrity(path)

    assert report["vector_index"]["available"] is False
    assert report["reasons"]["vec_index_unverifiable"] == 1


def test_readonly_verifier_rejects_deceptive_vec_text_before_loading(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deceptive-verifier.db"
    _create_profile(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE vec_embeddings (node_id TEXT DEFAULT 'USING vec0', embedding BLOB)"
        )

    readonly, _mode = open_readonly_verified(path)
    try:
        with pytest.raises(SQLiteWALUnsupportedError, match="declaration"):
            verify_readonly_profile(readonly)
    finally:
        readonly.close()


def test_audit_vec_unavailable_wal_keeps_sidecars_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vec-unavailable-wal.db"
    _create_profile(path)
    writer = sqlite3.connect(path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        writer.execute("CREATE TABLE vec_embeddings (node_id TEXT, embedding BLOB)")
        _add_node(writer, "wal-ordinary")
        writer.commit()
        before = _snapshot(path)
        monkeypatch.setattr(integrity, "_load_vec_readonly", lambda conn: False)

        report = audit_integrity(path)

        assert report["vector_index"]["available"] is False
        assert _snapshot(path) == before
        assert set(before) >= {str(path), f"{path}-wal", f"{path}-shm"}
    finally:
        writer.close()


def test_audit_keeps_live_wal_and_shm_bytes_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "wal.db"
    _create_profile(path)
    writer = sqlite3.connect(path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        _add_node(writer, "wal-node")
        writer.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, '2026-09-12')",
            ("wal-node", struct.pack("<4f", 1.0, 0.0, 0.0, 0.0), "model-a"),
        )
        writer.commit()
        before = _snapshot(path)
        real_connect = sqlite3.connect
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def readonly_connect(*args, **kwargs):
            calls.append((args, kwargs))
            assert kwargs.get("uri") is True
            assert "mode=ro" in str(args[0])
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", readonly_connect)
        report = audit_integrity(path)

        assert report["read_only"] is True
        assert calls
        assert _snapshot(path) == before
    finally:
        writer.close()


def test_audit_rejects_incomplete_schema_without_mutating(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 3")
    conn.execute("CREATE TABLE thought_nodes (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    report = audit_integrity(path)

    assert report["status"] == "findings"
    assert report["read_only"] is True
    assert report["mutated"] is False
    assert report["schema"]["missing_tables"]
    assert "schema_table_missing" in report["reasons"]


def test_apply_is_explicitly_unavailable_and_does_not_open_database(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "does-not-exist.db"

    def fail_connect(*args, **kwargs):
        raise AssertionError(
            "apply must not open a profile before stable repair exists"
        )

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    report = apply_integrity_repairs(
        path, confirm=True, backup_dir=tmp_path / "backups"
    )

    assert report == {
        "schema_version": 1,
        "status": "unavailable",
        "mutated": False,
        "confirmed": True,
        "reason": "stable_targeted_repair_api_unavailable",
        "message": (
            "Cashew repair remains unavailable until upstream provides a "
            "connection-aware targeted repair API with atomic ordinary/vec writes."
        ),
        "repairs": [],
    }
    assert not path.exists()


def test_legacy_apply_without_confirmation_preserves_unavailable_envelope(
    tmp_path: Path,
) -> None:
    """The pre-contract pin stays unavailable for every legacy path call."""
    report = apply_integrity_repairs(tmp_path / "does-not-exist.db")

    assert report == {
        "schema_version": 1,
        "status": "unavailable",
        "mutated": False,
        "confirmed": False,
        "reason": "stable_targeted_repair_api_unavailable",
        "message": (
            "Cashew repair remains unavailable until upstream provides a "
            "connection-aware targeted repair API with atomic ordinary/vec writes."
        ),
        "repairs": [],
    }
