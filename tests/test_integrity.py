"""Contract tests for the query-only #206 integrity boundary."""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from pathlib import Path

from plugins.memory.cashew.integrity import (
    apply_integrity_repairs,
    audit_integrity,
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
    conn.commit()
    before = _snapshot(path)

    report = audit_integrity(path)

    assert report["read_only"] is True
    assert report["mutated"] is False
    assert report["status"] == "findings"
    assert report["provenance"]["provider_model"] == "model-a"
    assert report["provenance"]["provider_embedding_dim"] == 4
    assert report["counts"]["orphan_embeddings"] == 1
    assert report["counts"]["nodes_without_embeddings"] == 2
    assert report["reasons"]["embedding_zero_norm"] == 1
    assert report["reasons"]["embedding_model_mismatch"] == 1
    assert report["reasons"]["permanent_and_decayed"] == 1
    assert report["reasons"]["vec_index_missing"] == 2
    assert report["uncertainty"] == ["historical_consolidation"]
    assert _snapshot(path) == before


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
