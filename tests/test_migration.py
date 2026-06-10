# tests/test_migration.py
# Phase 8: Schema migration tests — v0.1.0 → v0.2.0 transparent upgrade

from __future__ import annotations

import sqlite3

from plugins.memory.cashew import CashewMemoryProvider


def _make_v0_1_0_db(db_path):
    """Create a v0.1.0 schema DB (missing v0.2.0 columns)."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            confidence REAL,
            source_file TEXT,
            decayed INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            confidence REAL,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id)
        )
    """)
    conn.execute("""
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    """)
    conn.commit()
    conn.close()


def _get_columns(conn, table_name):
    """Return set of column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def test_v0_1_0_columns_added(tmp_path):
    """SCHEMA-01 through SCHEMA-03, SCHEMA-07 through SCHEMA-09:
    A v0.1.0 DB gains all missing columns after initialize()."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        cols = _get_columns(conn, "thought_nodes")
        assert "mood_state" in cols, "SCHEMA-02: mood_state column missing"
        assert "metadata" in cols, "SCHEMA-03: metadata column missing"
        assert "permanent" in cols, "SCHEMA-09: permanent column missing"
        assert "last_updated" in cols, "SCHEMA-07: last_updated column missing"
        assert "last_accessed" in cols, "SCHEMA-08: last_accessed column missing"
        assert "access_count" in cols, "access_count column missing"
        assert "tags" in cols, "tags column missing"
        assert "referent_time" in cols, "referent_time column missing"
        conn.close()
    finally:
        p.shutdown()


def test_migration_idempotent(tmp_path):
    """SCHEMA-05: Running initialize() twice on an already-migrated DB must not raise."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    p1 = CashewMemoryProvider()
    p1.initialize("s", hermes_home=str(tmp_path))
    p1.shutdown()

    p2 = CashewMemoryProvider()
    p2.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        cols = _get_columns(conn, "thought_nodes")
        assert "metadata" in cols
        conn.close()
    finally:
        p2.shutdown()


def test_core_columns_preserved(tmp_path):
    """SCHEMA-06: Core data columns are preserved after migration.
    Upstream v1.1.0 drops the dead confidence column (uncalibrated noise per
    cashew-brain PR #25) but preserves all meaningful data columns.
    """
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    # Pre-migration: confirm confidence exists in old schema
    conn = sqlite3.connect(str(db_path))
    pre_cols = _get_columns(conn, "thought_nodes")
    assert "confidence" in pre_cols
    conn.close()

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        post_cols = _get_columns(conn, "thought_nodes")
        # confidence is intentionally dropped by upstream
        assert "id" in post_cols
        assert "content" in post_cols
        assert "node_type" in post_cols
        assert "domain" in post_cols
        assert "timestamp" in post_cols
        conn.close()
    finally:
        p.shutdown()


def test_metadata_backfill(tmp_path):
    """D-06 / SCHEMA-03: Existing rows receive DEFAULT backfill for metadata."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type) VALUES ('n1', 'hello', 'fact')"
    )
    conn.commit()
    conn.close()

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT metadata, permanent, access_count FROM thought_nodes WHERE id = 'n1'"
        ).fetchone()
        assert row[0] == "{}", f"metadata backfill failed: got {row[0]!r}"
        assert row[1] == 0, f"permanent backfill failed: got {row[1]!r}"
        assert row[2] == 0, f"access_count backfill failed: got {row[2]!r}"
        conn.close()
    finally:
        p.shutdown()


def test_existing_data_preserved(tmp_path):
    """SCHEMA-06: Existing row data is preserved after migration (confidence column excluded —
    upstream v1.1.0 drops it intentionally per cashew-brain PR #25)."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp) "
        "VALUES ('n1', 'hello world', 'fact', 'test-domain', '2024-01-01T00:00:00')"
    )
    conn.commit()
    conn.close()

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT id, content, node_type, domain, timestamp "
            "FROM thought_nodes WHERE id = 'n1'"
        ).fetchone()
        assert row == (
            "n1",
            "hello world",
            "fact",
            "test-domain",
            "2024-01-01T00:00:00",
        ), f"Existing data was corrupted during migration: {row}"
        conn.close()
    finally:
        p.shutdown()


def test_fresh_db_has_all_columns(tmp_path):
    """Fresh DB (no prior schema) gets the complete v0.2.0 schema on first initialize()."""
    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        import pathlib

        db_path = pathlib.Path(str(tmp_path)) / "cashew" / "brain.db"
        conn = sqlite3.connect(str(db_path))
        cols = _get_columns(conn, "thought_nodes")
        assert "mood_state" in cols
        assert "metadata" in cols
        assert "permanent" in cols
        assert "last_updated" in cols
        assert "last_accessed" in cols
        assert "access_count" in cols
        assert "tags" in cols
        assert "referent_time" in cols
        assert "domain" in cols
        assert "timestamp" in cols
        conn.close()
    finally:
        p.shutdown()
