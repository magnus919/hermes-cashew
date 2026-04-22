# tests/test_migration.py
# Phase 8: Schema migration tests — v0.1.0 → v0.2.0 transparent upgrade

from __future__ import annotations

import sqlite3

import pytest

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
        assert "reasoning" in cols, "SCHEMA-01: reasoning column missing"
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
        assert "reasoning" in cols
        conn.close()
    finally:
        p2.shutdown()


def test_no_columns_dropped(tmp_path):
    """SCHEMA-06: Migration is strictly additive — no columns are dropped."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    # Pre-migration column count
    conn = sqlite3.connect(str(db_path))
    pre_cols = _get_columns(conn, "thought_nodes")
    pre_count = len(pre_cols)
    conn.close()

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        post_cols = _get_columns(conn, "thought_nodes")
        post_count = len(post_cols)
        assert post_count >= pre_count, (
            f"Column count decreased: {pre_count} → {post_count}. Migration dropped columns."
        )
        # Verify all original columns still exist
        assert "id" in post_cols
        assert "content" in post_cols
        assert "node_type" in post_cols
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


def test_vec_embeddings_guarded_when_unavailable(tmp_path, caplog):
    """SCHEMA-04: When sqlite-vec is unavailable, initialize() succeeds and logs DEBUG."""
    p = CashewMemoryProvider()
    with caplog.at_level("DEBUG", logger="plugins.memory.cashew"):
        p.initialize("s", hermes_home=str(tmp_path))
    try:
        # Must succeed even though sqlite-vec is not installed in test environment
        assert p._config is not None, "initialize() must succeed when sqlite-vec is unavailable"
        assert p._db_path is not None
        debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert any(
            "sqlite-vec not available" in r.getMessage() for r in debugs
        ), (
            f"Expected DEBUG about sqlite-vec unavailability; got: "
            f"{[r.getMessage() for r in debugs]}"
        )
    finally:
        p.shutdown()


def test_existing_data_preserved(tmp_path):
    """SCHEMA-06: All existing row data is preserved after migration."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_v0_1_0_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence) "
        "VALUES ('n1', 'hello world', 'fact', 'test-domain', '2024-01-01T00:00:00', 0.95)"
    )
    conn.commit()
    conn.close()

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT id, content, node_type, domain, timestamp, confidence "
            "FROM thought_nodes WHERE id = 'n1'"
        ).fetchone()
        assert row == ("n1", "hello world", "fact", "test-domain", "2024-01-01T00:00:00", 0.95), (
            f"Existing data was corrupted during migration: {row}"
        )
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
        assert "reasoning" in cols
        assert "mood_state" in cols
        assert "metadata" in cols
        assert "permanent" in cols
        assert "last_updated" in cols
        assert "last_accessed" in cols
        assert "access_count" in cols
        assert "tags" in cols
        assert "referent_time" in cols
        conn.close()
    finally:
        p.shutdown()
