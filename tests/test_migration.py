# tests/test_migration.py
# Phase 8: Schema migration tests — v0.1.0 → v0.2.0 transparent upgrade

from __future__ import annotations

import fcntl
import pathlib
import sqlite3

import numpy as np

from plugins.memory.cashew import CashewMemoryProvider, _patch_upstream_embedding
from plugins.memory.cashew.config import CashewConfig


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


def _load_sqlite_vec(conn):
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)


def _make_dimension_mismatch_db(db_path):
    """Create canonical tables with both stores fixed at the old 384 dimension."""
    from core.db import ensure_schema

    ensure_schema(str(db_path))
    conn = sqlite3.connect(str(db_path))
    _load_sqlite_vec(conn)
    conn.execute("DROP TABLE IF EXISTS vec_embeddings")
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id TEXT primary key, "
        "embedding float[384] distance_metric=cosine)"
    )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp) "
        "VALUES ('n1', 'dimension migration', 'fact', 'test', '2026-01-01')"
    )
    vector = np.ones(384, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    vector_bytes = vector.tobytes()
    conn.execute(
        "INSERT INTO embeddings (node_id, vector, model, updated_at) "
        "VALUES ('n1', ?, 'all-MiniLM-L6-v2', '2026-01-01')",
        (vector_bytes,),
    )
    conn.execute(
        "INSERT INTO vec_embeddings (node_id, embedding) VALUES ('n1', ?)",
        (vector_bytes,),
    )
    conn.commit()
    conn.close()


def _fake_migrate_to_1024(db_path, *, confirm, quiet):
    assert confirm is True
    assert quiet is True
    conn = sqlite3.connect(str(db_path))
    _load_sqlite_vec(conn)
    conn.execute("DELETE FROM embeddings")
    conn.execute("DROP TABLE vec_embeddings")
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id TEXT primary key, "
        "embedding float[1024] distance_metric=cosine)"
    )
    vector = np.ones(1024, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    vector_bytes = vector.tobytes()
    conn.execute(
        "INSERT INTO embeddings (node_id, vector, model, updated_at) "
        "VALUES ('n1', ?, 'thenlper/gte-large', '2026-01-02')",
        (vector_bytes,),
    )
    conn.execute(
        "INSERT INTO vec_embeddings (node_id, embedding) VALUES ('n1', ?)",
        (vector_bytes,),
    )
    conn.commit()
    conn.close()
    return {"nodes_embedded": 1}


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


def test_initialize_backs_up_and_migrates_embedding_dimension(tmp_path, monkeypatch):
    """A 384-dim brain is backed up and becomes a usable 1024-dim index."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", _fake_migrate_to_1024
    )

    p = CashewMemoryProvider()
    p.save_config({"embedding_model": "thenlper/gte-large"}, str(tmp_path))
    p.initialize("migrate", hermes_home=str(tmp_path))
    try:
        assert p._config is not None
        assert p._embedding_dimensions(db_path) == ({1024}, 1024)
        backups = list((db_path.parent / "backups").glob("graph.db.*"))
        assert len(backups) == 1

        conn = sqlite3.connect(str(db_path))
        _load_sqlite_vec(conn)
        query = np.ones(1024, dtype=np.float32)
        query /= np.linalg.norm(query)
        row = conn.execute(
            "SELECT node_id, distance FROM vec_embeddings "
            "WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
            (query.tobytes(),),
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "n1"
        assert row[1] == 0.0
    finally:
        p.shutdown()


def test_failed_embedding_migration_restores_backup(tmp_path, monkeypatch, caplog):
    """A destructive upstream failure restores both embedding stores."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)

    def destructive_failure(db_path, *, confirm, quiet):
        del confirm, quiet
        conn = sqlite3.connect(str(db_path))
        _load_sqlite_vec(conn)
        conn.execute("DELETE FROM embeddings")
        conn.execute("DROP TABLE vec_embeddings")
        conn.commit()
        conn.close()
        raise RuntimeError("synthetic migration failure")

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", destructive_failure
    )
    p = CashewMemoryProvider()
    p._config = CashewConfig(embedding_model="thenlper/gte-large")

    p._repair_embedding_dimension(db_path)

    assert p._embedding_dimensions(db_path) == ({384}, 384)
    conn = sqlite3.connect(str(db_path))
    assert conn.execute(
        "SELECT content FROM thought_nodes WHERE id='n1'"
    ).fetchone() == ("dimension migration",)
    conn.close()
    assert "restoring pre-migration backup" in caplog.text


def test_embedding_migration_defers_while_sleep_lock_is_held(
    tmp_path, monkeypatch, caplog
):
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    called = False

    def unexpected_migration(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", unexpected_migration
    )
    lock_fd = pathlib.Path(f"{db_path}.sleep.lock").open("a+")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        p = CashewMemoryProvider()
        p._config = CashewConfig(embedding_model="thenlper/gte-large")
        p._repair_embedding_dimension(db_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    assert called is False
    assert p._embedding_dimensions(db_path) == ({384}, 384)
    assert "migration deferred" in caplog.text


def test_upstream_embedding_configuration_is_idempotent():
    import core.config
    import core.embeddings

    embed_nodes = core.embeddings.embed_nodes
    try:
        _patch_upstream_embedding("BAAI/bge-small-en-v1.5")
        _patch_upstream_embedding("BAAI/bge-small-en-v1.5")
        assert core.config.config.embedding_model == "BAAI/bge-small-en-v1.5"
        assert core.embeddings.embed_nodes is embed_nodes
    finally:
        _patch_upstream_embedding("thenlper/gte-large")
