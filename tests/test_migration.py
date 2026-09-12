# tests/test_migration.py
# Phase 8: Schema migration tests — v0.1.0 → v0.2.0 transparent upgrade

from __future__ import annotations

import fcntl
import importlib
import json
import pathlib
import sqlite3
import types

import numpy as np
import pytest

import plugins.memory.cashew as cashew_module
from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig
from plugins.memory.cashew.cron_reconcile import cron_prompt, profile_identity


@pytest.fixture(autouse=True)
def _verified_child_handshake(request, monkeypatch: pytest.MonkeyPatch):
    """Keep migration assertions offline while preserving the handshake contract."""

    if request.node.get_closest_marker("real_embedding_child"):
        yield
        return

    class FakeSupervisor:
        dimension = 1024

        def __init__(self, *, dimension: int, **_kwargs) -> None:
            self.dimension = dimension or self.dimension

        def start(self) -> int:
            return self.dimension

        def serve_generation(self):
            from contextlib import nullcontext

            return nullcontext()

        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), self.dimension), dtype=np.float32)

        def _when_closed(self, callback):
            callback()

        def close(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    yield


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


def _logical_embedding_snapshot(db_path):
    """Capture migration-relevant SQLite data without comparing file bytes."""
    conn = sqlite3.connect(str(db_path))
    try:
        _load_sqlite_vec(conn)
        return {
            "nodes": conn.execute(
                "SELECT id, content, node_type, domain, timestamp FROM thought_nodes ORDER BY id"
            ).fetchall(),
            "embeddings": conn.execute(
                "SELECT node_id, vector, model, updated_at FROM embeddings ORDER BY node_id"
            ).fetchall(),
            "vec": conn.execute(
                "SELECT node_id, embedding FROM vec_embeddings ORDER BY node_id"
            ).fetchall(),
            "vec_schema": conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
            ).fetchone(),
        }
    finally:
        conn.close()


def _make_legacy_vec_schema(db_path):
    """Replace the canonical vec table with the pre-node-id layout."""
    conn = sqlite3.connect(str(db_path))
    try:
        _load_sqlite_vec(conn)
        conn.execute("DROP TABLE vec_embeddings")
        conn.execute(
            "CREATE VIRTUAL TABLE vec_embeddings USING vec0(embedding float[384])"
        )
        vector = np.ones(384, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        conn.execute(
            "INSERT INTO vec_embeddings (embedding) VALUES (?)", (vector.tobytes(),)
        )
        conn.commit()
    finally:
        conn.close()


def _logical_legacy_vec_snapshot(db_path):
    """Compare legacy vec state by schema and blob rows, never file bytes."""
    conn = sqlite3.connect(str(db_path))
    try:
        _load_sqlite_vec(conn)
        return {
            "embeddings": conn.execute(
                "SELECT node_id, vector, model, updated_at FROM embeddings ORDER BY node_id"
            ).fetchall(),
            "vec": conn.execute(
                "SELECT rowid, embedding FROM vec_embeddings ORDER BY rowid"
            ).fetchall(),
            "vec_schema": conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
            ).fetchone(),
        }
    finally:
        conn.close()


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


def test_initialize_finalizes_vec_when_active_identity_already_matches(
    tmp_path, monkeypatch
):
    """A no-op identity inspection still permits the post-repair vec step."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    finalized: list[pathlib.Path] = []
    real_finalize = CashewMemoryProvider._finalize_vec_schema

    def record_finalize(provider, path):
        finalized.append(path)
        return real_finalize(provider, path)

    monkeypatch.setattr(CashewMemoryProvider, "_finalize_vec_schema", record_finalize)
    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "all-MiniLM-L6-v2"}, str(tmp_path))
    provider.initialize("already-matching", hermes_home=str(tmp_path))
    try:
        assert finalized == [db_path]
        assert provider._vector_available is True
        assert provider._embedding_dimensions(db_path) == ({384}, 384)
    finally:
        provider.shutdown()


def test_existing_vec_schema_without_extension_keeps_identity_ready(
    tmp_path, monkeypatch
):
    """Optional vec loss cannot reject an otherwise verified ordinary identity."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", _fake_migrate_to_1024
    )

    import sqlite_vec

    real_load = sqlite_vec.load
    loads = 0

    def unavailable_after_migration(conn):
        nonlocal loads
        loads += 1
        if loads == 1:
            return real_load(conn)
        raise RuntimeError("sqlite-vec unavailable")

    embedded_queries: list[dict] = []
    persisted_extracts: list[dict] = []

    def record_embedding_route(**kwargs):
        embedded_queries.append(kwargs)
        return []

    def persist_extract(**kwargs):
        persisted_extracts.append(kwargs)
        return types.SimpleNamespace(new_nodes=["written"], new_edges=[])

    monkeypatch.setattr(sqlite_vec, "load", unavailable_after_migration)
    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", record_embedding_route)
    monkeypatch.setattr("core.session.end_session", persist_extract, raising=False)
    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "thenlper/gte-large"}, str(tmp_path))
    provider.initialize("vec-unavailable-repair", hermes_home=str(tmp_path))
    try:
        assert provider._embedding_identity_ready is True
        assert provider._vector_available is False
        assert provider._embedding_dimensions(db_path) == ({1024}, 1024)

        extract = json.loads(
            provider.handle_tool_call(
                "cashew_extract",
                {"user_content": "write", "assistant_content": "admitted"},
            )
        )
        assert extract["ok"] is True
        assert persisted_extracts

        provider.prefetch("ordinary BFS route")
        assert embedded_queries == [
            {
                "db_path": str(db_path),
                "query": "ordinary BFS route",
                "top_k": provider._config.recall_k,
                "domain": None,
                "tags": None,
                "exclude_tags": None,
            }
        ]
    finally:
        provider.shutdown()


@pytest.mark.real_embedding_child
def test_pinned_upstream_migration_uses_owned_child_embedding_boundary(tmp_path):
    """The dd57 migration persists child-produced vectors without a parent model."""
    # The suite's broad guard replaces core.embed_nodes. Restore just the
    # pinned upstream function here; its service remains the owned child path.
    core_embeddings = importlib.import_module("core.embeddings")
    migration_script = importlib.import_module("scripts.migrate_embeddings")
    importlib.reload(core_embeddings)
    importlib.reload(migration_script)
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)

    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "thenlper/gte-large"}, str(tmp_path))
    provider.initialize("real-dd57-migration", hermes_home=str(tmp_path))
    try:
        assert provider._embedding_dimensions(db_path) == ({1024}, 1024)
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute(
                "SELECT model, LENGTH(vector) / 4 FROM embeddings WHERE node_id='n1'"
            ).fetchone() == ("thenlper/gte-large", 1024)
        finally:
            conn.close()
    finally:
        provider.shutdown()


@pytest.mark.real_embedding_child
def test_pinned_upstream_migrates_same_dimension_model_identity(tmp_path):
    """A 384-to-384 model switch is still a destructive identity migration."""
    core_embeddings = importlib.import_module("core.embeddings")
    migration_script = importlib.import_module("scripts.migrate_embeddings")
    importlib.reload(core_embeddings)
    importlib.reload(migration_script)
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)

    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "thenlper/gte-small"}, str(tmp_path))
    provider.initialize("real-dd57-same-dimension", hermes_home=str(tmp_path))
    try:
        assert provider._embedding_dimensions(db_path) == ({384}, 384)
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute(
                "SELECT model, LENGTH(vector) / 4 FROM embeddings WHERE node_id='n1'"
            ).fetchone() == ("thenlper/gte-small", 384)
        finally:
            conn.close()
        assert list((db_path.parent / "backups").glob("graph.db.*"))
    finally:
        provider.shutdown()


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
    p._embedding_supervisor = type("Verified", (), {"dimension": 1024})()

    p._repair_embedding_dimension(db_path)

    assert p._embedding_dimensions(db_path) == ({384}, 384)
    conn = sqlite3.connect(str(db_path))
    assert conn.execute(
        "SELECT content FROM thought_nodes WHERE id='n1'"
    ).fetchone() == ("dimension migration",)
    conn.close()
    assert "restoring pre-migration backup" in caplog.text


def test_initialize_failure_restores_legacy_vec_before_wrapper_migration(
    tmp_path, monkeypatch, caplog
):
    """Initialize never drops old vec rows before the repair rollback boundary."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    _make_legacy_vec_schema(db_path)
    before = _logical_legacy_vec_snapshot(db_path)

    def destructive_failure(path, *, confirm, quiet):
        del confirm, quiet
        conn = sqlite3.connect(str(path))
        try:
            _load_sqlite_vec(conn)
            conn.execute("DELETE FROM embeddings")
            conn.execute("DROP TABLE vec_embeddings")
            conn.commit()
        finally:
            conn.close()
        raise RuntimeError("synthetic migration failure")

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", destructive_failure
    )
    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "thenlper/gte-large"}, str(tmp_path))
    provider.initialize("failed-initialize", hermes_home=str(tmp_path))
    try:
        assert _logical_legacy_vec_snapshot(db_path) == before
        assert provider._vector_available is False
        assert "restoring pre-migration backup" in caplog.text
    finally:
        provider.shutdown()


def test_unresolved_identity_is_keyword_only_and_recovers_after_reinitialize(
    tmp_path, monkeypatch
):
    """Repair failure admits no semantic work, but leaves safe query available."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    before = _logical_embedding_snapshot(db_path)
    removed: list[str] = []

    import sys

    jobs = [
        {
            "id": "cashew",
            "name": "cashew-sleep-cycle",
            "script": "cashew-sleep-cycle.py",
            "schedule": "every 12h",
            "prompt": cron_prompt(profile_identity(tmp_path)),
            "no_agent": True,
            "repeat": None,
        },
        {"id": "other", "name": "unrelated-job", "script": "other.py"},
    ]
    created: list[dict] = []

    def list_jobs():
        return list(jobs)

    def remove_job(job_id):
        removed.append(job_id)
        jobs[:] = [job for job in jobs if job["id"] != job_id]

    def create_job(**kwargs):
        created.append(kwargs)
        job = {
            "id": "recovered-cashew",
            **kwargs,
        }
        jobs.append(job)
        return job

    cron_package = types.ModuleType("cron")
    cron_package.__path__ = []  # type: ignore[attr-defined]
    cron_jobs = types.ModuleType("cron.jobs")
    cron_jobs.list_jobs = list_jobs
    cron_jobs.remove_job = remove_job
    cron_jobs.create_job = create_job
    monkeypatch.setitem(sys.modules, "cron", cron_package)
    monkeypatch.setitem(sys.modules, "cron.jobs", cron_jobs)
    monkeypatch.setattr(cashew_module, "_HAS_HERMES_CRON", True)
    source = pathlib.Path(__file__).parents[1] / "plugins" / "memory" / "cashew"
    anchor = tmp_path / "hermes-agent" / "plugins" / "memory" / "cashew"
    anchor.parent.mkdir(parents=True)
    anchor.symlink_to(source, target_is_directory=True)

    def destructive_failure(path, *, confirm, quiet):
        del confirm, quiet
        conn = sqlite3.connect(str(path))
        try:
            _load_sqlite_vec(conn)
            conn.execute("DELETE FROM embeddings")
            conn.execute("DROP TABLE vec_embeddings")
            conn.commit()
        finally:
            conn.close()
        raise RuntimeError("synthetic migration failure")

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", destructive_failure
    )
    called_embedding_retrieval = False

    def must_not_embed(**_kwargs):
        nonlocal called_embedding_retrieval
        called_embedding_retrieval = True
        raise AssertionError("identity-unresolved recall must not embed")

    real_embedding_wait = cashew_module._retrieve_with_embedding_wait
    monkeypatch.setattr(cashew_module, "_retrieve_with_embedding_wait", must_not_embed)
    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "thenlper/gte-large"}, str(tmp_path))
    provider.initialize("unresolved", hermes_home=str(tmp_path))
    try:
        assert provider._embedding_identity_ready is False
        assert provider.health_status()["reason_code"] == "identity_unresolved"
        assert removed == ["cashew"]
        assert provider.prefetch("dimension")
        response = json.loads(
            provider.handle_tool_call("cashew_query", {"query": "dimension"})
        )
        assert response["ok"] is True
        assert "dimension migration" in response["context"]
        assert called_embedding_retrieval is False

        provider.sync_turn("new user", "new assistant")
        assert provider._sync_queue is not None and provider._sync_queue.empty()
        assert (
            provider._drain_once(("new user", "new assistant", "unresolved")) is False
        )
        assert (
            json.loads(
                provider.handle_tool_call(
                    "cashew_extract",
                    {"user_content": "new user", "assistant_content": "new assistant"},
                )
            )["ok"]
            is False
        )
        provider._model_fn = lambda _prompt: "[]"
        assert provider.on_pre_compress([{"role": "user", "content": "one"}] * 6) == ""
        provider._update_access_metrics(["n1"])
        assert _logical_embedding_snapshot(db_path) == before
    finally:
        provider.shutdown()

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", _fake_migrate_to_1024
    )
    monkeypatch.setattr(
        cashew_module, "_retrieve_with_embedding_wait", real_embedding_wait
    )
    embedded_queries: list[dict] = []
    persisted_extracts: list[dict] = []

    def record_embedding_route(**kwargs):
        embedded_queries.append(kwargs)
        return []

    def persist_extract(**kwargs):
        persisted_extracts.append(kwargs)
        conn = sqlite3.connect(kwargs["db_path"])
        try:
            conn.execute(
                "INSERT INTO thought_nodes (id, content, node_type, domain, timestamp) "
                "VALUES ('recovered-write', 'recovered write', 'fact', 'test', '2026-01-03')"
            )
            conn.commit()
        finally:
            conn.close()
        return types.SimpleNamespace(new_nodes=["recovered-write"], new_edges=[])

    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", record_embedding_route)
    monkeypatch.setattr("core.session.end_session", persist_extract, raising=False)
    provider.initialize("recovered", hermes_home=str(tmp_path))
    try:
        assert provider._embedding_identity_ready is True
        assert provider._embedding_dimensions(db_path) == ({1024}, 1024)
        import core.embedding_service

        service = core.embedding_service.get_default_service()
        assert isinstance(service, cashew_module._GenerationBoundEmbeddingService)
        assert service._supervisor is provider._embedding_supervisor
        assert service.embed_np(["recovered generation"]).shape == (1, 1024)

        extract = json.loads(
            provider.handle_tool_call(
                "cashew_extract",
                {"user_content": "write", "assistant_content": "recovered"},
            )
        )
        assert extract["ok"] is True
        assert persisted_extracts
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute(
                "SELECT content FROM thought_nodes WHERE id='recovered-write'"
            ).fetchone() == ("recovered write",)
        finally:
            conn.close()

        provider.prefetch("recovered route")
        assert embedded_queries == [
            {
                "db_path": str(db_path),
                "query": "recovered route",
                "top_k": provider._config.recall_k,
                "domain": None,
                "tags": None,
                "exclude_tags": None,
            }
        ]
        assert created and provider._sleep_cron_job_id == "recovered-cashew"
        assert any(job["id"] == "other" for job in jobs)
    finally:
        provider.shutdown()


def test_partial_embedding_migration_restores_backup(tmp_path, monkeypatch, caplog):
    """A zero-row upstream success is not allowed to commit destructive work."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    before = _logical_embedding_snapshot(db_path)

    def incomplete_success(db_path, *, confirm, quiet):
        del confirm, quiet
        conn = sqlite3.connect(str(db_path))
        _load_sqlite_vec(conn)
        conn.execute("DELETE FROM embeddings")
        conn.execute("DROP TABLE vec_embeddings")
        conn.commit()
        conn.close()
        return {"nodes_embedded": 0}

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", incomplete_success
    )
    p = CashewMemoryProvider()
    p._config = CashewConfig(embedding_model="thenlper/gte-large")
    p._embedding_supervisor = type("Verified", (), {"dimension": 1024})()
    p._repair_embedding_dimension(db_path)

    assert p._embedding_dimensions(db_path) == ({384}, 384)
    assert _logical_embedding_snapshot(db_path) == before
    assert "restoring pre-migration backup" in caplog.text


def test_migration_rolls_back_wrong_model_or_vec_identity(
    tmp_path, monkeypatch, caplog
):
    """Matching counts cannot hide a wrong-model or wrong sqlite-vec result."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    before = _logical_embedding_snapshot(db_path)

    def wrong_postcondition(path, *, confirm, quiet):
        summary = _fake_migrate_to_1024(path, confirm=confirm, quiet=quiet)
        conn = sqlite3.connect(str(path))
        try:
            _load_sqlite_vec(conn)
            conn.execute("UPDATE embeddings SET model = 'wrong/model'")
            conn.execute("DELETE FROM vec_embeddings WHERE node_id = 'n1'")
            conn.commit()
        finally:
            conn.close()
        return summary

    monkeypatch.setattr(
        "scripts.migrate_embeddings.migrate_embeddings", wrong_postcondition
    )
    p = CashewMemoryProvider()
    p._config = CashewConfig(embedding_model="thenlper/gte-large")
    p._embedding_supervisor = type("Verified", (), {"dimension": 1024})()
    p._repair_embedding_dimension(db_path)

    assert _logical_embedding_snapshot(db_path) == before
    assert "restoring pre-migration backup" in caplog.text


def test_vec_only_postcondition_mismatch_rolls_back_to_identity_unresolved(
    tmp_path, monkeypatch, caplog
):
    """A loaded vec table must still contain every ordinary active node ID."""
    db_path = tmp_path / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True)
    _make_dimension_mismatch_db(db_path)
    before = _logical_embedding_snapshot(db_path)

    def wrong_vec_only(path, *, confirm, quiet):
        summary = _fake_migrate_to_1024(path, confirm=confirm, quiet=quiet)
        conn = sqlite3.connect(str(path))
        try:
            _load_sqlite_vec(conn)
            vector = conn.execute(
                "SELECT embedding FROM vec_embeddings WHERE node_id = 'n1'"
            ).fetchone()[0]
            conn.execute("DELETE FROM vec_embeddings WHERE node_id = 'n1'")
            conn.execute(
                "INSERT INTO vec_embeddings (node_id, embedding) VALUES ('wrong-id', ?)",
                (vector,),
            )
            conn.commit()
        finally:
            conn.close()
        return summary

    monkeypatch.setattr("scripts.migrate_embeddings.migrate_embeddings", wrong_vec_only)
    provider = CashewMemoryProvider()
    provider.save_config({"embedding_model": "thenlper/gte-large"}, str(tmp_path))
    provider.initialize("vec-only-mismatch", hermes_home=str(tmp_path))
    try:
        assert provider._embedding_identity_ready is False
        assert provider.health_status()["reason_code"] == "identity_unresolved"
        assert _logical_embedding_snapshot(db_path) == before
        assert "restoring pre-migration backup" in caplog.text
    finally:
        provider.shutdown()


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
        p._embedding_supervisor = type("Verified", (), {"dimension": 1024})()
        p._repair_embedding_dimension(db_path)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    assert called is False
    assert p._embedding_dimensions(db_path) == ({384}, 384)
    assert "migration deferred" in caplog.text
