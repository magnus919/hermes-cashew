"""Issue #191 coordination and journal safety contracts."""

from __future__ import annotations

import sqlite3
import threading
import types
from hashlib import sha256

import pytest

from plugins.memory.cashew import (
    CashewMemoryProvider,
    _GenerationBoundEmbeddingCache,
    _NoopEmbeddingCache,
    _sqlite_profile_policy,
)
from plugins.memory.cashew.admission import (
    OperationAdmissionError,
    admit_operation,
)
from plugins.memory.cashew.config import CashewConfig
from plugins.memory.cashew.locking import (
    SQLiteWALUnsupportedError,
    bootstrap_sqlite_journal,
    guard_sqlite_journal,
    open_readonly_verified,
    sqlite_journal_report,
    sqlite_wal_reset_vulnerable,
    try_maintenance_lock,
    verify_readonly_profile,
)


@pytest.mark.parametrize(
    ("version", "vulnerable"),
    [
        ((3, 44, 5), True),
        ((3, 44, 6), False),
        ((3, 50, 4), True),
        ((3, 50, 7), False),
        ((3, 51, 2), True),
        ((3, 51, 3), False),
    ],
)
def test_wal_reset_classifier_has_backport_boundaries(version, vulnerable):
    assert sqlite_wal_reset_vulnerable(version) is vulnerable


def test_affected_fresh_db_stays_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        assert guard_sqlite_journal(conn) == "delete"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        conn.close()


def test_fixed_bootstrap_enables_wal_and_ordinary_guard_preserves_mode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 7))
    db = tmp_path / "fixed.db"
    with sqlite3.connect(db) as conn:
        assert bootstrap_sqlite_journal(conn) == "wal"
    with sqlite3.connect(db) as conn:
        assert guard_sqlite_journal(conn) == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    report = sqlite_journal_report(db)
    assert report["sqlite_version"]
    assert report["sqlite_source_id"]
    assert report["journal_mode"] == "wal"


def test_affected_existing_wal_is_write_refused(tmp_path, monkeypatch):
    db = tmp_path / "existing-wal.db"
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    with sqlite3.connect(db) as conn:
        with pytest.raises(SQLiteWALUnsupportedError, match="existing WAL"):
            guard_sqlite_journal(conn)


def test_cache_only_admission_does_not_require_graph_lock(tmp_path):
    cache = tmp_path / "embedding-cache.db"
    with admit_operation(cache_path=cache, model="model-a", embedding_dim=384) as token:
        assert token.graph_path is None
        assert token.cache_path == cache.resolve()
        assert token.cache_lease is not None


def test_shared_cache_excludes_a_different_graph_identity(tmp_path):
    """The cache lease is a shared profile resource, independent of graph path."""
    cache = tmp_path / "embedding-cache.db"
    first_graph = tmp_path / "one.db"
    second_graph = tmp_path / "two.db"
    entered = threading.Event()
    finished = threading.Event()
    outcome: list[BaseException | None] = []

    def competing_graph() -> None:
        entered.wait(timeout=1)
        try:
            with admit_operation(
                graph_path=second_graph,
                cache_path=cache,
                model="model-a",
                embedding_dim=384,
                cache_exclusive=True,
                deadline=0.05,
            ):
                outcome.append(None)
        except BaseException as exc:
            outcome.append(exc)
        finally:
            finished.set()

    with admit_operation(
        graph_path=first_graph,
        cache_path=cache,
        model="model-a",
        embedding_dim=384,
        cache_exclusive=True,
    ):
        thread = threading.Thread(target=competing_graph)
        thread.start()
        entered.set()
        assert finished.wait(timeout=1)
    thread.join(timeout=1)
    assert len(outcome) == 1
    assert isinstance(outcome[0], OperationAdmissionError)


def test_graph_lease_releases_when_cache_admission_fails(tmp_path):
    graph = tmp_path / "brain.db"
    cache = tmp_path / "embedding-cache.db"
    with try_maintenance_lock(cache) as holder:
        assert holder is not None
        with pytest.raises(OperationAdmissionError, match="deadline"):
            with admit_operation(
                graph_path=graph,
                cache_path=cache,
                exclusive=True,
                cache_exclusive=True,
                deadline=0.05,
            ):
                pass
    with try_maintenance_lock(graph) as released:
        assert released is not None


def test_generation_bound_cache_rejects_stale_put_without_exact_admission(tmp_path):
    cache_path = tmp_path / "embedding-cache.db"
    calls: list[tuple[str, str]] = []

    class RawCache:
        def put(self, model, text, _vector):
            calls.append((model, text))

    supervisor = object()
    cache = _GenerationBoundEmbeddingCache(
        RawCache(),
        path=cache_path,
        model="model-a",
        embedding_dim=384,
        supervisor=supervisor,
        generation="generation-a",
    )
    with pytest.raises(OperationAdmissionError, match="not admitted"):
        cache.put("model-a", "stale", [0.0])
    with admit_operation(
        cache_path=cache_path,
        model="model-a",
        embedding_dim=384,
        supervisor=supervisor,
        embedding_generation="generation-a",
        cache_exclusive=True,
    ):
        cache.put("model-a", "fresh", [0.0])
    assert calls == [("model-a", "fresh")]


def test_nested_cache_mismatch_is_rejected_without_reacquiring(tmp_path):
    graph = tmp_path / "brain.db"
    cache = tmp_path / "cache.db"
    with admit_operation(
        graph_path=graph,
        cache_path=cache,
        model="model-a",
        embedding_dim=384,
    ):
        with pytest.raises(OperationAdmissionError, match="model mismatch"):
            with admit_operation(
                graph_path=graph,
                cache_path=cache,
                model="model-b",
                embedding_dim=384,
            ):
                pass


def test_nested_upgrade_and_full_identity_are_rejected(tmp_path):
    graph = tmp_path / "brain.db"
    cache = tmp_path / "cache.db"
    with admit_operation(
        graph_path=graph,
        cache_path=cache,
        model="model-a",
        embedding_dim=384,
        vec_dim=384,
    ):
        with pytest.raises(OperationAdmissionError, match="graph lease upgrade"):
            with admit_operation(
                graph_path=graph,
                cache_path=cache,
                model="model-a",
                embedding_dim=384,
                vec_dim=384,
                exclusive=True,
            ):
                pass


def test_transferred_admission_holds_and_releases_both_leases(tmp_path):
    from plugins.memory.cashew.locking import try_maintenance_lock

    graph = tmp_path / "brain.db"
    cache = tmp_path / "cache.db"
    with admit_operation(
        graph_path=graph,
        cache_path=cache,
        model="model-a",
        embedding_dim=384,
        vec_dim=384,
        exclusive=True,
        cache_exclusive=True,
    ) as admission:
        owner = admission.lease_owner
        assert owner is not None
        owner.transfer()
        with try_maintenance_lock(graph) as graph_lease:
            assert graph_lease is None
        with try_maintenance_lock(cache) as cache_lease:
            assert cache_lease is None
        owner.close()
    with try_maintenance_lock(graph) as graph_lease:
        assert graph_lease is not None
    with try_maintenance_lock(cache) as cache_lease:
        assert cache_lease is not None


def test_readonly_profile_verifies_persisted_identity_and_dimensions(tmp_path):
    db = tmp_path / "readonly.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, content TEXT, domain TEXT);
            CREATE TABLE derivation_edges (parent_id TEXT, child_id TEXT);
            CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO thought_nodes VALUES ('n1', 'memory', 'user');
            INSERT INTO embeddings VALUES ('n1', zeroblob(16), 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '2');
            """
        )
    conn, mode = open_readonly_verified(db)
    try:
        assert mode == "delete"
        report = verify_readonly_profile(conn)
    finally:
        conn.close()
    assert report["provider_model"] == "model-a"
    assert report["provider_embedding_dim"] == "4"


def test_readonly_vec_profile_loads_extension_without_profile_mutation(tmp_path):
    """A vec0 table is inspectable through a genuinely read-only connection."""
    import sqlite_vec

    db = tmp_path / "readonly-vec.db"
    with sqlite3.connect(db) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.executescript(
            """
            CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, content TEXT, domain TEXT);
            CREATE TABLE derivation_edges (parent_id TEXT, child_id TEXT);
            CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding float[4]);
            INSERT INTO thought_nodes VALUES ('n1', 'memory', 'user');
            INSERT INTO embeddings VALUES ('n1', zeroblob(16), 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '2');
            """
        )
        conn.execute(
            "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
            ("n1", sqlite_vec.serialize_float32([0.0] * 4)),
        )
    tracked = [db, db.with_name(f"{db.name}-wal"), db.with_name(f"{db.name}-shm")]
    before = {
        path: sha256(path.read_bytes()).hexdigest() for path in tracked if path.exists()
    }
    conn, _mode = open_readonly_verified(db)
    try:
        assert verify_readonly_profile(conn)["provider_vec_dim"] == "4"
    finally:
        conn.close()
    after = {
        path: sha256(path.read_bytes()).hexdigest() for path in tracked if path.exists()
    }
    assert after == before


def test_affected_profile_that_cannot_be_verified_is_unavailable(tmp_path, monkeypatch):
    db = tmp_path / "affected.db"
    db.touch()
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    fake_conn = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        "plugins.memory.cashew.open_readonly_verified", lambda _path: (fake_conn, "wal")
    )
    monkeypatch.setattr(
        "plugins.memory.cashew.verify_readonly_profile",
        lambda _conn: (_ for _ in ()).throw(SQLiteWALUnsupportedError("bad profile")),
    )
    with pytest.raises(SQLiteWALUnsupportedError, match="verification failed"):
        _sqlite_profile_policy(db)


def test_runtime_identity_epoch_is_stable_for_same_identity_and_fences_stale_owner(
    tmp_path,
):
    """Concurrent same-identity startup cannot invalidate an admitted owner."""
    db = tmp_path / "brain.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )

    def provider(model: str) -> CashewMemoryProvider:
        instance = CashewMemoryProvider()
        instance._db_path = db
        instance._config = CashewConfig(embedding_model=model)
        instance._cache_writes_disabled = True
        instance._embedding_identity_ready = True
        instance._embedding_supervisor = types.SimpleNamespace(
            dimension=384, generation="generation"
        )
        instance._embedding_generation = "generation"
        return instance

    first = provider("all-MiniLM-L6-v2")
    first._write_runtime_identity(db)
    first_epoch = first._runtime_epoch
    second = provider("all-MiniLM-L6-v2")
    second._write_runtime_identity(db)
    assert second._runtime_epoch == first_epoch == 1

    replacement = provider("thenlper/gte-large")
    replacement._write_runtime_identity(db)
    assert replacement._runtime_epoch == 2
    with admit_operation(
        graph_path=db,
        model="all-MiniLM-L6-v2",
        embedding_dim=384,
        vec_dim=384,
        epoch=first_epoch,
        supervisor=first._embedding_supervisor,
        embedding_generation="generation",
    ) as admission:
        with pytest.raises(OperationAdmissionError, match="epoch"):
            first._validate_runtime_identity(admission)


def test_noop_cache_is_exactly_non_persistent(tmp_path):
    cache = _NoopEmbeddingCache(tmp_path / "cache.db")
    assert cache.get_many("model-a", ["a", "b"]) == [None, None]
    assert cache.put_many("model-a", [("a", [1.0])]) == 0
    assert cache.get("model-a", "a") is None
    assert cache.size("model-a") == 0
    assert not (tmp_path / "cache.db").exists()


def test_admitted_busy_failure_is_not_replayed(tmp_path, monkeypatch):
    db = tmp_path / "brain.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    provider = CashewMemoryProvider()
    provider._db_path = db
    provider._config = CashewConfig(embedding_model="all-MiniLM-L6-v2")
    provider._cache_writes_disabled = True
    provider._embedding_identity_ready = True
    provider._embedding_supervisor = types.SimpleNamespace(
        dimension=384, generation="g1"
    )
    calls = 0

    def busy(**_kwargs):
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("core.session.end_session", busy, raising=False)
    with pytest.raises(RuntimeError, match="not replayable"):
        provider._drain_once(("u", "a", "s"))
    assert calls == 1


def test_opaque_upstream_prefix_is_partial_and_unknown_is_not_replayed(
    tmp_path, monkeypatch
):
    """A persisted prefix is visible in the ledger and never schedules a replay."""
    db = tmp_path / "brain.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE thought_nodes (id TEXT PRIMARY KEY);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
    provider = CashewMemoryProvider()
    provider._db_path = db
    provider._config = CashewConfig(embedding_model="all-MiniLM-L6-v2")
    provider._cache_writes_disabled = True
    provider._embedding_identity_ready = True
    provider._embedding_supervisor = types.SimpleNamespace(
        dimension=384, generation="g1"
    )
    calls = 0

    def writes_prefix_then_fails(**_kwargs):
        nonlocal calls
        calls += 1
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO thought_nodes VALUES ('prefix')")
        raise RuntimeError("upstream failed after commit")

    monkeypatch.setattr(
        "core.session.end_session", writes_prefix_then_fails, raising=False
    )
    ledger = provider._outcomes
    ledger.admit()
    ledger.start()
    with pytest.raises(RuntimeError, match="not replayable") as caught:
        provider._drain_once(("user", "assistant", "session"))
    provider._fail_worker_turn(ledger, provider._health_generation, caught.value)
    assert calls == 1
    assert ledger.work_snapshot() == {
        "accepted": 1,
        "completed": 0,
        "failed": 0,
        "dropped": 0,
        "rejected": 0,
        "pending": 0,
        "in_flight": 0,
        "reconciled": True,
        "partial": 1,
        "uncertain": 0,
    }


def test_think_claim_failure_is_uncertain_and_blocks_replay(tmp_path, monkeypatch):
    db = tmp_path / "brain.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    provider = CashewMemoryProvider()
    provider._db_path = db
    provider._config = CashewConfig(
        embedding_model="all-MiniLM-L6-v2", think_cycles=True, think_interval=1
    )
    provider._cache_writes_disabled = True
    provider._embedding_identity_ready = True
    provider._model_fn = lambda _prompt: ""
    provider._embedding_supervisor = types.SimpleNamespace(
        dimension=384, generation="g1"
    )
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("model failed")

    monkeypatch.setattr("core.session.think_cycle", fail, raising=False)
    provider._run_think_cycle_if_due()
    provider._run_think_cycle_if_due()
    with sqlite3.connect(db) as conn:
        state = conn.execute(
            "SELECT value FROM hermes_provider_meta WHERE key='think_claim_state'"
        ).fetchone()
    assert calls == 1
    assert state == ("uncertain",)


def test_persisted_model_mismatch_requires_migration_even_with_null_rows(tmp_path):
    db = tmp_path / "brain.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT)"
        )
        conn.execute(
            "CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, decayed INTEGER, content TEXT)"
        )
        conn.execute(
            "CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a')"
        )
    provider = CashewMemoryProvider()
    provider._config = CashewConfig(embedding_model="model-b")
    provider._embedding_supervisor = types.SimpleNamespace(dimension=384)
    assert provider._embedding_migration_required(db) is True
