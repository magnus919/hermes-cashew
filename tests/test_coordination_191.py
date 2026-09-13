"""Issue #191 coordination and journal safety contracts."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
import threading
import time
import types
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import pytest

from plugins.memory.cashew import (
    CashewMemoryProvider,
    _bootstrap_sqlite_profile,
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


def _hold_transferred_admission(graph: str, cache: str, ready) -> None:
    """Child helper used to prove kernel release after abrupt process death."""
    with admit_operation(
        graph_path=graph,
        cache_path=cache,
        exclusive=True,
        cache_exclusive=True,
        deadline=2.0,
    ):
        ready.set()
        time.sleep(30)


def _run_think_process(
    db_path: str,
    control,
    calls_path: str,
    entered_path: str,
    crash: bool = False,
    recover: bool = False,
) -> None:
    """Exercise the production think caller with only the opaque model mocked."""
    import core.session

    from plugins.memory.cashew import CashewMemoryProvider
    from plugins.memory.cashew.config import CashewConfig

    provider = CashewMemoryProvider()
    provider._db_path = Path(db_path)
    provider._config = CashewConfig(
        embedding_model="model-a", think_cycles=True, think_interval=1
    )
    provider._cache_writes_disabled = True
    provider._embedding_identity_ready = True
    provider._embedding_generation = "g1"
    provider._runtime_epoch = 1
    provider._embedding_supervisor = types.SimpleNamespace(
        dimension=384, generation="g1"
    )
    provider._model_fn = lambda _prompt: ""
    if recover:
        provider._recover_think_claim(Path(db_path))

    def opaque_think(**_kwargs):
        with open(calls_path, "a", encoding="utf-8") as stream:
            stream.write("call\n")
        Path(entered_path).write_text("entered", encoding="utf-8")
        control.send("entered")
        if crash:
            import os

            os._exit(23)
        assert control.poll(15)
        assert control.recv() == "release"
        return types.SimpleNamespace(new_nodes=[], new_edges=[])

    core.session.think_cycle = opaque_think
    provider._run_think_cycle_if_due()
    control.send("done")


def _prepare_think_db(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, content TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '384');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '384');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '1');
            INSERT INTO hermes_provider_meta VALUES ('think_counter', '0');
            INSERT INTO hermes_provider_meta VALUES ('think_claim_state', 'none');
            """
        )


def _probe_exclusive_leases(graph: str, cache: str, control) -> None:
    """Report whether an independent process can acquire either lease."""
    with try_maintenance_lock(graph) as graph_lease:
        graph_available = graph_lease is not None
    with try_maintenance_lock(cache) as cache_lease:
        cache_available = cache_lease is not None
    control.send((graph_available, cache_available))
    control.close()


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


def test_readonly_uri_escapes_literal_percent_sequences_and_missing_is_readonly(
    tmp_path,
):
    """Literal ``%2e%2e`` path components cannot select another profile."""
    intended_dir = tmp_path / "literal%2e%2e"
    decoy_dir = tmp_path / "literal.."
    intended_dir.mkdir()
    decoy_dir.mkdir()
    intended = intended_dir / "profile.db"
    decoy = decoy_dir / "profile.db"
    for path, marker in ((intended, "intended"), (decoy, "decoy")):
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
            conn.execute("INSERT INTO marker VALUES (?)", (marker,))

    conn, _mode = open_readonly_verified(intended)
    try:
        assert conn.execute("SELECT value FROM marker").fetchone() == ("intended",)
        assert conn.execute("PRAGMA query_only").fetchone() == (1,)
    finally:
        conn.close()

    missing = intended_dir / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        sqlite_journal_report(missing)
    assert not missing.exists()


def test_affected_existing_wal_is_write_refused(tmp_path, monkeypatch):
    db = tmp_path / "existing-wal.db"
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    tracked = {
        path: sha256(path.read_bytes()).hexdigest()
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm"))
        if path.exists()
    }
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    with sqlite3.connect(db) as conn:
        with pytest.raises(SQLiteWALUnsupportedError, match="existing WAL"):
            guard_sqlite_journal(conn)
    assert {
        path: sha256(path.read_bytes()).hexdigest() for path in tracked if path.exists()
    } == tracked


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
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            "CREATE TABLE hermes_cashew_cache_meta "
            "(model TEXT PRIMARY KEY, embedding_dim INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO hermes_cashew_cache_meta VALUES (?, ?)",
            ("model-a", 384),
        )

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


def test_real_cache_facade_get_compute_put_preserves_per_model_metadata(tmp_path):
    """The pinned EmbeddingCache uses one leased file with model-scoped metadata."""
    from core.embedding_cache import EmbeddingCache

    cache_path = tmp_path / "shared-cache.db"
    graph_a = tmp_path / "graph-a.db"
    graph_b = tmp_path / "graph-b.db"
    _bootstrap_sqlite_profile(
        graph_a, cache_path, cache_disabled=False, model="model-a", embedding_dim=4
    )
    _bootstrap_sqlite_profile(
        graph_b, cache_path, cache_disabled=False, model="model-b", embedding_dim=5
    )
    raw = EmbeddingCache(str(cache_path))
    supervisor = object()
    facade_a = _GenerationBoundEmbeddingCache(
        raw,
        path=cache_path,
        model="model-a",
        embedding_dim=4,
        supervisor=supervisor,
        generation="a",
    )
    with admit_operation(
        graph_path=graph_a,
        cache_path=cache_path,
        model="model-a",
        embedding_dim=4,
        supervisor=supervisor,
        embedding_generation="a",
        cache_exclusive=True,
    ):
        assert facade_a.get_many("model-a", ["text"]) == [None]
        assert facade_a.put_many("model-a", [("text", [1, 2, 3, 4])]) == 1
        assert len(facade_a.get_many("model-a", ["text"])[0]) == 4
    with sqlite3.connect(cache_path) as conn:
        metadata = dict(
            conn.execute(
                "SELECT model, embedding_dim FROM hermes_cashew_cache_meta"
            ).fetchall()
        )
    assert metadata == {"model-a": 4, "model-b": 5}

    with pytest.raises(OperationAdmissionError, match="model mismatch"):
        with admit_operation(
            graph_path=graph_b,
            cache_path=cache_path,
            model="model-b",
            embedding_dim=5,
            supervisor=supervisor,
            embedding_generation="b",
            cache_exclusive=True,
        ):
            facade_a.put_many("model-a", [("stale", [0, 0, 0, 0])])


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


def test_admission_leases_are_released_by_kernel_after_process_death(tmp_path):
    """A killed async owner cannot strand either ordered lease."""
    graph = tmp_path / "brain.db"
    cache = tmp_path / "cache.db"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child = context.Process(
        target=_hold_transferred_admission,
        args=(str(graph), str(cache), ready),
    )
    child.start()
    try:
        assert ready.wait(timeout=5)
        child.terminate()
        child.join(timeout=5)
        assert child.exitcode is not None
        with try_maintenance_lock(graph) as graph_lease:
            assert graph_lease is not None
        with try_maintenance_lock(cache) as cache_lease:
            assert cache_lease is not None
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=5)


def test_two_process_think_claim_has_one_opaque_call_and_no_duplicate(
    tmp_path, monkeypatch
):
    """The real provider claim transaction serializes two pinned-host callers."""
    del monkeypatch
    db = tmp_path / "think.db"
    _prepare_think_db(db)
    calls = tmp_path / "calls.log"
    context = multiprocessing.get_context("spawn")
    first_control, first_child = context.Pipe()
    second_control, second_child = context.Pipe()
    first_entered = tmp_path / "first.entered"
    second_entered = tmp_path / "second.entered"
    first = context.Process(
        target=_run_think_process,
        args=(str(db), first_child, str(calls), str(first_entered)),
    )
    second = context.Process(
        target=_run_think_process,
        args=(str(db), second_child, str(calls), str(second_entered)),
    )
    first.start()
    second.start()
    try:
        deadline = time.monotonic() + 15
        while not first_entered.exists() and not second_entered.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        selected_control = first_control if first_entered.exists() else second_control
        other_entered = second_entered if first_entered.exists() else first_entered
        assert selected_control.poll(2)
        assert selected_control.recv() == "entered"
        selected_control.send("release")
        first.join(timeout=15)
        second.join(timeout=15)
        assert first.exitcode == 0
        assert second.exitcode == 0
        assert not other_entered.exists()
        assert selected_control.poll(2)
        assert selected_control.recv() == "done"
        assert calls.read_text().splitlines() == ["call"]
        with sqlite3.connect(db) as conn:
            assert conn.execute(
                "SELECT value FROM hermes_provider_meta WHERE key='think_claim_state'"
            ).fetchone() == ("failed",)
    finally:
        if first.is_alive():
            first.kill()
            first.join(timeout=5)
        if second.is_alive():
            second.kill()
            second.join(timeout=5)


def test_crashed_think_claim_is_recovered_as_uncertain_before_next_call(tmp_path):
    """A restart observes an abandoned claim and blocks replay until resolution."""
    db = tmp_path / "think-crash.db"
    _prepare_think_db(db)
    calls = tmp_path / "calls.log"
    context = multiprocessing.get_context("spawn")
    control, child_control = context.Pipe()
    entered_path = tmp_path / "crashed.entered"
    crashed = context.Process(
        target=_run_think_process,
        args=(str(db), child_control, str(calls), str(entered_path), True),
    )
    crashed.start()
    try:
        deadline = time.monotonic() + 15
        while not entered_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert control.poll(2)
        assert control.recv() == "entered"
        crashed.join(timeout=15)
        assert crashed.exitcode == 23
        restarted_control, restarted_child = context.Pipe()
        restarted_entered = tmp_path / "restarted.entered"
        restarted = context.Process(
            target=_run_think_process,
            args=(
                str(db),
                restarted_child,
                str(calls),
                str(restarted_entered),
                False,
                True,
            ),
        )
        restarted.start()
        restarted.join(timeout=15)
        assert restarted.exitcode == 0
        assert restarted_control.poll()
        assert restarted_control.recv() == "done"
        assert calls.read_text().splitlines() == ["call"]
        with sqlite3.connect(db) as conn:
            assert conn.execute(
                "SELECT value FROM hermes_provider_meta WHERE key='think_claim_state'"
            ).fetchone() == ("uncertain",)
    finally:
        if crashed.is_alive():
            crashed.kill()
            crashed.join(timeout=5)


def test_readonly_profile_verifies_persisted_identity_and_dimensions(tmp_path):
    db = tmp_path / "readonly.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY, content TEXT, node_type TEXT, domain TEXT,
                timestamp TEXT, access_count INTEGER, last_accessed TEXT,
                source_file TEXT, decayed INTEGER, metadata TEXT, last_updated TEXT,
                mood_state TEXT, permanent INTEGER, tags TEXT, referent_time TEXT
            );
            CREATE TABLE derivation_edges (parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, timestamp TEXT);
            CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO thought_nodes (id, content, domain) VALUES ('n1', 'memory', 'user');
            INSERT INTO embeddings VALUES ('n1', zeroblob(16), 'model-a', NULL);
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '2');
            PRAGMA user_version = 3;
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
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY, content TEXT, node_type TEXT, domain TEXT,
                timestamp TEXT, access_count INTEGER, last_accessed TEXT,
                source_file TEXT, decayed INTEGER, metadata TEXT, last_updated TEXT,
                mood_state TEXT, permanent INTEGER, tags TEXT, referent_time TEXT
            );
            CREATE TABLE derivation_edges (parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, timestamp TEXT);
            CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding float[4]);
                INSERT INTO thought_nodes (id, content, domain) VALUES ('n1', 'memory', 'user');
            INSERT INTO embeddings VALUES ('n1', zeroblob(16), 'model-a', NULL);
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '2');
            PRAGMA user_version = 3;
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


def test_readonly_verifier_rejects_declared_vec_dimension_mismatch(tmp_path):
    db = tmp_path / "vec-mismatch.db"
    with sqlite3.connect(db) as conn:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.executescript(
            """
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY, content TEXT, node_type TEXT, domain TEXT,
                timestamp TEXT, access_count INTEGER, last_accessed TEXT,
                source_file TEXT, decayed INTEGER, metadata TEXT, last_updated TEXT,
                mood_state TEXT, permanent INTEGER, tags TEXT, referent_time TEXT
            );
            CREATE TABLE derivation_edges (parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, timestamp TEXT);
            CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT);
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding float[5]);
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '2');
            PRAGMA user_version = 3;
            """
        )
    conn, _mode = open_readonly_verified(db)
    try:
        with pytest.raises(
            SQLiteWALUnsupportedError, match="vec dimension is inconsistent"
        ):
            verify_readonly_profile(conn)
    finally:
        conn.close()


def test_affected_provider_rejects_supported_but_malformed_keyword_schema(
    tmp_path, monkeypatch
):
    """A version-stamped profile missing a keyword column is unavailable."""
    db = tmp_path / "malformed-keyword.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY, content TEXT, node_type TEXT, domain TEXT,
                timestamp TEXT, access_count INTEGER, last_accessed TEXT,
                source_file TEXT, decayed INTEGER, metadata TEXT, last_updated TEXT,
                mood_state TEXT, permanent INTEGER, referent_time TEXT
            );
            CREATE TABLE derivation_edges (
                parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, timestamp TEXT
            );
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT
            );
            CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO hermes_provider_meta VALUES ('embedding_model', 'model-a');
            INSERT INTO hermes_provider_meta VALUES ('embedding_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('vec_dim', '4');
            INSERT INTO hermes_provider_meta VALUES ('maintenance_epoch', '1');
            PRAGMA user_version = 3;
            """
        )
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    with pytest.raises(SQLiteWALUnsupportedError, match="verification failed"):
        _sqlite_profile_policy(db)


def test_cron_revalidates_identity_after_graph_cache_admission(
    tmp_path, monkeypatch, capsys
):
    """A persisted identity change at the lease boundary prevents sleep work."""
    import plugins.memory.cashew.sleep_cron_script as cron_module

    db = tmp_path / "cron.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO hermes_provider_meta VALUES (?, ?)",
            [
                ("embedding_model", "model-a"),
                ("embedding_dim", "4"),
                ("vec_dim", "4"),
                ("maintenance_epoch", "1"),
            ],
        )

    class Config:
        cashew_db_path = str(db)
        sleep_max_nodes = 10
        embedding_model = "model-a"
        embedding_device = "cpu"

    class Supervisor:
        dimension = 4
        generation = "cron-generation"

        def __init__(self, **_kwargs):
            self.closed = False

        def start(self):
            return self.dimension

        def close(self):
            self.closed = True

    sleep_calls: list[dict] = []
    sleep_module = types.SimpleNamespace(
        __package__="cron_fixture",
        run_sleep_cycle=lambda **kwargs: sleep_calls.append(kwargs),
    )
    config_module = types.SimpleNamespace(
        load_config=lambda _home: Config(),
        resolve_db_path=lambda _home, value: value,
        resolve_model_fn=lambda **_kwargs: None,
    )
    process_module = types.SimpleNamespace(EmbeddingSupervisor=Supervisor)
    filter_module = types.SimpleNamespace(
        acquire_provider_scrub_filters=lambda: None,
        release_provider_scrub_filters=lambda: None,
    )
    real_admit = admit_operation

    @contextmanager
    def mutate_at_admission(**kwargs):
        with real_admit(**kwargs) as token:
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE hermes_provider_meta SET value='model-b' WHERE key='embedding_model'"
                )
                conn.commit()
            yield token

    admission_module = types.SimpleNamespace(
        admit_operation=mutate_at_admission,
        OperationAdmissionError=OperationAdmissionError,
    )
    locking_module = types.SimpleNamespace(
        guard_sqlite_journal=lambda _conn: "delete",
        MaintenanceLockAcquisitionError=OSError,
        SQLiteWALUnsupportedError=SQLiteWALUnsupportedError,
    )

    def import_module(name):
        return {
            "cron_fixture.log_filter": filter_module,
            "cron_fixture.embedding_process": process_module,
            "cron_fixture.admission": admission_module,
            "cron_fixture.locking": locking_module,
        }[name]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        cron_module,
        "_load_profile_modules",
        lambda _home: (config_module, sleep_module),
    )
    monkeypatch.setattr(cron_module.importlib, "import_module", import_module)
    cron_module.main()
    assert json.loads(capsys.readouterr().out) == {}
    assert sleep_calls == []


def test_affected_wal_provider_never_enters_write_or_upstream_paths(
    tmp_path, monkeypatch
):
    """A verified affected-WAL profile remains read-only across provider APIs."""
    from core.db import ensure_schema

    db = tmp_path / "affected-live-wal.db"
    ensure_schema(str(db))
    (tmp_path / "cashew.json").write_text(
        '{"cashew_db_path": "affected-live-wal.db"}', encoding="utf-8"
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO hermes_provider_meta (key, value) VALUES (?, ?)",
            [
                ("embedding_model", "all-MiniLM-L6-v2"),
                ("embedding_dim", "384"),
                ("vec_dim", "384"),
                ("maintenance_epoch", "1"),
            ],
        )
        conn.commit()
    holder = sqlite3.connect(db)
    assert holder.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    # Keep a read transaction open while a separate writer commits.  This
    # forces the live -wal and -shm sidecars to exist for the provider probe.
    holder.execute("BEGIN")
    holder.execute("SELECT 1").fetchone()
    with sqlite3.connect(db) as writer:
        writer.execute(
            "INSERT INTO thought_nodes (id, content, node_type, domain) "
            "VALUES (?, ?, ?, ?)",
            ("wal-live", "committed while reader is open", "observation", "user"),
        )
        writer.commit()
    initial_tracked = {
        path: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
        for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm"))
    }
    assert len(initial_tracked) == 3
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 50, 4))
    calls: list[str] = []

    def forbidden(name):
        def fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"forbidden affected-WAL path: {name}")

        return fail

    monkeypatch.setattr("core.embedding_cache.EmbeddingCache", forbidden("cache"))
    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", forbidden("bfs"))
    monkeypatch.setattr("core.session.end_session", forbidden("extract"))
    monkeypatch.setattr("core.session.think_cycle", forbidden("think"))
    monkeypatch.setattr(
        "plugins.memory.cashew.sleep_adapter.run_sleep_cycle", forbidden("sleep")
    )
    provider = CashewMemoryProvider()
    try:
        provider.initialize("affected-wal", hermes_home=str(tmp_path))
        # Read-only connection setup may update SQLite's transient SHM lock
        # bytes.  Freeze the live sidecars after that setup, then prove all
        # provider calls leave the committed database and journal untouched.
        tracked = {
            path: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
            for path in initial_tracked
        }
        assert provider.prefetch("missing") == ""
        response = provider.handle_tool_call("cashew_query", {"query": "missing"})
        assert '"ok": true' in response
        provider.sync_turn("blocked", "write")
    finally:
        provider.shutdown()
    assert calls == []
    assert {
        path: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
        for path in tracked
    } == tracked
    assert holder.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    holder.rollback()
    holder.close()


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


def test_unknown_upstream_failure_is_uncertain_and_not_replayed(tmp_path, monkeypatch):
    """Unknown progress stays uncertain and cannot be replayed automatically."""
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

    def swallowed_progress(**_kwargs):
        nonlocal calls
        calls += 1
        # Model an upstream helper that catches its internal database error and
        # returns no ExtractionResult, leaving commit progress unknowable.
        return None

    monkeypatch.setattr("core.session.end_session", swallowed_progress, raising=False)
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
        "partial": 0,
        "uncertain": 1,
    }


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


def test_corrupt_think_counter_is_reset_without_opaque_call(tmp_path, monkeypatch):
    """Malformed persistent counter metadata is contained at the sync boundary."""
    db = tmp_path / "corrupt-think-counter.db"
    _prepare_think_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hermes_provider_meta SET value='not-a-number' "
            "WHERE key='think_counter'"
        )

    provider = CashewMemoryProvider()
    provider._db_path = db
    provider._config = CashewConfig(
        embedding_model="model-a", think_cycles=True, think_interval=1
    )
    provider._cache_writes_disabled = True
    provider._embedding_identity_ready = True
    provider._embedding_generation = "g1"
    provider._runtime_epoch = 1
    provider._model_fn = lambda _prompt: ""
    provider._embedding_supervisor = types.SimpleNamespace(
        dimension=384, generation="g1"
    )

    def forbidden(**_kwargs):
        raise AssertionError("corrupt metadata must not invoke think_cycle")

    monkeypatch.setattr("core.session.think_cycle", forbidden, raising=False)
    provider._run_think_cycle_if_due()

    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT value FROM hermes_provider_meta WHERE key='think_counter'"
        ).fetchone() == ("0",)


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
