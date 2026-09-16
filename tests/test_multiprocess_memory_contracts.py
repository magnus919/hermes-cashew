"""Bounded real-upstream subprocess contracts for shared Cashew brains.

The child only wraps ``core.session.end_session`` to establish a barrier; every
write still invokes the original upstream function. This makes overlap
deterministic without replacing persistence with a success stub.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import select
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from plugins.memory.cashew import CashewMemoryProvider

# The fast suite supports a missing dependency with a core.session stub.
# These subprocess contracts require the real upstream package instead.  Probe
# only the top-level package: probing core.context treats a corrupt or partial
# cashew-brain installation as absent and hides its normal import failure.
_cashew_spec = importlib.machinery.PathFinder.find_spec("core")
if _cashew_spec is None:
    pytest.skip(
        "cashew-brain is required for real-upstream multiprocess contracts",
        allow_module_level=True,
    )

_READY_TIMEOUT = 10
_EXIT_TIMEOUT = 15


_IMPORT_GUARD_RUNNER = r"""
import importlib.util
import pathlib
import runpy
import sys
import types

import pytest

repo = pathlib.Path(sys.argv[1])
fake_root = pathlib.Path(sys.argv[2])
mode = sys.argv[3]
sys.modules.pop("core", None)

if mode == "missing":
    # Keep stdlib locations but hide site-packages after pytest is loaded, so
    # the installed real core cannot satisfy the test module's probe.
    sys.path[:] = [str(fake_root), str(repo)] + [
        path for path in sys.path if "site-packages" not in path
    ]
    # Match the shared test fixture's no-dependency compatibility stub.
    sys.modules["core"] = types.ModuleType("core")
else:
    # This package shadows the installed core while installed dependencies
    # remain available to the module under test.
    sys.path.insert(0, str(fake_root))
    (fake_root / "core").mkdir()
    (fake_root / "core" / "__init__.py").write_text("")

try:
    namespace = runpy.run_path(repo / "tests" / "test_multiprocess_memory_contracts.py")
except pytest.skip.Exception:
    if mode != "missing":
        raise SystemExit("partial core installation was incorrectly skipped")
    print("missing-core-skipped")
    raise SystemExit(0)

if mode == "missing":
    raise SystemExit("missing core installation was not skipped")

# Execute the generated child source itself.  It imports core.session before
# any persistence setup, so a partial installation must fail through the
# normal real-upstream path rather than being treated as optional.
sys.argv[:] = ["child", str(fake_root / "profile"), "marker", "initialize"]
try:
    exec(namespace["_CHILD"], {"__name__": "__main__"})
except ModuleNotFoundError as exc:
    if exc.name != "core.session":
        raise
    print("partial-core-failed-naturally")
    raise SystemExit(0)
raise SystemExit("partial core installation unexpectedly reached child setup")
"""


class _KnownIssue191MissingVecRowError(AssertionError):
    """Exact stale-writer missing-vec-row inconsistency pending issue #191."""


class _KnownIssue191StaleVecDimensionError(AssertionError):
    """Exact stale-writer 1024/384 inconsistency pending issue #191."""


_CHILD = r"""
import fcntl, hashlib, json, os, sqlite3, sys, time
from pathlib import Path
import numpy as np
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
import plugins.memory.cashew as cashew_module
from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig
import core.session
import core.embedding_service
assert getattr(core.session, "__file__", ""), "real core.session required"

home = Path(sys.argv[1]); marker = sys.argv[2]; action = sys.argv[3]
overlap = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
assert (home / "cashew.json").exists(), "parent must provision profile before launch"

class DeterministicEmbeddingService:
    model = "thenlper/gte-large"
    dim = 1024
    def embed_np(self, texts):
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        if action == "extract_embed_barrier" and not getattr(self, "_paused_once", False):
            self._paused_once = True
            emit("writer_embeddings_snapshotted")
            if command() != "resume_embeddings":
                raise RuntimeError("expected resume_embeddings command")
        return np.stack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)

    def embed(self, text):
        vectors = self.embed_np([text] if isinstance(text, str) else text)
        return vectors[0].tolist() if isinstance(text, str) else vectors.tolist()

_embedding_service = DeterministicEmbeddingService()
core.embedding_service.get_default_service = lambda: _embedding_service

class VerifiedSupervisor:
    def __init__(self, model):
        self.model = model
        self.dimension = 384 if model == "thenlper/gte-small" else 1024
    def _when_closed(self, callback): callback()
    def close(self, **_kwargs): pass

cashew_module._bind_upstream_embedding = lambda model, *_args, **_kwargs: VerifiedSupervisor(model)

def emit(event, **payload):
    message = json.dumps({"event": event, "core": core.session.__file__, **payload})
    os.write(sys.stdout.fileno(), (message + "\n").encode())

def command():
    value = sys.stdin.readline().strip()
    if not value:
        raise RuntimeError("parent closed the subprocess control pipe")
    return value

def wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("overlap marker did not arrive: " + str(path))
        time.sleep(0.01)

def initialize():
    provider = CashewMemoryProvider()
    provider.initialize("mp-" + marker, hermes_home=str(home))
    if provider._db_path is None:
        raise RuntimeError("provider failed to initialize")
    provider._model_fn = lambda _prompt: json.dumps([{
        "content": "durable marker " + marker,
        "type": "observation",
        "domain": "user",
        "tags": ["multiprocess"],
        "keep": True,
    }])
    return provider

def install_extract_barrier(provider):
    original = core.session.end_session
    failure = {}

    def gated_end_session(*args, **kwargs):
        emit("entered_upstream_write")
        if command() != "write":
            raise RuntimeError("expected write command")
        if overlap is not None:
            if action != "extract_embed_barrier":
                wait_for(overlap / "phase_started")
            (overlap / "writer_started").write_text(marker)
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            failure.update(type=type(exc).__name__, message=str(exc))
            raise

    core.session.end_session = gated_end_session
    provider._multiprocess_upstream_failure = failure

if action == "initialize":
    emit("ready_to_initialize")
    if command() != "initialize":
        raise RuntimeError("expected initialize command")
    provider = initialize()
    emit("initialized", available=provider._db_path is not None)
    provider.shutdown()

elif action in {"extract", "extract_embed_barrier"}:
    provider = initialize()
    install_extract_barrier(provider)
    emit("ready")
    if command() != "start":
        raise RuntimeError("expected start command")
    result = provider.handle_tool_call(
        "cashew_extract",
        {"user_content": marker, "assistant_content": "stored marker " + marker},
    )
    emit(
        "result",
        result=json.loads(result),
        upstream_failure=provider._multiprocess_upstream_failure,
    )
    provider.shutdown()

elif action == "sync":
    provider = initialize()
    install_extract_barrier(provider)
    emit("ready")
    if command() != "start":
        raise RuntimeError("expected start command")
    provider.sync_turn(marker, "stored marker " + marker)
    emit("queued")
    assert provider._sync_queue is not None
    provider._sync_queue.join()
    provider.shutdown()
    emit("result", result={"drained": True})

elif action == "sleep":
    import plugins.memory.cashew.sleep_adapter as sleep
    original_guard = sleep.guard_sqlite_journal
    def paused_guard(conn):
        original_guard(conn)
        emit("maintenance_locked")
        if command() != "release":
            raise RuntimeError("expected release command")
        assert overlap is not None
        (overlap / "phase_started").write_text("sleep")
        wait_for(overlap / "writer_started")
    sleep.guard_sqlite_journal = paused_guard
    emit("ready")
    if command() != "start":
        raise RuntimeError("expected start command")
    result = sleep.run_sleep_cycle(str(home / "brain.db"), model_fn=None)
    emit("result", result=result)

elif action == "migration":
    import logging

    supervisor = VerifiedSupervisor("thenlper/gte-small")
    _embedding_service.model = "thenlper/gte-small"
    _embedding_service.dim = 384
    import core.config
    core.config.config.embedding_model = "thenlper/gte-small"
    core.embedding_service._KNOWN_DIMS["thenlper/gte-small"] = 384
    core.embedding_service._default_service = _embedding_service
    provider = CashewMemoryProvider()
    provider._config = CashewConfig(embedding_model="thenlper/gte-small")
    provider._embedding_supervisor = supervisor
    original_repair = provider._repair_embedding_dimension_locked
    migration_records = []
    class MigrationCapture(logging.Handler):
        def emit(self, record):
            if "embedding migration" in record.getMessage():
                migration_records.append({
                    "message": record.getMessage(),
                    "exception": str(record.exc_info[1]) if record.exc_info else "",
                })
    capture = MigrationCapture()
    logging.getLogger("plugins.memory.cashew").addHandler(capture)
    def paused_repair(db_path):
        emit("maintenance_locked")
        if overlap is not None:
            if command() != "release":
                raise RuntimeError("expected release command")
            (overlap / "phase_started").write_text("migration")
        return original_repair(db_path)
    provider._repair_embedding_dimension_locked = paused_repair
    emit("ready")
    if command() != "start":
        raise RuntimeError("expected start command")
    dimensions_before = provider._embedding_dimensions(home / "brain.db")
    try:
        provider._repair_embedding_dimension(home / "brain.db")
    finally:
        logging.getLogger("plugins.memory.cashew").removeHandler(capture)
    dimensions_after = provider._embedding_dimensions(home / "brain.db")
    before = [sorted(dimensions_before[0]), dimensions_before[1]]
    after = [sorted(dimensions_after[0]), dimensions_after[1]]
    if after == [[384], 384]:
        outcome = "migrated"
    elif overlap is not None and after == before and migration_records:
        outcome = "deferred_by_concurrent_writer"
    else:
        raise RuntimeError("unexpected post-migration dimensions: " + repr(after))
    emit(
        "result",
        result={
            "completed": True,
            "dimensions_before": before,
            "dimensions_after": after,
            "outcome": outcome,
            "migration_records": migration_records,
        },
    )

elif action == "lock":
    lock_path = Path(str(home / "brain.db") + ".sleep.lock")
    emit("ready")
    if command() != "hold":
        raise RuntimeError("expected hold command")
    with lock_path.open("a+") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        emit("maintenance_locked")
        if command() != "release":
            raise RuntimeError("expected release command")
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    emit("result", result={"released": True})

elif action == "sqlite_write_lock":
    emit("ready")
    if command() != "hold":
        raise RuntimeError("expected hold command")
    connection = sqlite3.connect(home / "brain.db")
    try:
        connection.execute("BEGIN IMMEDIATE")
        emit("sqlite_write_locked")
        if command() != "release":
            raise RuntimeError("expected release command")
        connection.commit()
    finally:
        connection.close()
    emit("result", result={"released": True})

elif action == "rapid_events":
    emit("ready")
    if command() != "start":
        raise RuntimeError("expected start command")
    emit("queued")
    emit("entered_upstream_write")
    emit("result", result={"rapid": True})

else:
    raise RuntimeError("unknown action: " + action)
"""


def _provision_profile(home: Path) -> None:
    """Create one profile before children race on its shared database."""
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "cashew.json"
    if config_path.exists():
        return
    staged = config_path.with_suffix(".json.tmp")
    staged.write_text(
        json.dumps(
            {
                "cashew_db_path": "brain.db",
                "llm_aux_role": None,
                "think_cycles": False,
            }
        )
    )
    staged.replace(config_path)


def _start(
    home: Path, marker: str, action: str, overlap: Path | None = None
) -> subprocess.Popen[str]:
    _provision_profile(home)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(home),
            marker,
            action,
            str(overlap) if overlap else "",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        env={**os.environ, "HOME": str(home / "user")},
    )
    try:
        expected = "ready_to_initialize" if action == "initialize" else "ready"
        event = _event(process, expected)
        assert event["core"]
        return process
    except BaseException as exc:
        detail = _child_error(process)
        _terminate(process)
        raise AssertionError(f"child startup failed: {detail}") from exc


def _event(
    process: subprocess.Popen[str], expected: str, timeout: float = _READY_TIMEOUT
) -> dict[str, Any]:
    pending: list[dict[str, Any]] = getattr(process, "_event_pending", [])
    buffer: bytearray = getattr(process, "_event_buffer", bytearray())
    process._event_pending = pending
    process._event_buffer = buffer
    deadline = time.monotonic() + timeout
    while True:
        for index, payload in enumerate(pending):
            if payload["event"] == expected:
                pending.pop(index)
                assert payload["core"], "child must import installed core.session"
                return payload
        parsed = False
        while b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer[:] = remainder
            if line:
                pending.append(json.loads(line))
                parsed = True
        if parsed:
            continue
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"child did not emit {expected!r} within {timeout}s"
        assert process.stdout is not None
        readable, _, _ = select.select([process.stdout.fileno()], [], [], remaining)
        if not readable:
            continue
        chunk = os.read(process.stdout.fileno(), 4096)
        assert chunk, f"child exited before {expected!r}: {_child_error(process)}"
        buffer.extend(chunk)


def _send(process: subprocess.Popen[str], value: str) -> None:
    assert process.stdin is not None
    process.stdin.write(f"{value}\n".encode())
    process.stdin.flush()


def _child_error(process: subprocess.Popen[str]) -> str:
    if process.poll() is None or process.stderr is None:
        return "child is still running"
    return process.stderr.read().decode(errors="replace")


def _terminate(process: subprocess.Popen[str]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _run_import_guard_probe(
    tmp_path: Path, mode: str
) -> subprocess.CompletedProcess[str]:
    """Run the module's real guard in an interpreter with a controlled core."""
    fake_root = tmp_path / mode
    fake_root.mkdir()
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_GUARD_RUNNER,
            str(Path(__file__).parents[1]),
            str(fake_root),
            mode,
        ],
        text=True,
        capture_output=True,
        timeout=_EXIT_TIMEOUT,
        check=False,
    )


def test_real_upstream_guard_skips_only_when_top_level_core_is_absent(
    tmp_path: Path,
) -> None:
    """An environment without Cashew skips this real-upstream-only module."""
    result = _run_import_guard_probe(tmp_path, "missing")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "missing-core-skipped"


def test_real_upstream_guard_does_not_hide_partial_core_installation(
    tmp_path: Path,
) -> None:
    """A present core package missing session fails through the child import."""
    result = _run_import_guard_probe(tmp_path, "partial")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "partial-core-failed-naturally"


def _result(process: subprocess.Popen[str]) -> dict[str, Any]:
    try:
        payload = _event(process, "result", timeout=_EXIT_TIMEOUT)
        assert process.wait(timeout=_EXIT_TIMEOUT) == 0, _child_error(process)
        result = payload["result"]
        if payload.get("upstream_failure"):
            result = {**result, "upstream_failure": payload["upstream_failure"]}
        return result
    finally:
        _terminate(process)


def _extract(home: Path, marker: str, overlap: Path | None = None) -> dict[str, Any]:
    process = _start(home, marker, "extract", overlap)
    try:
        _send(process, "start")
        _event(process, "entered_upstream_write")
        _send(process, "write")
        return _result(process)
    except BaseException:
        _terminate(process)
        raise


def _markers(db_path: Path) -> set[str]:
    connection = sqlite3.connect(db_path)
    try:
        return {
            row[0] for row in connection.execute("SELECT content FROM thought_nodes")
        }
    finally:
        connection.close()


def _assert_consistent(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert (
            connection.execute(
                "SELECT e.node_id FROM embeddings e "
                "LEFT JOIN thought_nodes n ON n.id = e.node_id WHERE n.id IS NULL"
            ).fetchall()
            == []
        )
        assert (
            connection.execute(
                "SELECT parent_id FROM derivation_edges "
                "WHERE parent_id NOT IN (SELECT id FROM thought_nodes) "
                "UNION ALL SELECT child_id FROM derivation_edges "
                "WHERE child_id NOT IN (SELECT id FROM thought_nodes)"
            ).fetchall()
            == []
        )
        has_vec_index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
        ).fetchone()
        if has_vec_index:
            import sqlite_vec

            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            embedding_dimensions = dict(
                connection.execute("SELECT node_id, LENGTH(vector) / 4 FROM embeddings")
            )
            vec_dimensions = dict(
                connection.execute(
                    "SELECT node_id, LENGTH(embedding) / 4 FROM vec_embeddings"
                )
            )
            assert set(vec_dimensions) == set(embedding_dimensions)
            assert vec_dimensions == embedding_dimensions
    finally:
        connection.close()


def _raise_if_exact_issue_191_stale_writer_mismatch(
    db_path: Path,
    *,
    seed_marker: str,
    writer_marker: str,
    expected_writer_vec_dimension: int | None,
) -> None:
    """Raise only for an exact verified #191 stale-writer divergence."""
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert (
            connection.execute(
                "SELECT e.node_id FROM embeddings e "
                "LEFT JOIN thought_nodes n ON n.id = e.node_id WHERE n.id IS NULL"
            ).fetchall()
            == []
        )
        assert (
            connection.execute(
                "SELECT parent_id FROM derivation_edges "
                "WHERE parent_id NOT IN (SELECT id FROM thought_nodes) "
                "UNION ALL SELECT child_id FROM derivation_edges "
                "WHERE child_id NOT IN (SELECT id FROM thought_nodes)"
            ).fetchall()
            == []
        )
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
        ).fetchone()
        import sqlite_vec

        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        ordinary_dimensions = dict(
            connection.execute("SELECT node_id, LENGTH(vector) / 4 FROM embeddings")
        )
        vec_dimensions = dict(
            connection.execute(
                "SELECT node_id, LENGTH(embedding) / 4 FROM vec_embeddings"
            )
        )
        seed_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM thought_nodes WHERE content = ?",
                (f"durable marker {seed_marker}",),
            )
        ]
        writer_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM thought_nodes WHERE content = ?",
                (f"durable marker {writer_marker}",),
            )
        ]
        assert len(seed_ids) == 1
        assert len(writer_ids) == 1
        seed_id, writer_id = seed_ids[0], writer_ids[0]
        assert ordinary_dimensions[seed_id] == 384
        assert vec_dimensions[seed_id] == 384
        assert ordinary_dimensions[writer_id] == 1024
        if expected_writer_vec_dimension is None:
            assert writer_id not in vec_dimensions
            assert set(ordinary_dimensions) == set(vec_dimensions) | {writer_id}
            assert all(
                ordinary_dimensions[node_id] == vec_dimensions[node_id]
                for node_id in vec_dimensions
            )
        else:
            assert vec_dimensions[writer_id] == expected_writer_vec_dimension
            assert set(ordinary_dimensions) == set(vec_dimensions)
            assert all(
                ordinary_dimensions[node_id] == vec_dimensions[node_id]
                for node_id in vec_dimensions
                if node_id != writer_id
            )
    finally:
        connection.close()
    if expected_writer_vec_dimension is None:
        raise _KnownIssue191MissingVecRowError(
            "issue #191: preinitialized 1024-dimensional writer omitted its vec row "
            "after a 384-dimensional migration"
        )
    raise _KnownIssue191StaleVecDimensionError(
        "issue #191: preinitialized writer stored a 1024-dimensional ordinary "
        f"embedding beside a {expected_writer_vec_dimension}-dimensional vec row"
    )


def test_two_processes_initialize_extract_and_recall_one_brain(tmp_path: Path) -> None:
    """Two real provider initializations and upstream writes overlap by barrier."""
    home = tmp_path / "shared"
    first = _start(home, "marker-alpha", "initialize")
    second = _start(home, "marker-beta", "initialize")
    try:
        _send(first, "initialize")
        _send(second, "initialize")
        assert _event(first, "initialized")["available"] is True
        assert _event(second, "initialized")["available"] is True
        assert first.wait(timeout=_EXIT_TIMEOUT) == 0, _child_error(first)
        assert second.wait(timeout=_EXIT_TIMEOUT) == 0, _child_error(second)
    finally:
        _terminate(first)
        _terminate(second)

    with sqlite3.connect(home / "brain.db") as conn:
        # Concurrent owners of the same verified identity must not fence each
        # other merely by starting in a different process.
        assert conn.execute(
            "SELECT value FROM hermes_provider_meta WHERE key='maintenance_epoch'"
        ).fetchone() == ("1",)

    first = _start(home, "marker-alpha", "extract")
    second = _start(home, "marker-beta", "extract")
    try:
        _send(first, "start")
        _send(second, "start")
        _event(first, "entered_upstream_write")
        _event(second, "entered_upstream_write")
        _send(first, "write")
        _send(second, "write")
        assert _result(first)["ok"] is True
        assert _result(second)["ok"] is True
    finally:
        _terminate(first)
        _terminate(second)

    stored = _markers(home / "brain.db")
    assert any("marker-alpha" in value for value in stored)
    assert any("marker-beta" in value for value in stored)
    _assert_consistent(home / "brain.db")

    provider = CashewMemoryProvider()
    try:
        provider.initialize("recall", hermes_home=str(home))
        assert "marker-alpha" in provider.prefetch("marker-alpha")
        assert "marker-beta" in provider.prefetch("marker-beta")
    finally:
        provider.shutdown()


def test_separate_profiles_do_not_share_persisted_records(tmp_path: Path) -> None:
    first_home = tmp_path / "one"
    second_home = tmp_path / "two"
    assert _extract(first_home, "only-one")["ok"] is True
    assert _extract(second_home, "only-two")["ok"] is True
    assert any("only-one" in value for value in _markers(first_home / "brain.db"))
    assert not any("only-two" in value for value in _markers(first_home / "brain.db"))
    _assert_consistent(first_home / "brain.db")
    _assert_consistent(second_home / "brain.db")


def test_abrupt_sync_worker_exit_only_requires_post_restart_durability(
    tmp_path: Path,
) -> None:
    """A killed worker has no durability promise; a restarted explicit write does."""
    home = tmp_path / "shared"
    killed = _start(home, "pre-kill", "sync")
    try:
        _send(killed, "start")
        _event(killed, "queued")
        _event(killed, "entered_upstream_write")
        killed.kill()
        assert killed.wait(timeout=_EXIT_TIMEOUT) != 0
    finally:
        _terminate(killed)

    assert _extract(home, "after-restart")["ok"] is True
    assert any("after-restart" in value for value in _markers(home / "brain.db"))
    _assert_consistent(home / "brain.db")


def test_event_reader_retains_rapid_out_of_order_child_events(tmp_path: Path) -> None:
    """A buffered child line cannot hide the next event from a later read."""
    child = _start(tmp_path / "events", "rapid", "rapid_events")
    try:
        _send(child, "start")
        assert (
            _event(child, "entered_upstream_write")["event"] == "entered_upstream_write"
        )
        assert _event(child, "queued")["event"] == "queued"
        assert _result(child) == {"rapid": True}
    finally:
        _terminate(child)


def test_sleep_lock_contention_has_bounded_named_skip(tmp_path: Path) -> None:
    """A different process holding maintenance lock produces a bounded skip."""
    home = tmp_path / "shared"
    holder = _start(home, "lock-holder", "lock")
    try:
        _send(holder, "hold")
        _event(holder, "maintenance_locked")
        started = time.monotonic()
        from plugins.memory.cashew.sleep_adapter import run_sleep_cycle

        assert run_sleep_cycle(str(home / "brain.db"), model_fn=None) == {}
        assert time.monotonic() - started < 3
        _send(holder, "release")
        assert _result(holder) == {"released": True}
    finally:
        _terminate(holder)


def test_real_extract_overlaps_sleep_and_preserves_its_marker(tmp_path: Path) -> None:
    """Extraction is rejected before upstream while sleep owns maintenance."""
    home = tmp_path / "shared"
    assert _extract(home, "sleep-seed-one")["ok"] is True
    assert _extract(home, "sleep-seed-two")["ok"] is True
    overlap = tmp_path / "extract-sleep"
    overlap.mkdir()
    sleeper = _start(home, "sleep", "sleep", overlap)
    writer = _start(home, "extract-during-sleep", "extract", overlap)
    try:
        _send(sleeper, "start")
        _event(sleeper, "maintenance_locked")
        _send(writer, "start")
        writer_result = _result(writer)
        assert writer_result["ok"] is False
        (overlap / "writer_started").write_text("rejected")
        _send(sleeper, "release")
        sleep_result = _result(sleeper)
        assert sleep_result.get("error") is None
        assert sleep_result["nodes_selected"] >= 2
    finally:
        _terminate(sleeper)
        _terminate(writer)

    assert not any(
        "extract-during-sleep" in value for value in _markers(home / "brain.db")
    )
    _assert_consistent(home / "brain.db")


def test_uncontended_migration_reaches_new_dimension_and_preserves_seed(
    tmp_path: Path,
) -> None:
    """The upstream migration must change a seeded 1024-dimensional brain."""
    home = tmp_path / "shared"
    assert _extract(home, "migration-uncontended-seed")["ok"] is True
    migration = _start(home, "migration", "migration")
    try:
        _send(migration, "start")
        _event(migration, "maintenance_locked")
        result = _result(migration)
        assert result["dimensions_before"] == [[1024], 1024]
        assert result["outcome"] == "migrated"
        assert result["dimensions_after"] == [[384], 384]
    finally:
        _terminate(migration)
    assert any(
        "migration-uncontended-seed" in value for value in _markers(home / "brain.db")
    )
    _assert_consistent(home / "brain.db")


def test_migration_maintenance_stages_real_old_writer_until_migration_completes(
    tmp_path: Path,
) -> None:
    """A concurrent writer is rejected before migration can begin upstream work."""
    home = tmp_path / "shared"
    assert _extract(home, "migration-seed")["ok"] is True
    overlap = tmp_path / "migration-write"
    overlap.mkdir()
    migration = _start(home, "migration", "migration", overlap)
    writer = _start(home, "writer-during-migration", "extract", overlap)
    try:
        _send(migration, "start")
        _event(migration, "maintenance_locked")
        _send(writer, "start")
        writer_result = _result(writer)
        assert writer_result["ok"] is False
        _send(migration, "release")
        migration_result = _result(migration)
        assert migration_result["completed"] is True
        assert migration_result["dimensions_before"] == [[1024], 1024]
        assert migration_result["outcome"] == "migrated"
        assert migration_result["dimensions_after"] == [[384], 384]
    finally:
        _terminate(migration)
        _terminate(writer)

    assert not any(
        "writer-during-migration" in value for value in _markers(home / "brain.db")
    )
    assert any("migration-seed" in value for value in _markers(home / "brain.db"))
    _assert_consistent(home / "brain.db")


def test_migration_overlap_preserves_observed_stale_vec_dimension(
    tmp_path: Path,
) -> None:
    """A writer cannot snapshot vectors until migration owns the profile."""
    home = tmp_path / "shared"
    assert _extract(home, "migration-overlap-seed")["ok"] is True
    overlap = tmp_path / "migration-overlap"
    overlap.mkdir()
    migration = _start(home, "migration", "migration", overlap)
    writer = _start(home, "writer-overlap", "extract_embed_barrier", overlap)
    try:
        _send(migration, "start")
        _event(migration, "maintenance_locked")
        _send(writer, "start")
        writer_result = _result(writer)
        assert writer_result["ok"] is False
        _send(migration, "release")
        migration_result = _result(migration)
        assert migration_result["completed"] is True
        assert migration_result["dimensions_before"] == [[1024], 1024]
        assert migration_result["outcome"] == "migrated"
        assert migration_result["dimensions_after"] == [[384], 384]
    finally:
        _terminate(migration)
        _terminate(writer)

    assert not any("writer-overlap" in value for value in _markers(home / "brain.db"))
    assert any(
        "migration-overlap-seed" in value for value in _markers(home / "brain.db")
    )
    _assert_consistent(home / "brain.db")


def test_real_admitted_writer_serializes_before_pinned_migration(
    tmp_path: Path,
) -> None:
    """A real upstream writer admitted before migration cannot publish stale vectors."""
    home = tmp_path / "shared"
    assert _extract(home, "migration-admitted-seed")["ok"] is True
    overlap = tmp_path / "admitted-overlap"
    overlap.mkdir()
    writer = _start(home, "admitted-old-writer", "extract_embed_barrier")
    migration = _start(home, "migration", "migration", overlap)
    try:
        _send(writer, "start")
        _event(writer, "entered_upstream_write")
        _send(writer, "write")
        _event(writer, "writer_embeddings_snapshotted")

        # The writer owns the ordinary graph admission while the real pinned
        # upstream embedding call is paused. Maintenance must wait for that
        # accepted call instead of migrating around it.
        _send(migration, "start")
        migration_result = _result(migration)
        assert migration_result["outcome"] == "deferred_by_concurrent_writer"
        assert not (overlap / "phase_started").exists()

        _send(writer, "resume_embeddings")
        writer_result = _result(writer)
        assert writer_result["ok"] is True
        migration = _start(home, "migration-after-writer", "migration", overlap)
        _send(migration, "start")
        _event(migration, "maintenance_locked")
        _send(migration, "release")
        migration_result = _result(migration)
        assert migration_result["completed"] is True
        assert migration_result["dimensions_after"] == [[384], 384]
    finally:
        _terminate(writer)
        _terminate(migration)

    with sqlite3.connect(home / "brain.db") as conn:
        assert conn.execute(
            "SELECT DISTINCT LENGTH(vector) / 4 FROM embeddings"
        ).fetchall() == [(384,)]
        assert conn.execute("SELECT DISTINCT model FROM embeddings").fetchall() == [
            ("thenlper/gte-small",)
        ]
    assert any("admitted-old-writer" in value for value in _markers(home / "brain.db"))
    _assert_consistent(home / "brain.db")


def test_real_sqlite_write_lock_reports_extract_failure_within_timeout(
    tmp_path: Path,
) -> None:
    """A persistence write blocked after initialization reports its actual failure."""
    home = tmp_path / "shared"
    assert _extract(home, "sqlite-lock-seed")["ok"] is True
    writer = _start(home, "sqlite-lock-writer", "extract")
    holder = _start(home, "sqlite-lock-holder", "sqlite_write_lock")
    try:
        _send(holder, "hold")
        _event(holder, "sqlite_write_locked")
        _send(writer, "start")
        _event(writer, "entered_upstream_write")
        started = time.monotonic()
        _send(writer, "write")
        result = _result(writer)
        assert time.monotonic() - started < _EXIT_TIMEOUT
        assert result["ok"] is False
        assert result["upstream_failure"]["type"] == "OperationalError"
        assert "database is locked" in result["upstream_failure"]["message"].lower()
        _send(holder, "release")
        assert _result(holder) == {"released": True}
    finally:
        _terminate(writer)
        _terminate(holder)
