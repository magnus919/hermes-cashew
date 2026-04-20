# tests/test_sync_worker.py
# Phase 4 Plan 04-03 Task 1: worker lifecycle + poisoned turn + shutdown hang.
# Covers SYNC-02, SYNC-05, SYNC-06, ABC-05, ABC-06.
from __future__ import annotations

import logging
import threading
import time
import types
from typing import Any

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CONFIG_FILENAME


# -- Module-level helpers (copy-paste idiom per PHASE_DESIGN_NOTES Decision Point 5) --


def drain_queue(provider: CashewMemoryProvider, budget_s: float = 2.0) -> bool:
    """Poll unfinished_tasks until 0 or budget expires. Returns True iff drained."""
    deadline = time.monotonic() + budget_s
    while (
        provider._sync_queue is not None
        and provider._sync_queue.unfinished_tasks > 0
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    return provider._sync_queue is None or provider._sync_queue.unfinished_tasks == 0


def wait_for_thread_exit(baseline: int, budget_s: float = 2.0) -> bool:
    """Poll threading.active_count until it equals baseline or budget expires."""
    deadline = time.monotonic() + budget_s
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.01)
    return threading.active_count() == baseline


def fake_end_session_ok(calls_list: list) -> Any:
    """Mock end_session: records kwargs, returns ExtractionResult-shaped object."""

    def _fake(*args: Any, **kwargs: Any) -> Any:
        calls_list.append(kwargs)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    return _fake


def fake_end_session_slow(sleep_s: float) -> Any:
    """Mock end_session that sleeps — for burst/overflow/hang testing."""

    def _fake(*args: Any, **kwargs: Any) -> Any:
        time.sleep(sleep_s)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    return _fake


def fake_end_session_raises(exc: Exception) -> Any:
    """Mock end_session that raises — for silent-degrade testing."""

    def _fake(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return _fake


def make_initialized_provider(tmp_path) -> CashewMemoryProvider:
    """Save default config + initialize. Returns a provider with worker alive."""
    p = CashewMemoryProvider()
    p.save_config({}, str(tmp_path))
    p.initialize("test-sync", hermes_home=str(tmp_path))
    return p


# -- Tests --


def test_worker_starts_after_initialize_on_happy_path(tmp_path):
    baseline = threading.active_count()
    p = make_initialized_provider(tmp_path)
    try:
        assert threading.active_count() == baseline + 1
        assert p._sync_worker is not None
        assert p._sync_worker.is_alive()
        assert p._sync_worker.daemon is False
        assert p._sync_worker.name.startswith("cashew-sync-")
        assert "test-sync" in p._sync_worker.name
    finally:
        p.shutdown()
        wait_for_thread_exit(baseline)


def test_worker_not_started_on_corrupt_config(tmp_path, caplog):
    (tmp_path / CONFIG_FILENAME).write_text("not json {")
    baseline = threading.active_count()
    p = CashewMemoryProvider()
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        p.initialize("s", hermes_home=str(tmp_path))
    try:
        assert threading.active_count() == baseline
        assert p._sync_worker is None
        # sync_turn is silent no-op
        p.sync_turn("u", "a")
        assert p._sync_worker is None  # still None after call
    finally:
        p.shutdown()


def test_worker_processes_single_turn(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_ok(calls), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        p.sync_turn("hi", "hello")
        assert drain_queue(
            p, budget_s=2.0
        ), f"drain timeout; unfinished={p._sync_queue.unfinished_tasks}"
        assert len(calls) == 1
        assert calls[0]["session_id"] == "test-sync"
        assert calls[0]["conversation_text"] == "User: hi\nAssistant: hello"
        assert calls[0]["db_path"] == str(p._db_path)
        assert calls[0]["model_fn"] is None
    finally:
        p.shutdown()


def test_worker_processes_multiple_turns_fifo(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_ok(calls), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        for i in range(5):
            p.sync_turn(f"u{i}", f"a{i}")
        assert drain_queue(p, budget_s=2.0)
        assert len(calls) == 5
        for i, call in enumerate(calls):
            assert f"User: u{i}" in call["conversation_text"]
    finally:
        p.shutdown()


def test_poisoned_turn_does_not_break_worker(tmp_path, monkeypatch, caplog):
    baseline = threading.active_count()
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        if len(calls) == 7:
            raise RuntimeError("synthetic poison")
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    monkeypatch.setattr("core.session.end_session", _fake, raising=False)
    p = make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            for i in range(16):
                p.sync_turn(f"u{i}", f"a{i}")
            assert drain_queue(p, budget_s=3.0)
        assert (
            len(calls) == 16
        ), f"expected all 16 turns attempted; got {len(calls)}"
        # ONE WARNING from the poisoned turn
        worker_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "turn failed" in r.getMessage()
        ]
        assert len(worker_warnings) == 1
        assert worker_warnings[0].exc_info is not None
        # Worker still alive
        assert p._sync_worker.is_alive()
    finally:
        p.shutdown()
        # Assert the worker thread has fully exited before this test returns.
        # Without this, a future regression where the worker crashes into a
        # wedged state (but shutdown still posts sentinel that exits the
        # wedge) would pass silently — active_count leak would contaminate
        # the next test's baseline capture.
        assert wait_for_thread_exit(baseline), (
            f"worker thread did not exit after poisoned-turn test: "
            f"active={threading.active_count()} baseline={baseline}"
        )


def test_shutdown_posts_sentinel_and_joins(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_ok([]), raising=False
    )
    baseline = threading.active_count()
    p = make_initialized_provider(tmp_path)
    p.shutdown()
    assert wait_for_thread_exit(
        baseline
    ), f"thread leak: {threading.active_count()} vs {baseline}"
    assert p._sync_worker is None
    assert p._sync_queue is None
    assert p._config is None
    assert p._db_path is None


def test_shutdown_hung_worker_logs_warning_no_raise(tmp_path, monkeypatch, caplog):
    """0.3s sleep gives the shutdown timeout (0.1s) time to fire while
    guaranteeing the thread exits cleanly before the next test starts."""
    baseline = threading.active_count()
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(0.3), raising=False
    )
    p = CashewMemoryProvider()
    p.save_config({"sync_queue_timeout": 0.1}, str(tmp_path))
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        p.sync_turn("u", "a")
        # Give the worker a moment to pick up the turn and start sleeping
        time.sleep(0.05)
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            # shutdown returns without raising even though worker won't exit in 0.1s
            start = time.monotonic()
            p.shutdown()  # MUST NOT RAISE
            elapsed = time.monotonic() - start
        assert elapsed >= 0.1, f"shutdown returned too quickly: {elapsed}s"
        assert elapsed < 2.0, f"shutdown took too long: {elapsed}s"
        hang_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "did not exit within" in r.getMessage()
        ]
        assert len(hang_warnings) == 1
    finally:
        # Wait for the worker thread to exit (0.3s sleep completes, then
        # the worker returns to queue.get() and exits on sentinel). This
        # prevents thread-leak contamination of the next test's baseline.
        assert wait_for_thread_exit(baseline, budget_s=1.0), (
            f"hung worker did not exit within 1s budget: "
            f"active={threading.active_count()} baseline={baseline}"
        )


def test_on_session_end_polls_and_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(0.1), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        for i in range(3):
            p.sync_turn(f"u{i}", f"a{i}")
        start = time.monotonic()
        p.on_session_end([])
        elapsed = time.monotonic() - start
        assert elapsed < 1.3, f"on_session_end took too long: {elapsed}s"
        assert p._sync_queue.unfinished_tasks == 0
        # Worker still alive — on_session_end does not stop it
        assert p._sync_worker.is_alive()
    finally:
        p.shutdown()


def test_on_session_end_bounded_by_sync_queue_timeout(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(2.0), raising=False
    )
    p = CashewMemoryProvider()
    p.save_config({"sync_queue_timeout": 0.3}, str(tmp_path))
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        p.sync_turn("u", "a")
        time.sleep(0.05)  # let worker pick up the turn
        start = time.monotonic()
        p.on_session_end([])
        elapsed = time.monotonic() - start
        assert elapsed >= 0.25, f"poll exited too early: {elapsed}s"
        assert elapsed < 0.7, f"poll exceeded bound: {elapsed}s"
        assert p._sync_worker.is_alive()
        # Clear caplog BEFORE shutdown so the expected "did not exit"
        # WARNING from shutdown's bounded join (the in-flight 2.0s slow
        # call will not complete within shutdown's 0.3s timeout) does
        # not bleed into any caller-side assertions about on_session_end.
        caplog.clear()
    finally:
        # Swap fake for fast so shutdown doesn't hang on the hung turn
        # (the in-flight slow call will block shutdown's bounded join — 0.3s —
        # but we set the timeout low to keep the test fast)
        monkeypatch.setattr(
            "core.session.end_session", fake_end_session_ok([]), raising=False
        )
        p.shutdown()


def test_pre_initialize_shutdown_still_safe_noop(tmp_path, caplog):
    baseline = threading.active_count()
    p = CashewMemoryProvider()
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        p.shutdown()
    assert threading.active_count() == baseline
    # No WARNING logged for safe no-op path
    cashew_warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "cashew" in r.getMessage().lower()
    ]
    assert not cashew_warnings
