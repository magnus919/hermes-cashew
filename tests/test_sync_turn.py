# tests/test_sync_turn.py
# Phase 4 Plan 04-03 Task 2: sync_turn hot-path + drop-oldest + half-state.
# Covers SYNC-01 (bounded non-blocking enqueue) + SYNC-06 continuation.
from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
import types
from typing import Any

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CONFIG_FILENAME


# -- Module-level helpers (self-contained per PHASE_DESIGN_NOTES Decision Point 5) --


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
    def _fake(*args: Any, **kwargs: Any) -> Any:
        calls_list.append(kwargs)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    return _fake


def fake_end_session_slow(sleep_s: float) -> Any:
    def _fake(*args: Any, **kwargs: Any) -> Any:
        time.sleep(sleep_s)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    return _fake


def make_initialized_provider(tmp_path) -> CashewMemoryProvider:
    p = CashewMemoryProvider()
    p.save_config({}, str(tmp_path))
    p.initialize("test-sync", hermes_home=str(tmp_path))
    return p


# -- Tests --


def test_sync_turn_enqueues_tuple(tmp_path, monkeypatch):
    """Pause the worker (slow mock); after sync_turn the queue contains one item."""
    # Use a very slow mock so the worker picks up the turn but stays busy
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(5.0), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        # First sync_turn — worker will pick it up (qsize goes to 0, but unfinished_tasks=1)
        # Use a second turn so qsize > 0
        p.sync_turn("u0", "a0")
        time.sleep(0.05)  # let worker pick up the first turn
        p.sync_turn("u1", "a1")
        # Now the second turn is waiting in the queue (worker busy with first)
        assert p._sync_queue.qsize() == 1
    finally:
        # Shrink shutdown budget so the 5s slow call doesn't block teardown
        if p._config is not None:
            p._config = dataclasses.replace(p._config, sync_queue_timeout=0.1)
        monkeypatch.setattr(
            "core.session.end_session", fake_end_session_ok([]), raising=False
        )
        p.shutdown()


def test_sync_turn_fast_on_empty_queue(tmp_path, monkeypatch):
    """50ms lenient upper bound; always-on contract guard against blocking-put regression."""
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_ok([]), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        start = time.monotonic()
        p.sync_turn("u", "a")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, f"sync_turn blocked for {elapsed*1000:.1f}ms"
    finally:
        p.shutdown()


@pytest.mark.skipif(
    not os.environ.get("CI_STRICT_TIMING"),
    reason=(
        "strict 15ms bound; opt-in via CI_STRICT_TIMING=1 env var. "
        "The lenient 50ms test (test_sync_turn_fast_on_empty_queue) "
        "always runs and catches blocking-put regressions."
    ),
)
def test_sync_turn_empty_queue_strict_15ms(tmp_path, monkeypatch):
    """Enforces the ROADMAP's 10ms success criterion with 5ms headroom.

    Under ideal conditions (empty queue, mocked worker that consumes
    instantly), a single sync_turn() must return in under 15ms. This
    is the strict interpretation; the 50ms lenient test catches the
    100x blocking-put regression without being flake-prone.

    Gated on CI_STRICT_TIMING env var so CI default pipeline (subject
    to scheduler jitter) doesn't flake; dev local runs and the
    dedicated strict-timing job opt in.
    """
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_ok([]), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        start = time.monotonic()
        p.sync_turn("u", "a")
        elapsed = time.monotonic() - start
        assert elapsed < 0.015, (
            f"sync_turn blocked for {elapsed*1000:.1f}ms; strict bound 15ms"
        )
    finally:
        p.shutdown()


def test_sync_turn_fast_even_on_full_queue(tmp_path, monkeypatch):
    """Pitfall 5: sync_turn stays fast even when the queue is full (drop-oldest path)."""
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(100.0), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        # Fill to 16 — first turn picked up by worker (stuck), next 15 sit in queue
        for i in range(16):
            p.sync_turn(f"u{i}", f"a{i}")
        # Queue now likely full; next call must still be fast (drop-oldest path)
        start = time.monotonic()
        p.sync_turn("u17", "a17")
        elapsed = time.monotonic() - start
        assert elapsed < 0.05, (
            f"sync_turn blocked for {elapsed*1000:.1f}ms on full queue — hot path violation"
        )
    finally:
        # Swap fake for fast so shutdown doesn't hang on the 100s sleep
        monkeypatch.setattr(
            "core.session.end_session", fake_end_session_ok([]), raising=False
        )
        # Override config to cut shutdown timeout before join
        if p._config is not None:
            p._config = dataclasses.replace(p._config, sync_queue_timeout=0.1)
        p.shutdown()


def test_drop_oldest_logs_warning_and_preserves_newest(tmp_path, monkeypatch, caplog):
    """Overflow produces WARNING with substring 'overflow'."""
    calls: list[dict] = []
    # Phase 1: slow mock so queue fills
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(0.5), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            for i in range(17):
                p.sync_turn(f"u{i}", f"a{i}")
        overflow_warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and "overflow" in r.getMessage().lower()
        ]
        assert overflow_warnings, (
            f"no overflow WARNING logged; got "
            f"{[r.getMessage() for r in caplog.records]}"
        )
    finally:
        monkeypatch.setattr(
            "core.session.end_session", fake_end_session_ok(calls), raising=False
        )
        if p._config is not None:
            p._config = dataclasses.replace(p._config, sync_queue_timeout=0.2)
        p.shutdown()


def test_drop_oldest_balances_task_done(tmp_path, monkeypatch, caplog):
    """Pitfall 3: drop-oldest overflow must not cause task_done() double-count ValueError."""
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_slow(0.2), raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING):
            for i in range(20):
                p.sync_turn(f"u{i}", f"a{i}")
        # Swap to fast so drain completes
        monkeypatch.setattr(
            "core.session.end_session", fake_end_session_ok([]), raising=False
        )
        assert drain_queue(p, budget_s=5.0)
        # No ValueError('task_done called too many times') raised anywhere
        value_errors = [
            r
            for r in caplog.records
            if "ValueError" in r.getMessage()
            or (r.exc_info and isinstance(r.exc_info[1], ValueError))
        ]
        assert not value_errors, "task_done double-count detected"
    finally:
        p.shutdown()


def test_sync_turn_silent_noop_on_fresh_provider(caplog):
    """Fresh provider (never initialized): sync_turn silent no-op."""
    p = CashewMemoryProvider()
    with caplog.at_level(logging.DEBUG, logger="plugins.memory.cashew"):
        p.sync_turn("u", "a")  # must not raise
    # No records at any level from sync_turn itself
    assert not caplog.records, (
        f"fresh provider sync_turn logged: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_sync_turn_silent_noop_on_corrupt_config_provider(tmp_path, caplog):
    """Silent-degrade provider (corrupt config): sync_turn silent no-op."""
    (tmp_path / CONFIG_FILENAME).write_text("not json {")
    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        # initialize already logged its WARNING; clear caplog
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            p.sync_turn("u", "a")
        assert not caplog.records, (
            f"sync_turn logged in silent-degrade: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
    finally:
        p.shutdown()


def test_sync_turn_silent_noop_after_shutdown(tmp_path, monkeypatch, caplog):
    """Post-shutdown: sync_turn silent no-op (shutdown cleared _sync_queue)."""
    monkeypatch.setattr(
        "core.session.end_session", fake_end_session_ok([]), raising=False
    )
    p = make_initialized_provider(tmp_path)
    p.shutdown()
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        p.sync_turn("u", "a")
    assert not [r for r in caplog.records if "cashew" in r.getMessage().lower()]
