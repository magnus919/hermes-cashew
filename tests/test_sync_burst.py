# tests/test_sync_burst.py
# Phase 4 Plan 04-04 Task 1: 50-turn burst — ROADMAP Phase 4 success criterion #2
# (revised wording: permissive, covers drop-oldest overflow).
#
# Criterion #2 (revised): "A 50-turn burst is handled without breaking the
# worker lifecycle — the hot path stays bounded, overflow degrades via
# drop-oldest with matching WARNINGs, and threading.active_count() returns
# to baseline after shutdown." (User-revised ROADMAP 2026-04-20.)
#
# Two tests satisfy the revised criterion by exercising two distinct axes:
#   1. With a larger queue, the worker CAN process all 50 distinct turns.
#   2. With the default queue, overflow happens cleanly and is auditable.
from __future__ import annotations

import queue as _queue_mod
import threading
import time
import types

from plugins.memory.cashew import CashewMemoryProvider


def _fake_end_session_ok(calls_list):
    def _fake(*args, **kwargs):
        calls_list.append(kwargs)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])
    return _fake


def _drain_queue(provider, budget_s=5.0):
    deadline = time.monotonic() + budget_s
    while (provider._sync_queue is not None
           and provider._sync_queue.unfinished_tasks > 0
           and time.monotonic() < deadline):
        time.sleep(0.01)
    return provider._sync_queue is None or provider._sync_queue.unfinished_tasks == 0


def _wait_for_thread_exit(baseline, budget_s=2.0):
    deadline = time.monotonic() + budget_s
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.01)
    return threading.active_count() == baseline


def test_burst_worker_drains_50_distinct_turns_with_larger_queue(tmp_path, monkeypatch):
    """Proves the worker can handle 50-turn volume when the queue is sized for it.

    Resizes `p._sync_queue` to Queue(maxsize=64) BEFORE the worker thread
    starts — the worker captures the queue reference at start time, so
    post-start swaps leave the worker blocked on the old queue's get().
    We achieve the pre-start swap by wrapping `_start_sync_worker`: the
    wrapper resizes the queue, then calls the original method. Enqueues
    50 turns, shuts down, asserts len(calls) == 50 and that
    threading.active_count() returns to baseline. This isolates the
    worker-throughput axis of the revised criterion #2 from the overflow
    axis.
    """
    calls: list = []
    monkeypatch.setattr("core.session.end_session",
                        _fake_end_session_ok(calls), raising=False)
    baseline = threading.active_count()
    p = CashewMemoryProvider()
    # Wrap _start_sync_worker so we can resize the queue AFTER initialize
    # built it but BEFORE the worker thread captures its reference. The
    # plan's original approach (swap after initialize()) fails because the
    # worker is already blocked on the old queue's .get(). Rule 1 fix.
    _orig_start = p._start_sync_worker
    def _start_with_larger_queue():
        p._sync_queue = _queue_mod.Queue(maxsize=64)
        _orig_start()
    p._start_sync_worker = _start_with_larger_queue  # type: ignore[method-assign]
    p.save_config({}, str(tmp_path))
    p.initialize("burst-large", hermes_home=str(tmp_path))
    try:
        start = time.monotonic()
        for i in range(50):
            p.sync_turn(f"u{i}", f"a{i}")
        enqueue_elapsed = time.monotonic() - start
        assert enqueue_elapsed < 1.0, (
            f"50 sync_turn calls took {enqueue_elapsed*1000:.0f}ms; "
            "hot-path criterion wants <1s"
        )
        assert _drain_queue(p, budget_s=5.0), (
            f"queue did not drain; unfinished={p._sync_queue.unfinished_tasks}"
        )
        # Strict: with maxsize=64 no overflow can happen — all 50 turns processed.
        assert len(calls) == 50, (
            f"expected all 50 distinct turns processed with larger queue; got {len(calls)}"
        )
    finally:
        p.shutdown()
    assert _wait_for_thread_exit(baseline, budget_s=2.0), (
        f"thread leak after larger-queue burst: "
        f"active={threading.active_count()} baseline={baseline}"
    )


def test_burst_default_queue_drops_oldest_cleanly(tmp_path, monkeypatch, caplog):
    """Proves the production overflow policy degrades gracefully under 50-turn burst.

    Uses the default maxsize=16 queue. Enqueues 50 turns as fast as possible
    so drop-oldest fires repeatedly. Asserts:
      - len(calls) <= 50 (no duplicates or extras).
      - Drop-oldest WARNING count matches the arithmetic:
        (enqueued - drained_before_drops), equivalently (50 - len(calls)).
      - p._dropped_turn_count matches that same count.
      - threading.active_count() returns to baseline after shutdown.
    """
    import logging
    calls: list = []
    monkeypatch.setattr("core.session.end_session",
                        _fake_end_session_ok(calls), raising=False)
    baseline = threading.active_count()
    p = CashewMemoryProvider()
    p.save_config({}, str(tmp_path))
    p.initialize("burst-default", hermes_home=str(tmp_path))
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            start = time.monotonic()
            for i in range(50):
                p.sync_turn(f"u{i}", f"a{i}")
            enqueue_elapsed = time.monotonic() - start
            assert enqueue_elapsed < 1.0, (
                f"50 sync_turn calls took {enqueue_elapsed*1000:.0f}ms; "
                "hot-path criterion wants <1s even under overflow"
            )
            assert _drain_queue(p, budget_s=5.0), (
                f"queue did not drain; unfinished={p._sync_queue.unfinished_tasks}"
            )
        # Drops happened; processed count is <=50.
        assert len(calls) <= 50, f"processed more turns than enqueued: {len(calls)}"
        drop_warnings = [
            r for r in caplog.records
            if r.levelname == "WARNING" and "overflow" in r.getMessage().lower()
        ]
        expected_drops = 50 - len(calls)
        assert len(drop_warnings) == expected_drops, (
            f"drop WARNING count ({len(drop_warnings)}) != "
            f"(enqueued - drained = {expected_drops})"
        )
        # Provider exposes dropped-turn counter that matches the WARNINGs.
        assert getattr(p, "_dropped_turn_count", None) == expected_drops, (
            f"_dropped_turn_count ({getattr(p, '_dropped_turn_count', 'MISSING')}) "
            f"!= WARNING count ({expected_drops})"
        )
    finally:
        p.shutdown()
    assert _wait_for_thread_exit(baseline, budget_s=2.0), (
        f"thread leak after default-queue burst: "
        f"active={threading.active_count()} baseline={baseline}"
    )
