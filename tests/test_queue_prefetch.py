"""Tests for CashewMemoryProvider.queue_prefetch and its supporting machinery."""

from __future__ import annotations

import logging
import queue
import threading
import time
from unittest.mock import MagicMock

from plugins.memory.cashew import CashewMemoryProvider


def _provider_with_mock_config(tmp_path):
    """Create a provider that has _config wired with prefetch defaults."""
    provider = CashewMemoryProvider()
    cfg = MagicMock()
    cfg.recall_k = 5
    cfg.prefetch_k = 3
    cfg.prefetch_cues = 3
    cfg.sync_queue_timeout = 1.0
    provider._config = cfg
    provider._db_path = tmp_path / "brain.db"
    provider._model_fn = None  # no LLM for unit tests
    provider._session_id = "test-session"
    return provider


def _half_state_provider():
    """Create a provider in half-state (_config is None)."""
    provider = CashewMemoryProvider()
    provider._config = None
    provider._db_path = None
    return provider


# ── queue_prefetch half-state guards ──────────────────────────────────────


def test_queue_prefetch_skips_when_config_is_none():
    """queue_prefetch must be a silent no-op when _config is None."""
    provider = _half_state_provider()
    provider._prefetch_pending = "stale"
    provider.queue_prefetch("hello")
    assert provider._prefetch_pending == "stale"


def test_queue_prefetch_skips_empty_query(tmp_path):
    """queue_prefetch must return immediately for empty/trivial queries."""
    provider = _provider_with_mock_config(tmp_path)
    provider.queue_prefetch("")
    assert provider._prefetch_pending is None


# ── warm cache integration with prefetch() ───────────────────────────────


def test_prefetch_uses_warm_cache_on_exact_match(tmp_path):
    """prefetch() must return cached context when the raw query matches exactly."""
    provider = _provider_with_mock_config(tmp_path)
    provider._warm_cache["leader election raft"] = "cached: raft consensus"
    result = provider.prefetch("leader election raft")
    assert result == "cached: raft consensus"
    assert provider._warm_cache == {}


def test_prefetch_uses_warm_cache_on_substring_match(tmp_path):
    """prefetch() must match when the query is a substring of the cached cue."""
    provider = _provider_with_mock_config(tmp_path)
    provider._warm_cache["we discussed the Raft consensus protocol earlier"] = (
        "cached: raft details"
    )
    result = provider.prefetch("Raft consensus")
    assert result == "cached: raft details"


def test_prefetch_uses_warm_cache_on_word_overlap(tmp_path):
    """prefetch() must match when ≥2 significant words overlap between cue and query."""
    provider = _provider_with_mock_config(tmp_path)
    provider._warm_cache["distributed database write throughput"] = (
        "cached: write concerns"
    )
    result = provider.prefetch("database write performance")
    assert result == "cached: write concerns"


def test_prefetch_cache_miss_falls_through(tmp_path):
    """prefetch() must fall through on cache miss, not return cached unrelated content."""
    provider = _provider_with_mock_config(tmp_path)
    provider._warm_cache["unrelated topic"] = "cached: unrelated"
    # No retrieve_recursive_bfs mocked, so falls to keyword search → returns ""
    result = provider.prefetch("completely different subject xyzw")
    assert result != "cached: unrelated"
    assert provider._warm_cache == {}


# ── staging slot handoff ─────────────────────────────────────────────────


def test_prefetch_swaps_pending_into_warm_cache(tmp_path):
    """Pending context retains its cue and uses normal relevance matching."""
    provider = _provider_with_mock_config(tmp_path)
    provider._prefetch_generation = 1
    provider._prefetch_pending = (
        1,
        "test-session",
        ["database migration"],
        "staged context",
    )
    result = provider.prefetch("database migration plan")
    assert result == "staged context"
    assert provider._prefetch_pending is None


def test_prefetch_does_not_relabel_unrelated_pending_context(tmp_path):
    provider = _provider_with_mock_config(tmp_path)
    provider._prefetch_generation = 1
    provider._prefetch_pending = (
        1,
        "test-session",
        ["pizza preferences"],
        "cached pizza context",
    )

    result = provider.prefetch("production database migration")

    assert result != "cached pizza context"


def test_stale_prefetch_worker_cannot_replace_newer_generation(tmp_path):
    provider = _provider_with_mock_config(tmp_path)
    provider._prefetch_generation = 2

    provider._stage_prefetch_result(1, "test-session", ["old"], "old context")
    assert provider._prefetch_pending is None

    provider._stage_prefetch_result(2, "test-session", ["new"], "new context")
    assert provider._prefetch_pending == (
        2,
        "test-session",
        ["new"],
        "new context",
    )


def test_prefetch_half_state_skips_warm_cache():
    """prefetch() must return '' when _config is None, ignoring warm cache."""
    provider = _half_state_provider()
    provider._warm_cache["hello"] = "cached"
    result = provider.prefetch("hello")
    assert result == ""


# ── background thread mechanics ──────────────────────────────────────────


def test_queue_prefetch_dispatches_tracked_background_thread(tmp_path, monkeypatch):
    """queue_prefetch tracks its daemon until the warmup exits."""
    provider = _provider_with_mock_config(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def blocked_retrieval(**kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs", blocked_retrieval, raising=False
    )
    provider.queue_prefetch("test query")
    assert started.wait(timeout=1.0)
    threads = tuple(provider._prefetch_threads)
    assert len(threads) == 1
    assert threads[0].daemon

    release.set()
    threads[0].join(timeout=1.0)
    assert not threads[0].is_alive()
    assert provider._prefetch_threads == set()


def test_queue_prefetch_rejects_new_worker_after_shutdown_starts(tmp_path):
    provider = _provider_with_mock_config(tmp_path)
    provider._shutdown_started.set()

    provider.queue_prefetch("too late")

    assert provider._prefetch_threads == set()


def test_shutdown_waits_for_accepted_prefetch_before_clearing_state(
    tmp_path, monkeypatch
):
    provider = _provider_with_mock_config(tmp_path)
    provider._sync_queue = queue.Queue()
    started = threading.Event()
    release = threading.Event()

    def blocked_retrieval(**kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs", blocked_retrieval, raising=False
    )
    provider.queue_prefetch("test query")
    assert started.wait(timeout=1.0)

    shutdown_thread = threading.Thread(target=provider.shutdown)
    shutdown_thread.start()
    assert provider._shutdown_started.wait(timeout=1.0)
    time.sleep(0.02)

    assert shutdown_thread.is_alive()
    assert provider._db_path == tmp_path / "brain.db"

    release.set()
    shutdown_thread.join(timeout=1.0)

    assert not shutdown_thread.is_alive()
    assert provider._db_path is None
    assert provider._config is None
    assert provider._prefetch_threads == set()


def test_shutdown_timeout_retains_state_until_prefetch_exits(
    tmp_path, monkeypatch, caplog
):
    provider = _provider_with_mock_config(tmp_path)
    provider._config.sync_queue_timeout = 0.05
    provider._sync_queue = queue.Queue()
    started = threading.Event()
    release = threading.Event()

    def blocked_retrieval(**kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs", blocked_retrieval, raising=False
    )
    provider.queue_prefetch("test query")
    assert started.wait(timeout=1.0)

    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        provider.shutdown()

    assert provider._db_path == tmp_path / "brain.db"
    assert provider._config is not None
    assert "prefetch worker(s) did not exit" in caplog.text

    release.set()
    deadline = time.monotonic() + 1.0
    while provider._db_path is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert provider._db_path is None
    assert provider._config is None
    assert provider._prefetch_threads == set()
