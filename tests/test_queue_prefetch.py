"""Tests for CashewMemoryProvider.queue_prefetch and its supporting machinery."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from plugins.memory.cashew import CashewMemoryProvider


def _provider_with_mock_config():
    """Create a provider that has _config wired with prefetch defaults."""
    provider = CashewMemoryProvider()
    cfg = MagicMock()
    cfg.recall_k = 5
    cfg.prefetch_k = 3
    cfg.prefetch_cues = 3
    provider._config = cfg
    provider._db_path = MagicMock()
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


def test_queue_prefetch_skips_empty_query():
    """queue_prefetch must return immediately for empty/trivial queries."""
    provider = _provider_with_mock_config()
    provider.queue_prefetch("")
    assert provider._prefetch_pending is None


# ── warm cache integration with prefetch() ───────────────────────────────


def test_prefetch_uses_warm_cache_on_exact_match():
    """prefetch() must return cached context when the raw query matches exactly."""
    provider = _provider_with_mock_config()
    provider._warm_cache["leader election raft"] = "cached: raft consensus"
    result = provider.prefetch("leader election raft")
    assert result == "cached: raft consensus"
    assert provider._warm_cache == {}


def test_prefetch_uses_warm_cache_on_substring_match():
    """prefetch() must match when the query is a substring of the cached cue."""
    provider = _provider_with_mock_config()
    provider._warm_cache["we discussed the Raft consensus protocol earlier"] = (
        "cached: raft details"
    )
    result = provider.prefetch("Raft consensus")
    assert result == "cached: raft details"


def test_prefetch_uses_warm_cache_on_word_overlap():
    """prefetch() must match when ≥2 significant words overlap between cue and query."""
    provider = _provider_with_mock_config()
    provider._warm_cache["distributed database write throughput"] = (
        "cached: write concerns"
    )
    result = provider.prefetch("database write performance")
    assert result == "cached: write concerns"


def test_prefetch_cache_miss_falls_through():
    """prefetch() must fall through on cache miss, not return cached unrelated content."""
    provider = _provider_with_mock_config()
    provider._warm_cache["unrelated topic"] = "cached: unrelated"
    # No retrieve_recursive_bfs mocked, so falls to keyword search → returns ""
    result = provider.prefetch("completely different subject xyzw")
    assert result != "cached: unrelated"
    assert provider._warm_cache == {}


# ── staging slot handoff ─────────────────────────────────────────────────


def test_prefetch_swaps_pending_into_warm_cache():
    """prefetch() must atomically swap _prefetch_pending into _warm_cache."""
    provider = _provider_with_mock_config()
    provider._prefetch_pending = "staged context"
    result = provider.prefetch("anything")
    assert result == "staged context"
    assert provider._prefetch_pending is None


def test_prefetch_half_state_skips_warm_cache():
    """prefetch() must return '' when _config is None, ignoring warm cache."""
    provider = _half_state_provider()
    provider._warm_cache["hello"] = "cached"
    result = provider.prefetch("hello")
    assert result == ""


# ── background thread mechanics ──────────────────────────────────────────


def test_queue_prefetch_dispatches_background_thread():
    """queue_prefetch must start a daemon thread."""
    provider = _provider_with_mock_config()
    provider.queue_prefetch("test query")
    threads = [
        t
        for t in threading.enumerate()
        if t.name and t.name.startswith("cashew-prefetch-")
    ]
    assert len(threads) >= 1
    assert all(t.daemon for t in threads)
