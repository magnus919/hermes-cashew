"""Tests for CashewMemoryProvider.queue_prefetch and its supporting machinery."""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig
from plugins.memory.cashew.metrics import _METRICS


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


def _stage_prefetch_result(
    provider: CashewMemoryProvider, cues: list[str], nodes: list[dict]
) -> None:
    """Stage synthetic nodes through the provider's production cache handoff."""
    identity = provider._prefetch_request_identity(
        session_id=provider._session_id,
        generation=provider._prefetch_generation,
        domain=None,
        tag=None,
        exclude_tags=None,
    )
    provider._stage_prefetch_result(identity, cues, nodes)


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
    _stage_prefetch_result(
        provider,
        ["leader election raft"],
        [{"id": "raft", "content": "cached raft consensus"}],
    )
    result = provider.prefetch("leader election raft")
    assert "cached raft consensus" in result
    assert provider._warm_cache == {}


def test_prefetch_uses_warm_cache_on_substring_match(tmp_path):
    """prefetch() must match when the query is a substring of the cached cue."""
    provider = _provider_with_mock_config(tmp_path)
    _stage_prefetch_result(
        provider,
        ["we discussed the Raft consensus protocol earlier"],
        [{"id": "raft", "content": "cached raft details"}],
    )
    result = provider.prefetch("Raft consensus")
    assert "cached raft details" in result


def test_prefetch_uses_warm_cache_on_word_overlap(tmp_path):
    """prefetch() must match when ≥2 significant words overlap between cue and query."""
    provider = _provider_with_mock_config(tmp_path)
    _stage_prefetch_result(
        provider,
        ["distributed database write throughput"],
        [{"id": "throughput", "content": "cached write concerns"}],
    )
    result = provider.prefetch("database write performance")
    assert "cached write concerns" in result


def test_prefetch_cache_miss_falls_through(tmp_path):
    """prefetch() must fall through on cache miss, not return cached unrelated content."""
    provider = _provider_with_mock_config(tmp_path)
    _stage_prefetch_result(
        provider,
        ["unrelated topic"],
        [{"id": "unrelated", "content": "cached unrelated"}],
    )
    # No retrieve_recursive_bfs mocked, so falls to keyword search → returns ""
    result = provider.prefetch("completely different subject xyzw")
    assert result != "cached: unrelated"
    assert provider._warm_cache == {}


# ── staging slot handoff ─────────────────────────────────────────────────


def test_prefetch_swaps_pending_into_warm_cache(tmp_path):
    """Pending context retains its cue and uses normal relevance matching."""
    provider = _provider_with_mock_config(tmp_path)
    provider._prefetch_generation = 1
    _stage_prefetch_result(
        provider,
        ["database migration"],
        [{"id": "migration", "content": "staged context"}],
    )
    result = provider.prefetch("database migration plan")
    assert "staged context" in result
    assert provider._prefetch_pending is None


def test_prefetch_does_not_relabel_unrelated_pending_context(tmp_path):
    provider = _provider_with_mock_config(tmp_path)
    provider._prefetch_generation = 1
    _stage_prefetch_result(
        provider,
        ["pizza preferences"],
        [{"id": "pizza", "content": "cached pizza context"}],
    )

    result = provider.prefetch("production database migration")

    assert result != "cached pizza context"


def test_stale_prefetch_worker_cannot_replace_newer_generation(tmp_path):
    provider = _provider_with_mock_config(tmp_path)
    provider._prefetch_generation = 2

    old_identity = provider._prefetch_request_identity(
        session_id="test-session",
        generation=1,
        domain=None,
        tag=None,
        exclude_tags=None,
    )
    provider._stage_prefetch_result(old_identity, ["old"], [{"content": "old"}])
    assert provider._prefetch_pending is None

    new_identity = provider._prefetch_request_identity(
        session_id="test-session",
        generation=2,
        domain=None,
        tag=None,
        exclude_tags=None,
    )
    provider._stage_prefetch_result(new_identity, ["new"], [{"content": "new"}])
    assert provider._prefetch_pending is not None
    assert provider._prefetch_pending.identity == new_identity


def test_prefetch_filtered_request_cold_falls_through_unfiltered_warm_result(
    tmp_path, monkeypatch
):
    """Domain and tag selectors cannot reuse an unfiltered warm result."""
    provider = _provider_with_mock_config(tmp_path)
    warm_nodes = [
        {"id": "personal", "content": "personal identity"},
        {"id": "private", "content": "private identity"},
    ]

    def cold_nodes(query, max_nodes, domain, tag, exclude_tags):
        assert query == "shared project memory"
        if domain == "work":
            return [{"id": "work", "content": "work identity"}]
        if tag == "approved":
            return [{"id": "approved", "content": "approved identity"}]
        assert exclude_tags == ["private"]
        return [{"id": "public", "content": "public identity"}]

    monkeypatch.setattr(provider, "_keyword_search", cold_nodes)
    for kwargs, expected in (
        ({"domain": "work"}, "work identity"),
        ({"tag": "approved"}, "approved identity"),
        ({"exclude_tags": ["private"]}, "public identity"),
    ):
        _stage_prefetch_result(provider, ["shared project memory"], warm_nodes)
        result = provider.prefetch("shared project memory", **kwargs)
        assert expected in result
        assert "personal identity" not in result
        assert "private identity" not in result


def test_prefetch_empty_or_whitespace_query_never_matches_cached_cue(
    tmp_path, monkeypatch
):
    provider = _provider_with_mock_config(tmp_path)
    monkeypatch.setattr(
        provider,
        "_keyword_search",
        lambda *args: [{"id": "cold", "content": "cold empty result"}],
    )
    for query in ("", "   "):
        _stage_prefetch_result(
            provider,
            ["specific cached cue"],
            [{"id": "cached", "content": "arbitrary cached context"}],
        )
        result = provider.prefetch(query)
        assert "cold empty result" in result
        assert "arbitrary cached context" not in result


def test_prefetch_warm_result_is_copied_and_limited_at_format_time(tmp_path):
    provider = _provider_with_mock_config(tmp_path)
    provider._config = CashewConfig(recall_k=1)
    nodes = [
        {"id": "first", "content": "first cached identity"},
        {"id": "second", "content": "second cached identity"},
    ]
    _stage_prefetch_result(provider, ["limit proof"], nodes)
    nodes[0]["content"] = "mutated caller value"

    result = provider.prefetch("limit proof")

    assert "first cached identity" in result
    assert "mutated caller value" not in result
    assert "second cached identity" not in result


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("db_path", "other-profile.db"),
        ("cashew_db_path", "other/brain.db"),
        ("embedding_model", "example/alternate-embedding-model"),
        ("embedding_device", "mps"),
        ("recall_k", 1),
        ("prefetch_k", 4),
        ("prefetch_cues", 0),
        ("user_domain", "other-user"),
        ("ai_domain", "other-ai"),
        ("_features", {"experimental_parallel_retrieval": True}),
    ],
)
def test_late_prefetch_rejects_each_same_session_identity_change(
    tmp_path, change, value
):
    provider = _provider_with_mock_config(tmp_path)
    provider._config = CashewConfig()
    old_identity = provider._prefetch_request_identity(
        session_id="test-session",
        generation=0,
        domain=None,
        tag=None,
        exclude_tags=None,
    )

    if change == "db_path":
        provider._db_path = tmp_path / value
    else:
        provider._config = dataclasses.replace(provider._config, **{change: value})
    provider._stage_prefetch_result(
        old_identity, ["late"], [{"id": "old", "content": "old identity"}]
    )

    assert provider._prefetch_pending is None


def test_queue_prefetch_roundtrip_hits_warm_cache_without_second_retrieval(
    tmp_path, monkeypatch
):
    """The public queue hook stages a reusable unfiltered result asynchronously."""
    provider = _provider_with_mock_config(tmp_path)
    calls: list[dict] = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(node_id="warm")]

    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", retrieve)
    monkeypatch.setattr(
        provider,
        "_enrich_results",
        lambda node_ids, **kwargs: [
            {"id": node_ids[0], "content": "queued warm identity"}
        ],
    )

    provider.queue_prefetch("queued cache proof")
    deadline = time.monotonic() + 1.0
    while provider._prefetch_pending is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert provider._prefetch_pending is not None
    assert "queued warm identity" in provider.prefetch("queued cache proof")
    assert len(calls) == 1


def test_queue_prefetch_filtered_request_cold_falls_through_staged_result(
    tmp_path, monkeypatch
):
    """A queued unfiltered result cannot satisfy a later domain-constrained recall."""
    provider = _provider_with_mock_config(tmp_path)
    calls: list[dict] = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        node_id = "work" if kwargs.get("domain") == "work" else "personal"
        return [SimpleNamespace(node_id=node_id)]

    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", retrieve)
    monkeypatch.setattr(
        provider,
        "_enrich_results",
        lambda node_ids, **kwargs: [
            {"id": node_ids[0], "content": f"{node_ids[0]} identity"}
        ],
    )

    provider.queue_prefetch("queued selector proof")
    deadline = time.monotonic() + 1.0
    while provider._prefetch_pending is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert provider._prefetch_pending is not None
    result = provider.prefetch("queued selector proof", domain="work")
    assert "work identity" in result
    assert "personal identity" not in result
    assert len(calls) == 2
    assert calls[1]["domain"] == "work"


def test_prefetch_half_state_skips_warm_cache():
    """prefetch() must return '' when _config is None, ignoring warm cache."""
    provider = _half_state_provider()
    provider._warm_cache["hello"] = "cached"
    result = provider.prefetch("hello")
    assert result == ""


# ── background thread mechanics ──────────────────────────────────────────


def test_queue_prefetch_dispatches_tracked_background_thread(tmp_path, monkeypatch):
    """queue_prefetch tracks one persistent daemon until shutdown."""
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
    threads = tuple(provider._prefetch_threads)
    assert len(threads) == 1
    assert threads[0].daemon

    release.set()
    deadline = time.monotonic() + 1.0
    while (
        provider._prefetch_active_identity is not None and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert threads[0].is_alive()
    provider.shutdown()
    assert provider._prefetch_threads == set()


def test_queue_prefetch_burst_has_one_active_and_one_latest_pending(
    tmp_path, monkeypatch
):
    """A burst coalesces behind one active worker and one latest request."""
    provider = _provider_with_mock_config(tmp_path)
    provider._sync_queue = queue.Queue()
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def blocked_retrieval(**kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(kwargs["query"])
        started.set()
        assert release.wait(timeout=2.0)
        with state_lock:
            active -= 1
        return []

    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs", blocked_retrieval, raising=False
    )
    before = _METRICS._snapshot()["prefetch_coalesced"]
    provider.queue_prefetch("active request")
    assert started.wait(timeout=1.0)
    for index in range(100):
        provider.queue_prefetch(f"queued request {index}")

    with provider._sync_state_lock:
        assert provider._prefetch_pending_request is not None
        assert provider._prefetch_pending_request.query == "queued request 99"
        assert len(provider._prefetch_threads) == 1
    release.set()
    worker = next(iter(provider._prefetch_threads))
    deadline = time.monotonic() + 2.0
    while (
        provider._prefetch_active_identity is not None and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    assert worker.is_alive()
    assert max_active == 1
    assert calls == ["active request", "queued request 99"]
    assert _METRICS._snapshot()["prefetch_coalesced"] - before >= 99
    provider.shutdown()


def test_superseded_active_prefetch_skips_retrieval_after_cue_extraction(
    tmp_path, monkeypatch
):
    """An active request invalidated during cue extraction does no retrieval."""
    provider = _provider_with_mock_config(tmp_path)
    provider._sync_queue = queue.Queue()
    provider._config.prefetch_cues = 1
    cue_started = threading.Event()
    release = threading.Event()
    retrieval_queries: list[str] = []
    model_calls = 0
    cancelled_before = _METRICS._snapshot()["prefetch_cancelled"]

    def blocked_model(_prompt: str) -> str:
        nonlocal model_calls
        model_calls += 1
        cue_started.set()
        assert release.wait(timeout=2.0)
        return "obsolete cue" if model_calls == 1 else "current request"

    provider._model_fn = blocked_model
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda **kwargs: retrieval_queries.append(kwargs["query"]) or [],
        raising=False,
    )
    provider.queue_prefetch("obsolete request")
    assert cue_started.wait(timeout=1.0)
    provider.queue_prefetch("current request")
    release.set()

    deadline = time.monotonic() + 2.0
    while (
        provider._prefetch_active_identity is not None and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    assert retrieval_queries == ["current request"]
    assert _METRICS._snapshot()["prefetch_cancelled"] > cancelled_before
    provider.shutdown()


def test_active_prefetch_drops_result_after_runtime_identity_change(
    tmp_path, monkeypatch
):
    """A config identity change prevents late active publication."""
    provider = _provider_with_mock_config(tmp_path)
    provider._config = CashewConfig()
    provider._sync_queue = queue.Queue()
    started = threading.Event()
    release = threading.Event()
    cancelled_before = _METRICS._snapshot()["prefetch_cancelled"]

    def blocked_retrieval(**kwargs):
        started.set()
        assert release.wait(timeout=2.0)
        return [SimpleNamespace(node_id="late")]

    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs", blocked_retrieval, raising=False
    )
    monkeypatch.setattr(
        provider,
        "_enrich_results",
        lambda node_ids, **kwargs: [{"id": node_ids[0], "content": "late"}],
    )
    provider.queue_prefetch("identity request")
    assert started.wait(timeout=1.0)
    with provider._sync_state_lock:
        assert provider._config is not None
        provider._config = dataclasses.replace(provider._config, recall_k=1)
    release.set()

    deadline = time.monotonic() + 2.0
    while (
        provider._prefetch_active_identity is not None and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    assert provider._prefetch_pending is None
    assert _METRICS._snapshot()["prefetch_cancelled"] > cancelled_before
    provider.shutdown()


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
