# tests/test_memory_manager_e2e.py
# Phase 3 Plan 03-03 Task 4: TEST-01 + success criterion #4.
# Single E2E lifecycle test through the stubbed MemoryManager.
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from agent.memory_manager import MemoryManager  # via tests/_memory_manager_stub injection

from plugins.memory.cashew import CashewMemoryProvider


def test_memory_manager_e2e_lifecycle(tmp_path):
    """TEST-01: full MemoryManager lifecycle — add_provider -> initialize_all -> handle_tool_call
    -> on_session_end -> shutdown_all. Success criterion #4: < 5s wall-clock.

    Uses mocked ContextRetriever (no real DB, no real embedder) — satisfies
    success criterion #5 (no network / embedding download traffic)."""
    t0 = time.monotonic()

    provider = CashewMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)

    # Seed a config so initialize_all has something to load.
    provider.save_config({"recall_k": 3}, str(tmp_path))

    mgr.initialize_all(session_id="t-1", platform="cli", hermes_home=str(tmp_path))

    # After initialize_all, provider._retriever exists (Plan 03-01 eager construction).
    # Swap in a MagicMock so the tool call doesn't touch a real SQLite DB.
    assert provider._retriever is not None
    mock_nodes = [MagicMock(), MagicMock(), MagicMock()]
    provider._retriever = MagicMock()
    provider._retriever.retrieve.return_value = mock_nodes
    provider._retriever.format_context.return_value = "seeded context about X"

    result = mgr.handle_tool_call("cashew_query", {"query": "test"})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["ok"] is True, f"expected success envelope; got {parsed}"
    assert parsed["tool"] == "cashew_query"
    assert parsed["query"] == "test"
    assert parsed["context"] == "seeded context about X"
    assert parsed["node_count"] == 3  # len(mock_nodes)

    # recall_k=3 from saved config should have been passed to retrieve as max_nodes.
    provider._retriever.retrieve.assert_called_once_with("test", max_nodes=3)

    # on_session_end + shutdown_all must be safe (Phase 3: no-ops on the queue — Phase 4 owns flush).
    mgr.on_session_end([])
    mgr.shutdown_all()

    # Provider cleared post-shutdown (Plan 03-01 + 02-02 invariants).
    assert provider._retriever is None
    assert provider._config is None

    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"E2E lifecycle exceeded 5s budget: {elapsed:.2f}s"


def test_memory_manager_routes_unknown_tool_to_none(tmp_path):
    """Stub contract sanity: unknown tool -> MemoryManager.handle_tool_call returns None
    (no provider claims it). Verifies the stub's dispatch logic, not the provider's.

    This is a test-infra test; its failure would indicate the stub is broken, not the plugin."""
    provider = CashewMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)
    provider.save_config({}, str(tmp_path))
    mgr.initialize_all(session_id="t-2", hermes_home=str(tmp_path))
    try:
        result = mgr.handle_tool_call("not_a_real_tool", {})
        assert result is None, (
            f"stub must return None when no provider claims the tool; got {result!r}"
        )
    finally:
        mgr.shutdown_all()


def test_full_lifecycle_with_sync_path(tmp_path, monkeypatch):
    """TEST-01 extended: MemoryManager.sync_all exercises the Phase 4 write path.

    add_provider -> initialize_all -> handle_tool_call (Phase 3 cashew_query) ->
    sync_all x3 (Phase 4 write path) -> handle_tool_call (Phase 4 cashew_extract)
    -> on_session_end -> shutdown_all.

    Mocks both:
      - ContextRetriever instance (Phase 3 pattern) for cashew_query.
      - core.session.end_session (Phase 4 pattern) for the worker drain
        AND the cashew_extract synchronous path.
    """
    import threading
    import time
    import types
    from unittest.mock import MagicMock

    calls: list = []
    def _fake(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(new_nodes=["n1"], new_edges=[], updated_nodes=[])
    monkeypatch.setattr("core.session.end_session", _fake, raising=False)

    baseline = threading.active_count()
    mgr = MemoryManager()
    provider = CashewMemoryProvider()
    provider.save_config({}, str(tmp_path))
    mgr.add_provider(provider)

    start = time.monotonic()
    mgr.initialize_all("session-e2e-04", hermes_home=str(tmp_path))

    # Swap in a mocked retriever so Phase 3 tools.py happy path works.
    provider._retriever = MagicMock()
    provider._retriever.retrieve.return_value = [MagicMock()]
    provider._retriever.format_context.return_value = "recalled context"

    # Phase 3 tool — recall path
    q_result = mgr.handle_tool_call("cashew_query", {"query": "what did we decide"})
    assert isinstance(q_result, str)
    assert json.loads(q_result)["ok"] is True

    # Phase 4 write path — sync_all
    for i in range(3):
        mgr.sync_all(f"user-{i}", f"assistant-{i}")

    # Phase 4 tool — synchronous extract path
    x_result = mgr.handle_tool_call("cashew_extract", {
        "user_content": "important turn",
        "assistant_content": "noted",
    })
    assert isinstance(x_result, str)
    assert json.loads(x_result) == {
        "ok": True, "tool": "cashew_extract", "new_nodes": 1, "new_edges": 0,
    }

    # Session end bounded-drains the sync_all queue
    mgr.on_session_end([])
    mgr.shutdown_all()
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"E2E over budget: {elapsed*1000:.0f}ms (cap 5s per TEST-01)"

    # Thread-leak guard (Phase 4 contract)
    deadline = time.monotonic() + 2.0
    while threading.active_count() > baseline and time.monotonic() < deadline:
        time.sleep(0.01)
    assert threading.active_count() == baseline, (
        f"post-E2E thread leak: {threading.active_count()} vs {baseline}"
    )

    # Sanity: end_session was called at least once via the sync_all path
    # (may be more via cashew_extract). Drop-oldest may have eliminated some
    # of the 3 sync_all turns, but extract is synchronous so count >= 1.
    assert len(calls) >= 1
