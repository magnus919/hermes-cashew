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
