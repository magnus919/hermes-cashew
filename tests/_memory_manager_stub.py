# tests/_memory_manager_stub.py
# Test-infra: minimal MemoryManager stub. Filename prefixed with _ so pytest's
# default `test_*.py` discovery skips it.
#
# PHASE_DESIGN_NOTES Decision Point 4: keep in a dedicated file (not conftest.py
# which is already 134 lines + autouse fixtures). 03-RESEARCH.md §6 describes
# the expected surface; only add methods that existing Phase 3 tests need.
from __future__ import annotations

import sys
import types
from typing import Any


class MemoryManager:
    """Minimal stand-in for Hermes's agent.memory_manager.MemoryManager.

    Implements only the five methods Phase 3's E2E test exercises:
      add_provider, initialize_all, handle_tool_call, on_session_end, shutdown_all.

    Phase 4 will extend with sync_all when the write path ships.
    """

    def __init__(self) -> None:
        self._providers: list = []

    def add_provider(self, provider) -> None:
        self._providers.append(provider)

    def initialize_all(self, session_id: str, **kwargs: Any) -> None:
        for p in self._providers:
            p.initialize(session_id, **kwargs)

    def handle_tool_call(self, name: str, args: dict) -> Any:
        """Route tool call to the first provider that declares `name` in its schemas.

        Returns the provider's handle_tool_call return value (always a JSON string
        in Phase 3 per Plan 03-02 contract). Returns None if no provider claims
        the tool — this is a test-harness choice; real Hermes may raise or warn.
        """
        for p in self._providers:
            schemas = p.get_tool_schemas()
            if any(s.get("name") == name for s in schemas):
                return p.handle_tool_call(name, args)
        return None

    def on_session_end(self, messages: list) -> None:
        for p in self._providers:
            p.on_session_end(messages)

    def shutdown_all(self) -> None:
        for p in self._providers:
            p.shutdown()


def inject_into_sys_modules() -> None:
    """Populate sys.modules['agent.memory_manager'] so `from agent.memory_manager
    import MemoryManager` succeeds in test code.

    Idempotent: if already injected, becomes a no-op (per-test pollution guard).
    Must be called AFTER the agent.memory_provider stub is in place (conftest.py
    import order ensures this).
    """
    if "agent.memory_manager" in sys.modules:
        return
    mm_mod = types.ModuleType("agent.memory_manager")
    mm_mod.MemoryManager = MemoryManager
    sys.modules["agent.memory_manager"] = mm_mod
