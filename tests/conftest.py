# tests/conftest.py
# Offline env vars per https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
# fake_embedder autouse fixture patterned on RESEARCH.md Example 1
# agent.memory_provider sys.modules stub is test-infra-specific: hermes-agent is NOT a pip dep,
# so we synthesize a minimal MemoryProvider ABC to satisfy `from agent.memory_provider import MemoryProvider`.
import os
import sys
import types
from abc import ABC, abstractmethod

# CRITICAL: set BEFORE any Cashew-reachable import runs. Process-wide, persists for full session.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Synthesize agent.memory_provider if hermes-agent is not installed.
# Verified minimal contract: MemoryProvider has `name` property and a set of abstract methods.
# We only need the ABC to be importable — actual method signatures are exercised by Hermes at runtime.
if "agent.memory_provider" not in sys.modules:
    _agent_pkg = types.ModuleType("agent")
    _agent_pkg.__path__ = []  # mark as package
    _mp_mod = types.ModuleType("agent.memory_provider")

    class MemoryProvider(ABC):
        """Minimal stub of the Hermes MemoryProvider ABC for test isolation."""

        @property
        @abstractmethod
        def name(self) -> str: ...

        def is_available(self) -> bool:
            return False

        def initialize(self, session_id: str, **kwargs) -> None: ...
        def get_config_schema(self) -> dict:
            return {}

        def save_config(self, values: dict, hermes_home: str) -> None: ...
        def get_tool_schemas(self) -> list:
            return []

        def handle_tool_call(self, name: str, args: dict):
            return None

        def prefetch(self, query: str) -> str:
            return ""

        def sync_turn(self, user_content: str, assistant_content: str) -> None: ...
        def on_session_end(self, messages: list) -> None: ...
        def shutdown(self) -> None: ...

    _mp_mod.MemoryProvider = MemoryProvider
    sys.modules["agent"] = _agent_pkg
    sys.modules["agent.memory_provider"] = _mp_mod

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """Block any attempt to load a real sentence-transformers model.

    Phase 1 tests don't touch Cashew, so this fixture is defensive scaffolding that Phase 3+ inherits.
    If Cashew's core.embeddings module is importable in this environment, patch load_embeddings to a
    RuntimeError-raising stub; if not (Phase 1 typical), the fixture is a no-op.
    """
    def _stub(*args, **kwargs):
        raise RuntimeError(
            "Real embedding load blocked in tests. Use a fixture-provided stub retriever."
        )
    try:
        import core.embeddings  # noqa: F401
        monkeypatch.setattr("core.embeddings.load_embeddings", _stub)
    except ImportError:
        pass
    yield
