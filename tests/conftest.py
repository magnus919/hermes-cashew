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

# Phase 3: MemoryManager stub injection — must happen after agent.memory_provider
# is already in sys.modules because the stub imports nothing from there directly
# but Hermes ABC ordering conventions keep memory_provider ahead of memory_manager.
from tests._memory_manager_stub import inject_into_sys_modules  # noqa: E402
inject_into_sys_modules()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """Block any attempt to load a real sentence-transformers model.

    Phase 1 tests don't touch Cashew, so this fixture is defensive scaffolding that Phase 3+ inherits.
    If Cashew's core.embeddings module is importable in this environment, patch every public
    model-loading/inference entry point to a RuntimeError-raising stub; if not (unusual in Phase 1,
    since cashew-brain IS a declared runtime dep), the fixture is a no-op.

    Candidate function names cover both the plan's historical spec (`load_embeddings`) and the
    actual Cashew API surface observed at pin 90d1c73 (`embed_text`, `embed_nodes`, `load_all_embeddings`).
    `raising=False` makes each patch a no-op if that attribute is absent on the current Cashew version —
    the fixture is belt-and-suspenders, not a pinned-version contract.
    """
    def _stub(*args, **kwargs):
        raise RuntimeError(
            "Real embedding load blocked in tests. Use a fixture-provided stub retriever."
        )
    try:
        import core.embeddings  # noqa: F401
        for attr in ("load_embeddings", "load_all_embeddings", "embed_text", "embed_nodes"):
            monkeypatch.setattr(f"core.embeddings.{attr}", _stub, raising=False)
    except ImportError:
        pass
    yield


@pytest.fixture
def home_snapshot():
    """Snapshot ~/.hermes mtime + recursive file listing before/after a test.

    Used by tests/test_no_home_leak.py to satisfy TEST-03: any plugin code path
    that writes under the user's real $HOME during a tmp_path-scoped lifecycle
    is a profile-isolation violation. Yields a dict with the pre-test snapshot;
    the test calls `assert_unchanged()` (also yielded) to verify post-test state.

    If ~/.hermes does not exist pre-test, asserts it does not exist post-test.
    Never creates files under ~ — purely observational.
    """
    import pathlib
    home_hermes = pathlib.Path.home() / ".hermes"

    def _snapshot():
        if not home_hermes.exists():
            return {"exists": False, "mtime": None, "files": frozenset()}
        files = frozenset(
            (p.relative_to(home_hermes).as_posix(), p.stat().st_mtime_ns)
            for p in home_hermes.rglob("*")
            if p.is_file()
        )
        return {
            "exists": True,
            "mtime": home_hermes.stat().st_mtime_ns,
            "files": files,
        }

    pre = _snapshot()

    def assert_unchanged():
        post = _snapshot()
        assert pre["exists"] == post["exists"], (
            f"~/.hermes existence changed: pre={pre['exists']} post={post['exists']}"
        )
        if pre["exists"]:
            assert pre["mtime"] == post["mtime"], (
                f"~/.hermes mtime changed: pre={pre['mtime']} post={post['mtime']}"
            )
            new_files = post["files"] - pre["files"]
            removed_files = pre["files"] - post["files"]
            assert not new_files, f"~/.hermes gained files during test: {sorted(new_files)}"
            assert not removed_files, f"~/.hermes lost files during test: {sorted(removed_files)}"

    yield {"pre": pre, "assert_unchanged": assert_unchanged}
    # Belt-and-suspenders: if the test forgot to call assert_unchanged, do it here.
    assert_unchanged()
