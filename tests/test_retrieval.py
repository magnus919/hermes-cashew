# tests/test_retrieval.py
# Phase 9 Plan 03 Task 1: Retrieval path tests (vec, keyword, fallback, empty graph).
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS, resolve_db_path


@pytest.fixture
def provider(tmp_path):
    """Create an initialized provider with a fresh DB."""
    p = CashewMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    yield p
    p.shutdown()


@pytest.fixture
def db_path(tmp_path):
    """Return the resolved DB path for direct sqlite3 access."""
    return resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])


def _seed_node(db_path, **kwargs):
    """Insert a row into thought_nodes with sensible defaults."""
    conn = sqlite3.connect(str(db_path))
    defaults = {
        "id": "n1",
        "content": "test content",
        "node_type": "thought",
        "domain": None,
        "timestamp": "2026-01-01T00:00:00",
        "access_count": 0,
        "last_accessed": None,
        "confidence": 0.5,
        "source_file": None,
        "decayed": 0,
        "metadata": "{}",
        "last_updated": None,
        "mood_state": None,
        "permanent": 0,
        "tags": None,
        "referent_time": None,
        "reasoning": None,
    }
    defaults.update(kwargs)
    columns = ", ".join(defaults.keys())
    placeholders = ", ".join("?" * len(defaults))
    conn.execute(
        f"INSERT INTO thought_nodes ({columns}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()
    conn.close()


def test_prefetch_empty_graph_returns_empty_string(provider):
    result = provider.prefetch("test query")
    assert result == ""


def test_prefetch_keyword_fallback_when_vec_unavailable(provider, db_path):
    _seed_node(db_path, id="n1", content="hello world")
    _seed_node(db_path, id="n2", content="goodbye world")
    result = provider.prefetch("hello")
    assert "hello world" in result
    assert "goodbye world" not in result


def test_prefetch_empty_graph_no_error(provider):
    result = provider.prefetch("anything")
    assert result == ""
    assert isinstance(result, str)


def test_handle_tool_call_keyword_fallback(provider, db_path):
    _seed_node(db_path, id="n1", content="test content")
    result = provider.handle_tool_call("cashew_query", {"query": "test"})
    d = json.loads(result)
    assert d["ok"] is True
    assert "test content" in d["context"]
    assert d["node_count"] >= 1


def test_retrieve_keyword_respects_max_nodes(provider, db_path):
    for i in range(5):
        _seed_node(db_path, id=f"n{i}", content="shared keyword")
    nodes = provider._retrieve_keyword("shared keyword", 3)
    assert len(nodes) <= 3


def test_retrieve_keyword_orders_by_referent_time(provider, db_path):
    _seed_node(db_path, id="old", content="news", referent_time="2024-01-01")
    _seed_node(db_path, id="new", content="news", referent_time="2026-01-01")
    nodes = provider._retrieve_keyword("news", 5)
    assert len(nodes) == 2
    assert nodes[0]["id"] == "new"
    assert nodes[1]["id"] == "old"


def test_lazy_import_preserves_module_loadability():
    result = subprocess.run(
        [sys.executable, "-c", "import plugins.memory.cashew"],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr


def test_format_context_with_domain_and_type(provider, db_path):
    _seed_node(db_path, id="n1", content="node content", domain="test_domain", node_type="belief")
    result = provider.prefetch("node")
    assert "[domain: test_domain | type: belief]" in result


def test_format_context_without_domain_or_type(provider, db_path):
    # Schema requires node_type NOT NULL, so use empty string to simulate no-type
    _seed_node(db_path, id="n1", content="plain content", domain="", node_type="")
    result = provider.prefetch("plain")
    # Both domain and node_type are falsy — no brackets should appear
    assert result == "=== RELEVANT CONTEXT ===\nplain content"


def test_macos_fallback_simulated(provider, db_path, monkeypatch):
    """Simulate macOS where sqlite-vec extension loading is unavailable — provider falls back to keyword search."""
    import sqlite3

    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("enable_load_extension not available on this Python build")

    try:
        type(sqlite3.Connection).enable_load_extension
    except TypeError:
        monkeypatch.setattr(provider, '_retrieve_with_vec', lambda *a, **k: [])
        _seed_node(db_path, id="n1", content="test content")
        result = provider.prefetch("test")
        assert "test content" in result
        return

    def _blocked_enable_load(self, *args, **kwargs):
        raise AttributeError("simulated macOS: extension loading blocked")

    monkeypatch.setattr(sqlite3.Connection, "enable_load_extension", _blocked_enable_load)
    _seed_node(db_path, id="n1", content="test content")
    result = provider.prefetch("test")
    assert "test content" in result


# Need pathlib import for test_lazy_import_preserves_module_loadability
import pathlib  # noqa: E402
