"""Integration tests: full provider lifecycle against a real SQLite DB.

These tests exercise the complete CashewMemoryProvider lifecycle —
initialize, config round-trip, tool schema registration, query,
sync, session end, shutdown — against a real (but temporary) SQLite
database. No embedding model download is triggered (HF_HUB_OFFLINE=1
is set in conftest.py before any Cashew import).

Each test uses tmp_path for isolation (no ~/.hermes writes) and seeds
the database with known nodes so retrieval has something to find.
"""

from __future__ import annotations

import json
import sqlite3
import time

from agent.memory_manager import MemoryManager

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import (
    DEFAULTS,
    is_feature_enabled,
    load_config,
    resolve_db_path,
)


def _seed_node(db_path, id_, content, **kwargs):
    """Insert a single thought node into the SQLite database."""
    conn = sqlite3.connect(str(db_path))
    defaults = {
        "id": id_,
        "content": content,
        "node_type": "thought",
        "domain": None,
        "timestamp": "2026-06-18T00:00:00",
        "access_count": 0,
        "last_accessed": None,
        "source_file": None,
        "decayed": 0,
        "metadata": "{}",
        "last_updated": None,
        "mood_state": None,
        "permanent": 0,
        "tags": None,
        "referent_time": None,
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


def test_full_lifecycle_initialize_to_shutdown(tmp_path):
    """Full provider lifecycle: init -> tool schemas -> shutdown.

    Verifies the provider can be initialized, registers tool schemas,
    and shuts down cleanly without errors.
    """
    provider = CashewMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)

    provider.save_config({"recall_k": 3}, str(tmp_path))
    mgr.initialize_all(
        session_id="lifecycle-1", platform="cli", hermes_home=str(tmp_path)
    )

    schemas = mgr.get_tool_schemas_all()
    assert len(schemas) == 1
    schema = schemas[0]
    assert "function" in schema
    assert schema["function"]["name"] in ("cashew_query",)

    mgr.shutdown_all()


def test_full_lifecycle_query_and_sync(tmp_path):
    """Query + sync turn against a seeded database.

    Seeds a known node, queries it, syncs a turn, and verifies both
    operations succeed without embedding model access.
    """
    provider = CashewMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)

    provider.save_config({"recall_k": 3}, str(tmp_path))
    mgr.initialize_all(
        session_id="lifecycle-2", platform="cli", hermes_home=str(tmp_path)
    )

    db_path = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db_path, "n1", "the capital of France is Paris", node_type="fact")
    _seed_node(
        db_path, "n2", "Python was created by Guido van Rossum", node_type="fact"
    )

    result = mgr.handle_tool_call("cashew_query", {"query": "capital of France"})
    parsed = json.loads(result)
    assert parsed["ok"] is True, f"Expected success envelope, got: {parsed}"
    assert parsed["node_count"] >= 1

    mgr.sync_all("what is the capital?", "The capital of France is Paris.")
    mgr.on_session_end([])
    mgr.shutdown_all()


def test_full_lifecycle_feature_flags(tmp_path):
    """Feature flag infrastructure: toggles are configurable and queryable.

    Writes _features to cashew.json, loads config, and verifies
    is_feature_enabled() returns the correct values.
    """
    provider = CashewMemoryProvider()
    provider.save_config(
        {
            "recall_k": 3,
            "_features": {
                "experimental_batch_sync": True,
                "experimental_parallel_retrieval": False,
            },
        },
        str(tmp_path),
    )

    config = load_config(str(tmp_path))
    assert is_feature_enabled(config, "experimental_batch_sync") is True
    assert is_feature_enabled(config, "experimental_parallel_retrieval") is False
    assert is_feature_enabled(config, "nonexistent_flag") is False


def test_full_lifecycle_config_roundtrip(tmp_path):
    """Config save + load round-trip preserves all keys."""
    provider = CashewMemoryProvider()
    custom = {
        "recall_k": 7,
        "token_budget": 1000,
        "sleep_schedule": "0 3 * * *",
        "_features": {
            "experimental_batch_sync": True,
            "experimental_parallel_retrieval": True,
        },
    }
    provider.save_config(custom, str(tmp_path))

    config = load_config(str(tmp_path))
    assert config.recall_k == 7
    assert config.token_budget == 1000
    assert config.sleep_schedule == "0 3 * * *"
    assert config._features == custom["_features"]

    # Non-overridden keys fall back to defaults
    assert config.walk_depth == DEFAULTS["walk_depth"]
    assert config.gc_mode == DEFAULTS["gc_mode"]


def test_full_lifecycle_timing(tmp_path):
    """Full lifecycle completes in under 5 seconds wall-clock."""
    t0 = time.monotonic()

    provider = CashewMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)

    provider.save_config({"recall_k": 3}, str(tmp_path))
    mgr.initialize_all(session_id="timing-1", platform="cli", hermes_home=str(tmp_path))

    db_path = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    for i in range(5):
        _seed_node(
            db_path, f"n{i}", f"node {i} content about topic {i}", node_type="thought"
        )

    mgr.handle_tool_call("cashew_query", {"query": "topic"})
    mgr.sync_all("user message", "assistant message")
    mgr.on_session_end([])
    mgr.shutdown_all()

    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"Full lifecycle took {elapsed:.1f}s, expected < 5s"
