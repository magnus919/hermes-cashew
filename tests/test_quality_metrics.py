# tests/test_quality_metrics.py
# Phase 9 Plan 03 Task 2: Quality and metrics tests (scoring, access metrics, permanent boost).
from __future__ import annotations

import sqlite3

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS, resolve_db_path


@pytest.fixture
def provider(tmp_path):
    p = CashewMemoryProvider()
    p.initialize(session_id="test", hermes_home=str(tmp_path))
    yield p
    p.shutdown()


@pytest.fixture
def db_path(tmp_path):
    return resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])


def _seed_node(db_path, **kwargs):
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


def test_access_count_incremented_after_retrieval(provider, db_path):
    _seed_node(db_path, id="n1", content="test", access_count=0)
    provider.prefetch("test")
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT access_count FROM thought_nodes WHERE id = ?", ("n1",)).fetchone()
    conn.close()
    assert row[0] == 1


def test_last_accessed_updated_after_retrieval(provider, db_path):
    _seed_node(db_path, id="n1", content="test", last_accessed=None)
    provider.prefetch("test")
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT last_accessed FROM thought_nodes WHERE id = ?", ("n1",)).fetchone()
    conn.close()
    assert row[0] is not None


def test_permanent_nodes_ranked_higher(provider, db_path):
    _seed_node(db_path, id="perm", content="permanent node", permanent=1, timestamp="2026-01-01")
    _seed_node(db_path, id="norm", content="normal node", permanent=0, timestamp="2026-01-01")
    result = provider.prefetch("node")
    perm_idx = result.find("permanent node")
    norm_idx = result.find("normal node")
    assert perm_idx < norm_idx and perm_idx != -1


def test_domain_filtering(provider, db_path):
    _seed_node(db_path, id="w1", content="work node", domain="work")
    _seed_node(db_path, id="p1", content="personal node", domain="personal")
    result = provider.prefetch("node", domain="work")
    assert "work node" in result
    assert "personal node" not in result


def test_tag_filtering(provider, db_path):
    _seed_node(db_path, id="i1", content="important node", tags="important,urgent")
    _seed_node(db_path, id="t1", content="trivial node", tags="trivial")
    result = provider.prefetch("node", tag="important")
    assert "important node" in result
    assert "trivial node" not in result


def test_recency_weighting_uses_referent_time(provider, db_path):
    _seed_node(db_path, id="old", content="old news", referent_time="2024-01-01", timestamp="2026-01-01")
    _seed_node(db_path, id="new", content="new news", referent_time="2026-01-01", timestamp="2024-01-01")
    result = provider.prefetch("news")
    new_idx = result.find("new news")
    old_idx = result.find("old news")
    assert new_idx < old_idx and new_idx != -1


def test_recency_fallback_to_timestamp(provider, db_path):
    _seed_node(db_path, id="old", content="old item", referent_time=None, timestamp="2024-01-01")
    _seed_node(db_path, id="new", content="new item", referent_time=None, timestamp="2026-01-01")
    result = provider.prefetch("item")
    new_idx = result.find("new item")
    old_idx = result.find("old item")
    assert new_idx < old_idx and new_idx != -1


def test_empty_filter_returns_all_nodes(provider, db_path):
    _seed_node(db_path, id="a", content="shared alpha", domain="x", tags="y")
    _seed_node(db_path, id="b", content="shared beta", domain="y", tags="z")
    _seed_node(db_path, id="c", content="shared gamma", domain="z", tags="x")
    result = provider.prefetch("shared")
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result


def test_hybrid_scoring_includes_confidence(provider, db_path):
    _seed_node(db_path, id="high_conf", content="high confidence test", confidence=0.9)
    _seed_node(db_path, id="low_conf", content="low confidence test", confidence=0.1)
    result = provider.prefetch("test")
    high_idx = result.find("high confidence test")
    low_idx = result.find("low confidence test")
    assert high_idx < low_idx and high_idx != -1


def test_permanent_flag_visible_in_context(provider, db_path):
    _seed_node(db_path, id="p1", content="important decision", permanent=1)
    result = provider.prefetch("important")
    assert "permanent" in result
