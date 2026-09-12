"""Legacy opt-in profiles use the ordinary retrieval path without executors."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
from types import SimpleNamespace

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS, load_config, save_config


@pytest.fixture(params=[True, False])
def legacy_provider(tmp_path, request, monkeypatch):
    profile = tmp_path / "cashew.json"
    profile.write_text(
        json.dumps(
            {
                "llm_aux_role": None,
                "sleep_cycles": False,
                "_features": {"experimental_parallel_retrieval": request.param},
            }
        )
    )
    provider = CashewMemoryProvider()
    provider.initialize("legacy", hermes_home=str(tmp_path))
    assert provider.is_available()

    def forbidden_executor(*args, **kwargs):
        raise AssertionError("retired retrieval must never create an executor")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", forbidden_executor)
    try:
        yield provider
    finally:
        provider.shutdown()
        assert provider._prefetch_threads == set()


def _seed(provider, rows):
    with sqlite3.connect(provider._db_path) as conn:
        conn.executemany(
            "INSERT INTO thought_nodes "
            "(id, content, node_type, domain, timestamp, tags, decayed, access_count) "
            "VALUES (?, ?, 'fact', ?, '2026-09-12T00:00:00', ?, 0, 0)",
            rows,
        )


def _counts(provider):
    with sqlite3.connect(provider._db_path) as conn:
        return dict(conn.execute("SELECT id, access_count FROM thought_nodes"))


def test_legacy_flag_uses_upstream_first_without_executor(legacy_provider, monkeypatch):
    provider = legacy_provider
    _seed(provider, [("semantic", "selected semantic result", "user", "keep")])
    calls = []

    def upstream(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(node_id="semantic")]

    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("nonempty upstream result must win")

    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", upstream)
    monkeypatch.setattr(provider, "_keyword_search", forbidden_fallback)
    for _ in range(3):
        assert provider.prefetch(
            "query", domain="user", tag="keep", exclude_tags=["private"]
        ) == (
            "=== RELEVANT CONTEXT ===\n[domain: user | type: fact] selected semantic result"
        )
    assert len(calls) == 3
    assert all(
        call
        == {
            "db_path": str(provider._db_path),
            "query": "query",
            "top_k": DEFAULTS["recall_k"],
            "domain": "user",
            "tags": ["keep"],
            "exclude_tags": ["private"],
        }
        for call in calls
    )
    assert _counts(provider) == {"semantic": 3}


@pytest.mark.parametrize("upstream_fails", [True, False])
def test_legacy_flag_preserves_filtered_keyword_fallback(
    legacy_provider, monkeypatch, upstream_fails
):
    provider = legacy_provider
    _seed(
        provider,
        [
            ("selected", "shared selected", "user", "keep"),
            ("private", "shared private", "user", "keep,private"),
            ("other-domain", "shared other domain", "other", "keep"),
            ("other-tag", "shared other tag", "user", "other"),
        ],
    )

    def upstream(**kwargs):
        if upstream_fails:
            raise RuntimeError("synthetic backend failure")
        return []

    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", upstream)
    result = provider.prefetch(
        "shared", domain="user", tag="keep", exclude_tags=["private"]
    )
    assert "shared selected" in result
    assert "shared private" not in result
    assert "shared other" not in result
    assert _counts(provider) == {
        "selected": 1,
        "private": 0,
        "other-domain": 0,
        "other-tag": 0,
    }


def test_legacy_flag_total_failure_is_neutral(legacy_provider, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", fail)
    monkeypatch.setattr(legacy_provider, "_keyword_search", fail)
    assert legacy_provider.prefetch("unavailable") == ""


@pytest.mark.parametrize("enabled", [True, False])
def test_load_and_save_prune_only_retired_flag(tmp_path, enabled):
    profile = tmp_path / "cashew.json"
    flags = {
        "experimental_parallel_retrieval": enabled,
        "experimental_batch_sync": True,
        "future_flag": True,
    }
    profile.write_text(json.dumps({"_features": flags}))
    expected = {"experimental_batch_sync": True, "future_flag": True}
    assert load_config(tmp_path)._features == expected
    # Reading is compatible without rewriting an existing profile.
    assert json.loads(profile.read_text())["_features"] == flags
    save_config({"recall_k": 7}, tmp_path)
    assert json.loads(profile.read_text())["_features"] == expected
    assert load_config(tmp_path).recall_k == 7
    assert "experimental_parallel_retrieval" not in DEFAULTS["_features"]
