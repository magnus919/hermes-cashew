# tests/test_handle_tool_call.py
# Phase 3 + Phase 9: handle_tool_call tests — RECALL-03 + RECALL-04 + stack-trace-leak guard.
#
# Phase 9 update: tests now seed a real SQLite DB instead of mocking ContextRetriever.
from __future__ import annotations

import json
import logging
import sqlite3

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS, resolve_db_path


def _make_initialized_provider(tmp_path, recall_k: int | None = None) -> CashewMemoryProvider:
    """Mirror helper from test_recall.py: initialize with real DB."""
    p = CashewMemoryProvider()
    overrides = {"recall_k": recall_k} if recall_k is not None else {}
    p.save_config(overrides, str(tmp_path))
    p.initialize("s", hermes_home=str(tmp_path))
    return p


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


def test_happy_path_returns_valid_success_envelope(tmp_path):
    """RECALL-03: JSON string success envelope with full shape."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="hello world")
    _seed_node(db, id="n2", content="goodbye world")
    try:
        result = p.handle_tool_call("cashew_query", {"query": "hello"})
        assert isinstance(result, str)
        d = json.loads(result)
        assert d["ok"] is True
        assert d["tool"] == "cashew_query"
        assert d["query"] == "hello"
        assert "hello world" in d["context"]
        assert d["node_count"] >= 1
    finally:
        p.shutdown()


def test_happy_path_uses_recall_k_when_max_nodes_omitted(tmp_path):
    """When args has no max_nodes, self._config.recall_k caps results."""
    p = _make_initialized_provider(tmp_path, recall_k=2)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    for i in range(5):
        _seed_node(db, id=f"n{i}", content=f"shared keyword {i}")
    try:
        result = p.handle_tool_call("cashew_query", {"query": "shared keyword"})
        d = json.loads(result)
        assert d["ok"] is True
        assert d["node_count"] <= 2
    finally:
        p.shutdown()


def test_max_nodes_override_wins_over_recall_k(tmp_path):
    """When args['max_nodes'] is present, it overrides recall_k per-call."""
    p = _make_initialized_provider(tmp_path, recall_k=9)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    for i in range(5):
        _seed_node(db, id=f"n{i}", content=f"shared keyword {i}")
    try:
        result = p.handle_tool_call("cashew_query", {"query": "shared keyword", "max_nodes": 3})
        d = json.loads(result)
        assert d["ok"] is True
        assert d["node_count"] <= 3
    finally:
        p.shutdown()


def test_unknown_tool_returns_error_envelope_and_logs_once(tmp_path, caplog):
    """RECALL-03: name != cashew_query -> error envelope + one WARNING (no exc_info)."""
    p = _make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call("bogus_tool", {"query": "x"})
        assert isinstance(result, str)
        d = json.loads(result)
        assert d == {
            "ok": False,
            "tool": "cashew_query",
            "error": "unknown tool",
            "query": None,
        }
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert warnings[0].exc_info is None
    finally:
        p.shutdown()


def test_half_state_returns_error_envelope_no_log(tmp_path, caplog):
    """RECALL-04: pre-initialize -> error envelope, NO log (init-time WARNING is the audit trail)."""
    p = CashewMemoryProvider()  # never initialized
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        result = p.handle_tool_call("cashew_query", {"query": "hello"})
    assert json.loads(result) == {
        "ok": False,
        "tool": "cashew_query",
        "error": "cashew recall failed",
        "query": "hello",
    }
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("cashew" in m for m in msgs)


def test_missing_query_returns_error_envelope_and_logs_once(tmp_path, caplog):
    """RECALL-04: args={} -> KeyError inside main try block -> error envelope + one WARNING."""
    p = _make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call("cashew_query", {})
        d = json.loads(result)
        assert d["ok"] is False
        assert d["query"] is None
        assert d["error"] == "cashew recall failed"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_retrieval_exception_returns_error_envelope_and_logs_once(tmp_path, caplog, monkeypatch):
    """RECALL-04: retrieval raises -> error envelope + one WARNING with exc_info."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="test")

    def _raise(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(p, "_retrieve_with_vec", _raise)
    monkeypatch.setattr(p, "_retrieve_keyword", _raise)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call("cashew_query", {"query": "x"})
        d = json.loads(result)
        assert d["ok"] is False
        assert d["error"] == "cashew recall failed"
        assert d["query"] == "x"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_error_envelope_has_no_stack_trace_substrings(tmp_path, caplog, monkeypatch):
    """Success criterion #3: no internal stack trace leaks through the JSON envelope."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="test")

    def _raise(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked /some/secret/path/brain.db")

    monkeypatch.setattr(p, "_retrieve_with_vec", _raise)
    monkeypatch.setattr(p, "_retrieve_keyword", _raise)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call("cashew_query", {"query": "x"})
        forbidden = ["Traceback", "File \"", "sqlite3.OperationalError", "at line", "/some/secret/path/"]
        for f in forbidden:
            assert f not in result, f"error envelope leaks {f!r}: {result!r}"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings and warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_return_type_is_always_str(tmp_path):
    """Contract: every code path returns str."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="test")
    try:
        assert isinstance(p.handle_tool_call("cashew_query", {"query": "x"}), str)
        assert isinstance(p.handle_tool_call("cashew_query", {}), str)
        assert isinstance(p.handle_tool_call("bogus", {}), str)
    finally:
        p.shutdown()


def test_node_count_matches_retrieved_length(tmp_path):
    """Success envelope's node_count matches the number of returned nodes."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    for i in range(5):
        _seed_node(db, id=f"n{i}", content=f"node {i}")
    try:
        d = json.loads(p.handle_tool_call("cashew_query", {"query": "node"}))
        assert d["ok"] is True
        assert d["node_count"] == 5
    finally:
        p.shutdown()
