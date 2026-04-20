# tests/test_handle_tool_call.py
# Phase 3 Plan 03-03 Task 3: handle_tool_call tests — RECALL-03 + RECALL-04 + stack-trace-leak guard.
from __future__ import annotations

import json
import logging
import sqlite3
from unittest.mock import MagicMock

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS


def _make_initialized_provider(tmp_path, recall_k: int | None = None) -> CashewMemoryProvider:
    """Mirror helper from test_recall.py: initialize, then swap in a MagicMock retriever."""
    p = CashewMemoryProvider()
    overrides = {"recall_k": recall_k} if recall_k is not None else {}
    p.save_config(overrides, str(tmp_path))
    p.initialize("s", hermes_home=str(tmp_path))
    p._retriever = MagicMock()
    return p


def test_happy_path_returns_valid_success_envelope(tmp_path):
    """RECALL-03: JSON string success envelope with full shape."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.return_value = [MagicMock(), MagicMock()]  # 2 nodes
    p._retriever.format_context.return_value = "the context"
    try:
        result = p.handle_tool_call("cashew_query", {"query": "hello"})
        assert isinstance(result, str)
        d = json.loads(result)
        assert d == {
            "ok": True,
            "tool": "cashew_query",
            "query": "hello",
            "context": "the context",
            "node_count": 2,
        }
    finally:
        p.shutdown()


def test_happy_path_uses_recall_k_when_max_nodes_omitted(tmp_path):
    """When args has no max_nodes, self._config.recall_k wins."""
    p = _make_initialized_provider(tmp_path, recall_k=9)
    p._retriever.retrieve.return_value = []
    p._retriever.format_context.return_value = ""
    try:
        p.handle_tool_call("cashew_query", {"query": "x"})
        p._retriever.retrieve.assert_called_once_with("x", max_nodes=9)
    finally:
        p.shutdown()


def test_max_nodes_override_wins_over_recall_k(tmp_path):
    """When args['max_nodes'] is present, it overrides recall_k per-call."""
    p = _make_initialized_provider(tmp_path, recall_k=9)
    p._retriever.retrieve.return_value = []
    p._retriever.format_context.return_value = ""
    try:
        p.handle_tool_call("cashew_query", {"query": "x", "max_nodes": 3})
        p._retriever.retrieve.assert_called_once_with("x", max_nodes=3)
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
        # Unknown-tool path: exc_info must be None (no exception to record).
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
    # Pre-initialize: no relevant WARNINGs logged for this call.
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
        assert warnings[0].exc_info is not None  # exc_info should be attached per Plan 03-02
    finally:
        p.shutdown()


def test_retriever_exception_returns_error_envelope_and_logs_once(tmp_path, caplog):
    """RECALL-04: retrieve raises -> error envelope + one WARNING with exc_info."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.side_effect = sqlite3.OperationalError("database is locked")
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


def test_error_envelope_has_no_stack_trace_substrings(tmp_path, caplog):
    """Success criterion #3: no internal stack trace leaks through the JSON envelope."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.side_effect = sqlite3.OperationalError(
        "database is locked /some/secret/path/brain.db"
    )
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call("cashew_query", {"query": "x"})
        # The returned string is the JSON payload. It must NOT contain:
        forbidden = ["Traceback", "File \"", "sqlite3.OperationalError", "at line", "/some/secret/path/"]
        for f in forbidden:
            assert f not in result, (
                f"error envelope leaks {f!r}: {result!r}"
            )
        # AND the log record DOES contain exc_info (full traceback routed to logger, not JSON):
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings and warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_return_type_is_always_str(tmp_path):
    """Contract: every code path returns str. json.dumps guarantees it but guard against regression."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.return_value = []
    p._retriever.format_context.return_value = ""
    try:
        assert isinstance(p.handle_tool_call("cashew_query", {"query": "x"}), str)
        assert isinstance(p.handle_tool_call("cashew_query", {}), str)
        assert isinstance(p.handle_tool_call("bogus", {}), str)
    finally:
        p.shutdown()


def test_node_count_matches_retrieved_length(tmp_path):
    """Success envelope's node_count is len(retrieve return value) not len(format_context output)."""
    p = _make_initialized_provider(tmp_path)
    nodes = [MagicMock() for _ in range(5)]
    p._retriever.retrieve.return_value = nodes
    p._retriever.format_context.return_value = "short"  # format_context len unrelated
    try:
        d = json.loads(p.handle_tool_call("cashew_query", {"query": "x"}))
        assert d["node_count"] == 5
    finally:
        p.shutdown()
