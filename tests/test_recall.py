# tests/test_recall.py
# Phase 3 + Phase 9: prefetch behavior — RECALL-01 + RECALL-04.
#
# Phase 9 update: tests now seed a real SQLite DB instead of mocking ContextRetriever,
# since prefetch() implements its own three-tier retrieval (vec → keyword fallback).
from __future__ import annotations

import logging
import sqlite3

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS, resolve_db_path


def _make_initialized_provider(tmp_path, recall_k: int | None = None) -> CashewMemoryProvider:
    """Construct a provider, save a minimal config, and initialize it."""
    p = CashewMemoryProvider()
    overrides = {"recall_k": recall_k} if recall_k is not None else {}
    p.save_config(overrides, str(tmp_path))
    p.initialize("sess-1", hermes_home=str(tmp_path))
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


def test_prefetch_happy_path_returns_formatted_context(tmp_path):
    """RECALL-01: retrieve + format_context produces a non-empty recalled context string."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="hello world", node_type="belief", domain="test")
    try:
        result = p.prefetch("hello")
        assert "hello world" in result
        assert "=== RELEVANT CONTEXT ===" in result
    finally:
        p.shutdown()


def test_prefetch_honors_recall_k(tmp_path):
    """RECALL-01: recall_k from config caps the number of returned nodes."""
    p = _make_initialized_provider(tmp_path, recall_k=2)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    for i in range(5):
        _seed_node(db, id=f"n{i}", content=f"shared keyword {i}")
    try:
        result = p.prefetch("shared keyword")
        # Keyword search finds all 5, but recall_k=2 caps formatted output
        assert result.count("shared keyword") <= 2
    finally:
        p.shutdown()


def test_prefetch_half_state_returns_empty_no_log(tmp_path, caplog):
    """RECALL-04: pre-initialize (no _retriever) -> "" + NO log record (double-log prevention)."""
    p = CashewMemoryProvider()  # never initialized
    assert p._retriever is None
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        assert p.prefetch("x") == ""
    assert not any("recall failed" in r.getMessage() for r in caplog.records), (
        f"pre-initialize prefetch should not log; got {[r.getMessage() for r in caplog.records]}"
    )


def test_prefetch_corrupt_config_half_state_returns_empty_no_new_log(tmp_path, caplog):
    """RECALL-04 + Phase 2 carryover: corrupt config -> initialize emits WARNING;
    subsequent prefetch MUST NOT emit a second WARNING."""
    from plugins.memory.cashew.config import CONFIG_FILENAME
    (tmp_path / CONFIG_FILENAME).write_text("not json {")
    p = CashewMemoryProvider()
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        p.initialize("s", hermes_home=str(tmp_path))
        init_warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(init_warnings) == 1
        result = p.prefetch("anything")
    try:
        assert result == ""
        total_warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(total_warnings) == 1, (
            f"prefetch must not double-log after init-time silent-degrade; got {[r.getMessage() for r in total_warnings]}"
        )
    finally:
        p.shutdown()


def test_prefetch_retrieval_exception_logs_once_and_returns_empty(tmp_path, caplog, monkeypatch):
    """RECALL-04: retrieval raises -> "" + ONE WARNING with exc_info."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="test")

    def _raise(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(p, "_retrieve_with_vec", _raise)
    monkeypatch.setattr(p, "_retrieve_keyword", _raise)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.prefetch("test")
        assert result == ""
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, f"expected 1 WARNING; got {len(warnings)}"
        assert "cashew recall failed" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None, "exc_info must be attached"
    finally:
        p.shutdown()


def test_prefetch_format_context_exception_logs_once_and_returns_empty(tmp_path, caplog, monkeypatch):
    """RECALL-04: _format_context() raises -> same contract as retrieve() raising."""
    p = _make_initialized_provider(tmp_path)
    db = resolve_db_path(tmp_path, DEFAULTS["cashew_db_path"])
    _seed_node(db, id="n1", content="test")

    def _raise(*args, **kwargs):
        raise KeyError("boom")

    monkeypatch.setattr(p, "_format_context", _raise)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.prefetch("test")
        assert result == ""
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "cashew recall failed" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_prefetch_empty_retrieval_returns_empty_no_log(tmp_path, caplog):
    """Empty graph (no matching nodes) is not a failure — empty string, no log."""
    p = _make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            assert p.prefetch("nonexistent_query_xyz") == ""
        assert not any("recall failed" in r.getMessage() for r in caplog.records)
    finally:
        p.shutdown()
