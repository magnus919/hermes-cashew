# tests/test_recall.py
# Phase 3 Plan 03-03 Task 1: prefetch behavior — RECALL-01 + RECALL-04.
#
# Strategy (PHASE_DESIGN_NOTES Decision Point 5): mock ContextRetriever at the
# instance level. We never hit a real SQLite DB here; that's Phase 5's `verify`
# CLI responsibility.
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import DEFAULTS


def _make_initialized_provider(tmp_path, recall_k: int | None = None) -> CashewMemoryProvider:
    """Construct a provider, save a minimal config, initialize it, then swap in
    a MagicMock for _retriever so tests can control retrieve/format_context behavior
    without touching a real DB.

    If recall_k is specified, it's saved to the on-disk config so _config.recall_k
    reflects it after initialize.
    """
    p = CashewMemoryProvider()
    overrides = {"recall_k": recall_k} if recall_k is not None else {}
    p.save_config(overrides, str(tmp_path))
    p.initialize("sess-1", hermes_home=str(tmp_path))
    # Replace the real ContextRetriever (constructed by initialize per Plan 03-01)
    # with a MagicMock so we control retrieve + format_context return values.
    p._retriever = MagicMock()
    return p


def test_prefetch_happy_path_returns_formatted_context(tmp_path):
    """RECALL-01: retrieve + format_context produces a non-empty recalled context string."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.return_value = [MagicMock(), MagicMock(), MagicMock()]  # 3 nodes
    p._retriever.format_context.return_value = "formatted context string"
    try:
        assert p.prefetch("what did I decide about X?") == "formatted context string"
    finally:
        p.shutdown()


def test_prefetch_honors_recall_k(tmp_path):
    """RECALL-01: recall_k from config MUST be the max_nodes passed to retrieve()."""
    p = _make_initialized_provider(tmp_path, recall_k=7)
    p._retriever.retrieve.return_value = []
    p._retriever.format_context.return_value = ""
    try:
        p.prefetch("x")
        p._retriever.retrieve.assert_called_once_with("x", max_nodes=7)
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
        # There's exactly ONE WARNING from initialize at this point.
        init_warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(init_warnings) == 1
        # Now prefetch — must NOT add a second WARNING.
        result = p.prefetch("anything")
    try:
        assert result == ""
        total_warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(total_warnings) == 1, (
            f"prefetch must not double-log after init-time silent-degrade; got {[r.getMessage() for r in total_warnings]}"
        )
    finally:
        p.shutdown()


def test_prefetch_retrieve_exception_logs_once_and_returns_empty(tmp_path, caplog):
    """RECALL-04: retrieve() raises -> "" + ONE WARNING with exc_info."""
    import sqlite3
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.side_effect = sqlite3.OperationalError("database is locked")
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.prefetch("x")
        assert result == ""
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1, f"expected 1 WARNING; got {len(warnings)}"
        assert "cashew recall failed" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None, "exc_info must be attached"
    finally:
        p.shutdown()


def test_prefetch_format_context_exception_logs_once_and_returns_empty(tmp_path, caplog):
    """RECALL-04: format_context() raises -> same contract as retrieve() raising."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.return_value = [MagicMock()]
    p._retriever.format_context.side_effect = KeyError("parent_chain")
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.prefetch("x")
        assert result == ""
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "cashew recall failed" in warnings[0].getMessage()
        assert warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_prefetch_empty_retrieval_returns_empty_no_log(tmp_path, caplog):
    """Empty graph (retrieve returns []) is not a failure — empty string, no log."""
    p = _make_initialized_provider(tmp_path)
    p._retriever.retrieve.return_value = []
    p._retriever.format_context.return_value = ""
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            assert p.prefetch("x") == ""
        assert not any("recall failed" in r.getMessage() for r in caplog.records)
    finally:
        p.shutdown()
