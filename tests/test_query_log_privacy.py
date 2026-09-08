"""Privacy contracts for memory-query operational logging."""

from __future__ import annotations

import logging

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig


def _provider(tmp_path) -> CashewMemoryProvider:
    provider = CashewMemoryProvider()
    provider._config = CashewConfig()
    provider._db_path = tmp_path / "brain.db"
    return provider


def test_warm_cache_hit_logs_lengths_not_query_content(caplog, tmp_path):
    sensitive = "HIV treatment plan for Alice"
    provider = _provider(tmp_path)
    provider._warm_cache[sensitive] = "private context"

    with caplog.at_level(logging.INFO, logger="plugins.memory.cashew"):
        assert provider.prefetch(sensitive) == "private context"

    assert sensitive not in caplog.text
    assert "Alice" not in caplog.text
    assert f"query_len={len(sensitive)}" in caplog.text


def test_recall_failure_logs_length_not_query_content(caplog, monkeypatch, tmp_path):
    sensitive = "bankruptcy filing for Robert"
    provider = _provider(tmp_path)
    monkeypatch.setattr(
        provider,
        "_keyword_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
    )

    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        assert provider.prefetch(sensitive) == ""

    assert sensitive not in caplog.text
    assert "Robert" not in caplog.text
    assert f"query_len={len(sensitive)}" in caplog.text
