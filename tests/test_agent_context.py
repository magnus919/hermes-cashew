"""Write-safety contracts for Hermes agent execution contexts."""

from __future__ import annotations

import json

import pytest

from plugins.memory.cashew import CashewMemoryProvider


@pytest.mark.parametrize(
    ("agent_context", "write_enabled"),
    [
        ("primary", True),
        ("cron", False),
        ("subagent", False),
        ("flush", False),
        ("future-context", False),
    ],
)
def test_initialize_fails_closed_for_non_primary_contexts(
    tmp_path, agent_context: str, write_enabled: bool
) -> None:
    provider = CashewMemoryProvider()
    provider.save_config({}, str(tmp_path))
    provider.initialize(
        "context-test",
        hermes_home=str(tmp_path),
        agent_context=agent_context,
    )
    try:
        assert provider._write_enabled is write_enabled
    finally:
        provider.shutdown()


def test_non_primary_context_rejects_background_and_explicit_writes(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        "core.session.end_session",
        lambda **kwargs: calls.append(kwargs),
        raising=False,
    )
    provider = CashewMemoryProvider()
    provider.save_config({}, str(tmp_path))
    provider.initialize(
        "cron-session",
        hermes_home=str(tmp_path),
        agent_context="cron",
    )
    try:
        provider.sync_turn("system task", "system output")
        result = json.loads(
            provider.handle_tool_call(
                "cashew_extract",
                {"user_content": "system task", "assistant_content": "system output"},
            )
        )
        assert calls == []
        assert result["ok"] is False
    finally:
        provider.shutdown()
