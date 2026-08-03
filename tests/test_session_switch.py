"""Hermes mid-process session-switch contract tests."""

from plugins.memory.cashew import CashewMemoryProvider


def test_session_switch_rebinds_identity_and_clears_ephemeral_context() -> None:
    provider = CashewMemoryProvider()
    provider._session_id = "session-a"
    provider._warm_cache["old cue"] = "old context"
    provider._prefetch_pending = "pending old context"
    provider._last_assistant = "old assistant response"

    provider.on_session_switch(
        "session-b",
        parent_session_id="session-a",
        reset=False,
        rewound=True,
        future_contract_field="ignored",
    )

    assert provider._session_id == "session-b"
    assert provider._warm_cache == {}
    assert provider._prefetch_pending is None
    assert provider._last_assistant == ""
