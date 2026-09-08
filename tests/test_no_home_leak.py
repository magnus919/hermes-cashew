# tests/test_no_home_leak.py
# TEST-03: assert a tmp_path-scoped lifecycle never falls back to ~/.hermes.

from __future__ import annotations

from plugins.memory.cashew import CashewMemoryProvider


def test_full_lifecycle_does_not_touch_home(tmp_path, isolated_user_home):
    """A full tmp_path lifecycle must not create the default Hermes profile."""
    p = CashewMemoryProvider()
    p.save_config(
        {"recall_k": 9, "embedding_model": "BAAI/bge-small-en"}, str(tmp_path)
    )
    p.initialize("session-no-leak", hermes_home=str(tmp_path))
    # Mid-lifecycle check: cashew.json exists under tmp_path
    assert (tmp_path / "cashew.json").exists()
    # And under hermes_home, NOT under ~
    p.shutdown()

    # Explicit assertion — the fixture also runs this at teardown for safety.
    isolated_user_home()


def test_save_config_writes_only_under_tmp_path(tmp_path, isolated_user_home):
    """Defensive: even an isolated save_config (no initialize) must not touch ~."""
    p = CashewMemoryProvider()
    p.save_config({"recall_k": 5}, str(tmp_path))
    assert (tmp_path / "cashew.json").exists()
    isolated_user_home()
