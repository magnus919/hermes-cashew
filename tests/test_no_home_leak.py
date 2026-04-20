# tests/test_no_home_leak.py
# Phase 2 Plan 02-03 — TEST-03: assert ~/.hermes mtime + listing unchanged
# after a full save_config + initialize + shutdown lifecycle against tmp_path.

from __future__ import annotations

from plugins.memory.cashew import CashewMemoryProvider


def test_full_lifecycle_does_not_touch_home(tmp_path, home_snapshot):
    """TEST-03 + Phase 2 Success #2: save_config + initialize + shutdown vs tmp_path
    must leave the user's real ~/.hermes mtime + file listing unchanged.

    Uses the home_snapshot fixture (defined in tests/conftest.py) which captures
    pre-test state and provides assert_unchanged() for explicit verification.
    The fixture also calls assert_unchanged() at teardown as belt-and-suspenders.
    """
    p = CashewMemoryProvider()
    p.save_config({"recall_k": 9, "embedding_model": "BAAI/bge-small-en"}, str(tmp_path))
    p.initialize("session-no-leak", hermes_home=str(tmp_path))
    # Mid-lifecycle check: cashew.json exists under tmp_path
    assert (tmp_path / "cashew.json").exists()
    # And under hermes_home, NOT under ~
    p.shutdown()

    # Explicit assertion — the fixture also runs this at teardown for safety.
    home_snapshot["assert_unchanged"]()


def test_save_config_writes_only_under_tmp_path(tmp_path, home_snapshot):
    """Defensive: even an isolated save_config (no initialize) must not touch ~."""
    p = CashewMemoryProvider()
    p.save_config({"recall_k": 5}, str(tmp_path))
    assert (tmp_path / "cashew.json").exists()
    home_snapshot["assert_unchanged"]()
