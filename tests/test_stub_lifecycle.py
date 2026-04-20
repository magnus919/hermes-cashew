# tests/test_stub_lifecycle.py
# Phase 1 acceptance: the stub provider has the right shape and refuses to be "available" pre-config,
# without performing any filesystem or network I/O.
from plugins.memory.cashew import CashewMemoryProvider


def test_name_is_cashew():
    """ABC-01: name property returns 'cashew'."""
    assert CashewMemoryProvider().name == "cashew"


def test_is_available_false_before_config():
    """ABC-03: is_available returns False with zero I/O before Phase 2 config round-trip."""
    provider = CashewMemoryProvider()
    assert provider.is_available() is False


def test_is_available_no_filesystem_calls(monkeypatch):
    """ABC-03: no filesystem I/O on is_available()."""
    import pathlib
    calls = []
    original_exists = pathlib.Path.exists
    def _track_exists(self):
        calls.append(("exists", str(self)))
        return original_exists(self)
    monkeypatch.setattr("pathlib.Path.exists", _track_exists)

    original_open = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open
    open_calls = []
    def _track_open(*args, **kwargs):
        open_calls.append(args[0] if args else None)
        return original_open(*args, **kwargs)
    monkeypatch.setattr("builtins.open", _track_open)

    CashewMemoryProvider().is_available()
    assert calls == [], f"is_available() called Path.exists: {calls}"
    assert open_calls == [], f"is_available() called open(): {open_calls}"
