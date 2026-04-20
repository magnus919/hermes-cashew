# tests/test_initialize_lifecycle.py
# Phase 2 Plan 02-03: provider-level tests through the ABC surface.
# Covers initialize/shutdown/is_available, queue creation, threading baseline,
# silent-degrade on corrupt config, and "no Cashew runtime imported by initialize".

from __future__ import annotations

import json
import queue
import sys
import threading

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig, DEFAULTS, CONFIG_FILENAME


def test_fresh_provider_is_not_available():
    """ABC-04 + Phase 2 Success #3: pre-initialize, is_available is False unconditionally."""
    p = CashewMemoryProvider()
    assert p.is_available() is False


def test_shutdown_pre_initialize_is_safe_noop():
    """Phase 2 Success #5: shutdown without prior initialize must not raise or leak threads."""
    p = CashewMemoryProvider()
    baseline = threading.active_count()
    p.shutdown()  # must not raise
    assert threading.active_count() == baseline


def test_initialize_without_hermes_home_raises_keyerror():
    """ABC-04: hermes_home is mandatory. Error message must name it (UX-04 actionable)."""
    p = CashewMemoryProvider()
    with pytest.raises(KeyError) as exc:
        p.initialize("session-x", platform="cli")
    assert "hermes_home" in str(exc.value)


def test_initialize_creates_bounded_queue(tmp_path):
    """Queue is created with maxsize=16 in initialize(); Phase 4 now also starts the worker.

    PHASE_DESIGN_NOTES Decision Point 4: queue.Queue(maxsize=16) per CLAUDE.md ### Threading Rule.
    Phase 4 (Plan 04-01) landed the non-daemon worker thread, so initialize() now
    adds exactly one thread to active_count(); shutdown() returns it to baseline
    (covered by test_initialize_then_shutdown_returns_to_baseline_threads).
    """
    p = CashewMemoryProvider()
    baseline = threading.active_count()
    p.initialize("session-x", hermes_home=str(tmp_path))
    try:
        assert isinstance(p._sync_queue, queue.Queue)
        assert p._sync_queue.maxsize == 16
        assert threading.active_count() == baseline + 1, (
            "Phase 4 expects exactly one worker thread started by initialize()"
        )
    finally:
        p.shutdown()


def test_initialize_then_shutdown_returns_to_baseline_threads(tmp_path):
    """Phase 2 Success #5: init → shutdown leaves threading.active_count() at baseline."""
    p = CashewMemoryProvider()
    baseline = threading.active_count()
    p.initialize("s", hermes_home=str(tmp_path))
    p.shutdown()
    assert threading.active_count() == baseline


def test_initialize_loads_default_config_when_no_file(tmp_path):
    """No cashew.json yet → load_config returns DEFAULTS, provider stores them."""
    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        assert isinstance(p._config, CashewConfig)
        assert p._config.recall_k == DEFAULTS["recall_k"]
        assert p._config.cashew_db_path == DEFAULTS["cashew_db_path"]
    finally:
        p.shutdown()


def test_initialize_resolves_db_path_under_hermes_home(tmp_path):
    """CONF-03: DB at $HERMES_HOME/cashew/brain.db (nested) when default is in effect."""
    import pathlib
    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        assert p._db_path == pathlib.Path(str(tmp_path)) / "cashew" / "brain.db"
    finally:
        p.shutdown()


def test_full_roundtrip_save_then_initialize(tmp_path):
    """Goal-level: save_config + initialize round-trips a non-default value through the file."""
    p = CashewMemoryProvider()
    p.save_config({"recall_k": 9}, str(tmp_path))
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        assert p._config.recall_k == 9
    finally:
        p.shutdown()


def test_is_available_transitions_false_to_true_after_save_config(tmp_path):
    """Phase 2 Success #3: is_available True iff cashew.json exists under hermes_home."""
    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        assert p.is_available() is False, "no cashew.json yet"
        p.save_config({}, str(tmp_path))
        assert p.is_available() is True, "cashew.json now exists"
    finally:
        p.shutdown()


def test_is_available_calls_path_exists_exactly_once(tmp_path, monkeypatch):
    """Pitfall 5 enforcement: zero I/O beyond ONE Path.exists call. No content reads."""
    import pathlib
    open_calls = []
    read_text_calls = []
    exists_calls = []

    real_exists = pathlib.Path.exists
    real_read_text = pathlib.Path.read_text

    def _exists(self):
        exists_calls.append(str(self))
        return real_exists(self)

    def _read_text(self, *args, **kwargs):
        read_text_calls.append(str(self))
        return real_read_text(self, *args, **kwargs)

    real_open = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open

    def _open(*args, **kwargs):
        open_calls.append(args[0] if args else None)
        return real_open(*args, **kwargs)

    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        # Reset counters AFTER initialize (which legitimately reads files).
        exists_calls.clear()
        read_text_calls.clear()
        open_calls.clear()
        monkeypatch.setattr("pathlib.Path.exists", _exists)
        monkeypatch.setattr("pathlib.Path.read_text", _read_text)
        monkeypatch.setattr("builtins.open", _open)

        p.is_available()

        assert len(exists_calls) == 1, f"expected exactly 1 Path.exists call, got {exists_calls}"
        assert read_text_calls == [], f"is_available read file content: {read_text_calls}"
        assert open_calls == [], f"is_available called open(): {open_calls}"
    finally:
        p.shutdown()


def test_initialize_does_not_import_cashew_runtime(tmp_path):
    """Phase 2 ≠ Phase 3: initialize must NOT instantiate ContextRetriever or load embeddings."""
    pre = set(sys.modules.keys())
    p = CashewMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    try:
        post = set(sys.modules.keys())
        new = post - pre
        forbidden = {"core.embeddings"}  # Phase 3: core.context is now legitimately loaded at initialize; embedding still lazy.
        assert not (new & forbidden), (
            f"initialize() loaded forbidden Cashew runtime modules: {new & forbidden}"
        )
    finally:
        p.shutdown()


def test_initialize_silent_degrades_on_corrupt_config(tmp_path, caplog):
    """PROJECT.md Key Decision: silent degrade on Cashew failures. Corrupt JSON → WARNING + None config."""
    (tmp_path / CONFIG_FILENAME).write_text("not json {")
    p = CashewMemoryProvider()
    with caplog.at_level("WARNING"):
        p.initialize("s", hermes_home=str(tmp_path))
    try:
        assert p._config is None, "corrupt config must result in _config=None (silent degrade)"
        assert p._db_path is None
        # WARNING was logged with exc_info
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("cashew initialize failed" in r.getMessage() for r in warnings), (
            f"expected silent-degrade WARNING; got {[r.getMessage() for r in warnings]}"
        )
        # is_available probes the file directly — the file still exists, just is corrupt.
        # The contract is "config file exists" not "config is parseable", so True is correct.
        # Phase 3 prefetch() must None-guard self._config since is_available can be True
        # while _config is None (the half-state flagged in 02-02-SUMMARY.md).
        assert p.is_available() is True
    finally:
        p.shutdown()


def test_get_config_schema_returns_helper_module_schema():
    """The provider's ABC method delegates verbatim to the helper — no inline schema definition."""
    from plugins.memory.cashew.config import get_config_schema as helper_schema
    p = CashewMemoryProvider()
    assert p.get_config_schema() == helper_schema()
