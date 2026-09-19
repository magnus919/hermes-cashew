"""Tests for persistent profile-owned sleep-cron lifecycle behavior.

The sleep cycle was migrated from ``on_session_end()`` to a Hermes ``no_agent``
cron job. These tests verify that:
1. ``initialize()`` registers the cron job when sleeping is enabled
2. ``initialize()`` skips cron registration when sleeping is disabled
3. ordinary ``shutdown()`` preserves the profile-owned job for adoption
4. the cron script is installed correctly

These lifecycle tests install the current minimal Hermes cron API in-process so
they run in the standalone suite. The reconciliation module retains a separate
real-store test that skips only when Hermes itself is unavailable.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _fake_cron_api(monkeypatch):
    """Provide the current Hermes cron surface for installation-independent tests."""
    package = types.ModuleType("cron")
    package.__path__ = []
    jobs = types.ModuleType("cron.jobs")
    jobs.list_jobs = lambda *, include_disabled=False: []
    jobs.remove_job = lambda _job_id: None
    jobs.parse_schedule = lambda schedule: schedule
    jobs.create_job = lambda **_kwargs: {"id": "fake-job-id"}
    jobs.update_job = lambda _job_id, _updates: None

    @contextmanager
    def use_cron_store(_home):
        yield

    jobs.use_cron_store = use_cron_store
    monkeypatch.setitem(sys.modules, "cron", package)
    monkeypatch.setitem(sys.modules, "cron.jobs", jobs)
    monkeypatch.setattr("plugins.memory.cashew._HAS_HERMES_CRON", True)


def _install_development_anchor(hermes_home: Path) -> None:
    """Model the supported development layout required by cron registration."""
    source = Path(__file__).parents[1] / "plugins" / "memory" / "cashew"
    anchor = hermes_home / "hermes-agent" / "plugins" / "memory" / "cashew"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.symlink_to(source, target_is_directory=True)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS thought_nodes ("
        "  id TEXT PRIMARY KEY, content TEXT, node_type TEXT DEFAULT 'observation',"
        "  permanent INTEGER DEFAULT 0, decayed INTEGER DEFAULT 0,"
        "  access_count INTEGER DEFAULT 0, timestamp TEXT"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "  node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT"
        ")"
    )
    conn.commit()


def _make_config(hermes_home: Path, **overrides: str | bool | int) -> dict:
    """Build a cashew.json dict with defaults + overrides."""
    cfg = {
        "cashew_db_path": "cashew/brain.db",
        "sleep_cycles": True,
        "sleep_schedule": "every 12h",
        "think_cycles": False,
    }
    cfg.update(overrides)
    return cfg


# ── Cron registration tests ─────────────────────────────────────────────────


def test_initialize_skips_cron_when_sleep_disabled(tmp_path, monkeypatch):
    """When sleep_cycles=false, initialize() does NOT register a cron job."""
    hermes_home = tmp_path / "h1"
    hermes_home.mkdir()
    _install_development_anchor(hermes_home)
    cfg = hermes_home / "cashew.json"
    cfg.write_text(json.dumps(_make_config(hermes_home, sleep_cycles=False)))

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

    # Track whether create_job was called
    calls = []

    def fake_create_job(**kwargs):
        calls.append(kwargs)
        return {"id": "fake-job-id"}

    monkeypatch.setattr(
        "plugins.memory.cashew._remove_existing_sleep_job",
        lambda *a: None,
    )
    monkeypatch.setattr(
        "plugins.memory.cashew.CashewMemoryProvider._hermes_home",
        hermes_home,
        raising=False,
    )
    # Monkeypatch cron.jobs.create_job at the module level
    import cron.jobs as cron_jobs

    monkeypatch.setattr(cron_jobs, "create_job", fake_create_job)

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    # We can't check calls directly since _register_sleep_cron guards on
    # sleep_cycles being True. The fact that initialize() succeeded without
    # error is the basic pass. Instead check that cron wasn't registered
    # by examining the internal state:
    assert provider._sleep_cron_job_id is None

    provider.shutdown()


def test_initialize_registers_cron_when_sleep_enabled(tmp_path, monkeypatch):
    """When sleep_cycles=true and sleep_schedule is set, initialize() registers a cron job."""
    hermes_home = tmp_path / "h2"
    hermes_home.mkdir()
    _install_development_anchor(hermes_home)
    cfg = hermes_home / "cashew.json"
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")
    cfg.write_text(
        json.dumps(
            _make_config(
                hermes_home,
                sleep_cycles=True,
                sleep_schedule="every 6h",
            )
        )
    )

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

    calls = []

    def fake_create_job(**kwargs):
        calls.append(kwargs)
        return {"id": "cron-job-123"}

    monkeypatch.setattr(
        "plugins.memory.cashew._remove_existing_sleep_job",
        lambda *a: None,
    )
    import cron.jobs as cron_jobs

    monkeypatch.setattr(cron_jobs, "create_job", fake_create_job)

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    assert provider._sleep_cron_job_id == "cron-job-123"

    provider.shutdown()


def test_initialize_skips_cron_when_no_schedule(tmp_path, monkeypatch):
    """When sleep_schedule is empty, initialize() does NOT register a cron job."""
    hermes_home = tmp_path / "h3"
    hermes_home.mkdir()
    _install_development_anchor(hermes_home)
    cfg = hermes_home / "cashew.json"
    cfg.write_text(
        json.dumps(
            _make_config(
                hermes_home,
                sleep_cycles=True,
                sleep_schedule="",
            )
        )
    )

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

    calls = []

    def fake_create_job(**kwargs):
        calls.append(kwargs)
        return {"id": "cron-job-456"}

    monkeypatch.setattr(
        "plugins.memory.cashew._remove_existing_sleep_job",
        lambda *a: None,
    )
    import cron.jobs as cron_jobs

    monkeypatch.setattr(cron_jobs, "create_job", fake_create_job)

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    # sleep_schedule="" means cron is skipped
    assert provider._sleep_cron_job_id is None
    assert len(calls) == 0

    provider.shutdown()


def test_shutdown_preserves_profile_owned_cron_job(tmp_path, monkeypatch):
    """Ordinary session shutdown leaves scheduled profile maintenance intact."""
    hermes_home = tmp_path / "h4"
    hermes_home.mkdir()
    _install_development_anchor(hermes_home)
    cfg = hermes_home / "cashew.json"
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")
    cfg.write_text(json.dumps(_make_config(hermes_home)))

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

    remove_calls = []

    def fake_remove_job(job_id):
        remove_calls.append(job_id)

    def fake_create_job(**kwargs):
        return {"id": "cron-job-789"}

    monkeypatch.setattr(
        "plugins.memory.cashew._remove_existing_sleep_job",
        lambda *a: None,
    )
    import cron.jobs as cron_jobs

    monkeypatch.setattr(cron_jobs, "create_job", fake_create_job)
    monkeypatch.setattr(cron_jobs, "remove_job", fake_remove_job)
    monkeypatch.setattr(cron_jobs, "list_jobs", lambda *, include_disabled=False: [])

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    assert provider._sleep_cron_job_id == "cron-job-789"

    provider.shutdown()

    assert remove_calls == []
    assert provider._sleep_cron_job_id is None  # instance tracking is cleared


def test_cron_script_is_installed(tmp_path, monkeypatch):
    """initialize() writes the cron script to $HERMES_HOME/scripts/."""
    hermes_home = tmp_path / "h5"
    hermes_home.mkdir()
    _install_development_anchor(hermes_home)
    cfg = hermes_home / "cashew.json"
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")
    cfg.write_text(json.dumps(_make_config(hermes_home)))

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

    monkeypatch.setattr(
        "plugins.memory.cashew._remove_existing_sleep_job",
        lambda *a: None,
    )
    import cron.jobs as cron_jobs

    def fake_create_job(**kwargs):
        return {"id": "job-script-test"}

    monkeypatch.setattr(cron_jobs, "create_job", fake_create_job)

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    script_path = hermes_home / "scripts" / "cashew-sleep-cycle.py"
    assert script_path.exists(), f"Cron script not found at {script_path}"
    content = script_path.read_text()
    assert "run_sleep_cycle" in content
    assert "_INSTALLATION_MARKER = {" in content
    assert "_load_profile_modules" in content

    provider.shutdown()


def test_cron_script_imports_resolve_model_fn(tmp_path):
    """The cron script imports resolve_model_fn and passes it to run_sleep_cycle."""
    hermes_home = tmp_path / "h6"
    hermes_home.mkdir()
    cfg = hermes_home / "cashew.json"
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")
    cfg.write_text(json.dumps(_make_config(hermes_home)))

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

    # Install the cron script
    script_path = hermes_home / "scripts" / "cashew-sleep-cycle.py"
    script_path.parent.mkdir(parents=True)
    script_source = (
        Path(__file__).parent.parent
        / "plugins"
        / "memory"
        / "cashew"
        / "sleep_cron_script.py"
    ).read_text()
    script_path.write_text(script_source)
    script_path.chmod(0o755)

    # The generated entry point resolves the model through the profile-pinned
    # config module loaded from its validated installation marker.
    assert "config_module.resolve_model_fn(" in script_source
    assert "hermes_home=hermes_home, config=config" in script_source
    assert "model_fn=model_fn" in script_source
    # No longer hardcoded None
    assert "model_fn=None" not in script_source.replace(
        "# model_fn=None (fallback)", ""
    )
