"""Tests for the sleep cycle cron job lifecycle (v0.11.0).

The sleep cycle was migrated from ``on_session_end()`` to a Hermes ``no_agent``
cron job. These tests verify that:
1. ``initialize()`` registers the cron job when sleeping is enabled
2. ``initialize()`` skips cron registration when sleeping is disabled
3. ``shutdown()`` removes the cron job
4. The cron script is installed correctly

All tests in this file require the Hermes ``cron`` module, which is only
available in a full Hermes Agent environment — not in CI or standalone
test runs. The module-level skip handles this automatically.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

pytest.importorskip("cron.jobs", reason="Hermes Agent cron module not available")


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
    cfg = hermes_home / "cashew.json"
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")
    cfg.write_text(json.dumps(_make_config(
        hermes_home,
        sleep_cycles=True,
        sleep_schedule="every 6h",
    )))

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
    cfg = hermes_home / "cashew.json"
    cfg.write_text(json.dumps(_make_config(
        hermes_home,
        sleep_cycles=True,
        sleep_schedule="",
    )))

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


def test_shutdown_removes_cron_job(tmp_path, monkeypatch):
    """shutdown() removes the registered cron job."""
    hermes_home = tmp_path / "h4"
    hermes_home.mkdir()
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

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    assert provider._sleep_cron_job_id == "cron-job-789"

    provider.shutdown()

    assert len(remove_calls) == 1
    assert remove_calls[0] == "cron-job-789"
    assert provider._sleep_cron_job_id is None  # cleared after removal


def test_cron_script_is_installed(tmp_path, monkeypatch):
    """initialize() writes the cron script to $HERMES_HOME/scripts/."""
    hermes_home = tmp_path / "h5"
    hermes_home.mkdir()
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
    assert "plugins.memory.cashew.sleep_refactor" in content

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
    script_source = (Path(__file__).parent.parent / "plugins" / "memory"
                     / "cashew" / "sleep_cron_script.py").read_text()
    script_path.write_text(script_source)
    script_path.chmod(0o755)

    # Verify the script contains the resolve_model_fn import
    assert "resolve_model_fn" in script_source
    assert "model_fn = _resolve_model_fn" in script_source or \
           "resolve_model_fn(hermes_home" in script_source
    assert "model_fn=model_fn" in script_source
    # No longer hardcoded None
    assert "model_fn=None" not in script_source.replace(
        "# model_fn=None (fallback)", ""
    )
