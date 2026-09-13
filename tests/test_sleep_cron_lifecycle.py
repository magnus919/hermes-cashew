"""Tests for persistent profile-owned sleep-cron lifecycle behavior.

The sleep cycle was migrated from ``on_session_end()`` to a Hermes ``no_agent``
cron job. These tests verify that:
1. ``initialize()`` registers the cron job when sleeping is enabled
2. ``initialize()`` skips cron registration when sleeping is disabled
3. ordinary ``shutdown()`` preserves the profile-owned job for adoption
4. the cron script is installed correctly

All tests in this file require the Hermes ``cron`` module, which is only
available in a full Hermes Agent environment — not in CI or standalone
test runs. The module-level skip handles this automatically.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

if os.environ.get("HERMES_CASHEW_REAL_HERMES") == "1":
    try:
        import cron.jobs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pinned Hermes host lane could not import cron.jobs"
        ) from exc
else:
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


def _install_provider_layout(hermes_home: Path, layout: str) -> None:
    """Install a valid flat or development anchor for the real host lane."""
    source = Path(__file__).parents[1] / "plugins" / "memory" / "cashew"
    if layout == "flat":
        target = hermes_home / "plugins" / "cashew" / "plugins" / "memory" / "cashew"
    elif layout == "development":
        target = hermes_home / "hermes-agent" / "plugins" / "memory" / "cashew"
    else:
        raise ValueError(f"unknown provider layout: {layout}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def _prepare_real_cron_home(hermes_home: Path, layout: str) -> None:
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "model:\n  provider: test\n  default: test\n"
    )
    (hermes_home / "cashew.json").write_text(
        json.dumps(_make_config(hermes_home, sleep_schedule="every 6h"))
    )
    _install_provider_layout(hermes_home, layout)
    (hermes_home / "cashew").mkdir(parents=True)
    with sqlite3.connect(str(hermes_home / "cashew" / "brain.db")) as conn:
        _ensure_schema(conn)


# ── Cron registration tests ─────────────────────────────────────────────────


def test_pinned_host_provenance_and_isolated_environment():
    """The host subprocess uses only the requested Hermes checkout and sandbox."""
    cron_jobs = pytest.importorskip("cron.jobs")
    expected_root = os.environ.get("HERMES_CASHEW_HERMES_ROOT")
    sandbox = os.environ.get("HERMES_CASHEW_HOST_SANDBOX")
    assert expected_root and sandbox
    assert Path(cron_jobs.__file__).resolve().is_relative_to(
        Path(expected_root).resolve() / "cron"
    )
    sandbox_path = Path(sandbox).resolve()
    for name in (
        "HOME",
        "HERMES_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "MPLCONFIGDIR",
        "TMPDIR",
    ):
        assert Path(os.environ[name]).resolve().is_relative_to(sandbox_path)


@pytest.mark.parametrize("layout", ["flat", "development"])
def test_real_cron_store_persists_create_list_update_remove(tmp_path, layout):
    """The real pinned Hermes store persists the complete CRUD lifecycle."""
    cron_jobs = pytest.importorskip("cron.jobs")
    hermes_home = tmp_path / layout
    _prepare_real_cron_home(hermes_home, layout)
    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="real-crud", hermes_home=str(hermes_home))
    try:
        job_id = provider._sleep_cron_job_id
        assert isinstance(job_id, str)
        with cron_jobs.use_cron_store(hermes_home):
            persisted = cron_jobs.list_jobs(include_disabled=True)
            assert [job["id"] for job in persisted] == [job_id]
            assert cron_jobs.update_job(job_id, {"enabled": False})["enabled"] is False
            assert cron_jobs.list_jobs() == []
            paused = cron_jobs.list_jobs(include_disabled=True)
            assert paused[0]["id"] == job_id
            assert cron_jobs.remove_job(job_id) is True
            assert cron_jobs.list_jobs(include_disabled=True) == []
    finally:
        provider.shutdown()


@pytest.mark.parametrize("layout", ["flat", "development"])
def test_initialize_skips_cron_when_sleep_disabled(tmp_path, monkeypatch, layout):
    """When sleep_cycles=false, initialize() does NOT register a cron job."""
    hermes_home = tmp_path / "h1"
    hermes_home.mkdir()
    _install_provider_layout(hermes_home, layout)
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


@pytest.mark.parametrize("layout", ["flat", "development"])
def test_initialize_registers_cron_when_sleep_enabled(tmp_path, monkeypatch, layout):
    """When sleep_cycles=true and sleep_schedule is set, initialize() registers a cron job."""
    hermes_home = tmp_path / "h2"
    hermes_home.mkdir()
    _install_provider_layout(hermes_home, layout)
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

    import cron.jobs as cron_jobs

    monkeypatch.setattr(cron_jobs, "create_job", fake_create_job)

    from plugins.memory.cashew import CashewMemoryProvider

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test", hermes_home=str(hermes_home))

    time.sleep(0.3)

    assert provider._sleep_cron_job_id == "cron-job-123"

    provider.shutdown()


@pytest.mark.parametrize("layout", ["flat", "development"])
def test_initialize_skips_cron_when_no_schedule(tmp_path, monkeypatch, layout):
    """When sleep_schedule is empty, initialize() does NOT register a cron job."""
    hermes_home = tmp_path / "h3"
    hermes_home.mkdir()
    _install_provider_layout(hermes_home, layout)
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


@pytest.mark.parametrize("layout", ["flat", "development"])
def test_shutdown_preserves_profile_owned_cron_job(tmp_path, monkeypatch, layout):
    """Ordinary session shutdown leaves scheduled profile maintenance intact."""
    hermes_home = tmp_path / "h4"
    hermes_home.mkdir()
    _install_provider_layout(hermes_home, layout)
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


@pytest.mark.parametrize("layout", ["flat", "development"])
def test_cron_script_is_installed(tmp_path, monkeypatch, layout):
    """initialize() writes the cron script to $HERMES_HOME/scripts/."""
    hermes_home = tmp_path / "h5"
    hermes_home.mkdir()
    _install_provider_layout(hermes_home, layout)
    cfg = hermes_home / "cashew.json"
    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")
    cfg.write_text(json.dumps(_make_config(hermes_home)))

    (hermes_home / "cashew").mkdir(parents=True)
    conn = sqlite3.connect(str(hermes_home / "cashew" / "brain.db"))
    _ensure_schema(conn)
    conn.close()

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
    assert "_hermes_cashew_cron_impl" in content
    assert "sleep_adapter" in content

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

    # Verify the generated entry point resolves the configured auxiliary model
    # and passes that callable to upstream's cycle boundary.  Keep this as an
    # AST contract so harmless formatting or local variable names do not break
    # the test while a hard-coded ``None`` still fails it.
    tree = ast.parse(script_source)
    resolve_calls = []
    cycle_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name == "resolve_model_fn":
                resolve_calls.append(node)
            if function_name == "run_sleep_cycle":
                cycle_calls.append(node)
    assert resolve_calls
    assert cycle_calls
    model_fn_arguments = [
        keyword.value
        for call in cycle_calls
        for keyword in call.keywords
        if keyword.arg == "model_fn"
    ]
    assert model_fn_arguments
    assert all(
        not (isinstance(value, ast.Constant) and value.value is None)
        for value in model_fn_arguments
    )
