"""Subprocess contracts for the generated standalone sleep-cycle cron script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

import plugins.memory.cashew as provider_module
from plugins.memory.cashew import CashewMemoryProvider, sleep_cron_script
from plugins.memory.cashew.config import CashewConfig


def _install_fake_cron(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the small Hermes cron API surface used during registration."""
    cron_package = types.ModuleType("cron")
    cron_package.__path__ = []  # type: ignore[attr-defined]
    cron_jobs = types.ModuleType("cron.jobs")
    cron_jobs.list_jobs = lambda: []
    cron_jobs.remove_job = lambda _job_id: None
    cron_jobs.create_job = lambda **_kwargs: {"id": "cashew-sleep-test"}
    monkeypatch.setitem(sys.modules, "cron", cron_package)
    monkeypatch.setitem(sys.modules, "cron.jobs", cron_jobs)


def _write_installation(implementation: Path, identity: str) -> None:
    """Create a minimal loadable provider implementation for a subprocess."""
    implementation.mkdir(parents=True)
    source_template = Path(provider_module.__file__).parent / "sleep_cron_script.py"
    (implementation / "sleep_cron_script.py").write_text(source_template.read_text())
    source_root = Path(provider_module.__file__).parent
    # The generated entry point must exercise the same admission and journal
    # modules as a profile install, rather than a test-only replacement.
    (implementation / "admission.py").write_text(
        (source_root / "admission.py").read_text()
    )
    (implementation / "locking.py").write_text((source_root / "locking.py").read_text())
    (implementation / "config.py").write_text(
        "import json\n"
        "import sqlite3\n"
        "from pathlib import Path\n"
        "class Config:\n"
        "    def __init__(self, values):\n"
        "        self.cashew_db_path = values.get('cashew_db_path', 'cashew/brain.db')\n"
        "        self.sleep_max_nodes = values.get('sleep_max_nodes', 2000)\n"
        "        self.embedding_model = values.get('embedding_model', 'thenlper/gte-large')\n"
        "        self.embedding_device = values.get('embedding_device', 'cpu')\n"
        "        self.sleep_cycles = values.get('sleep_cycles', True)\n"
        "        self.sleep_schedule = values.get('sleep_schedule', 'every 1h')\n"
        "def load_effective_config_snapshot(home, values):\n"
        "    config = Config(values)\n"
        "    db = Path(home) / config.cashew_db_path\n"
        "    db.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with sqlite3.connect(db) as conn:\n"
        "        conn.execute('CREATE TABLE IF NOT EXISTS hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')\n"
        "        conn.executemany('INSERT OR REPLACE INTO hermes_provider_meta VALUES (?, ?)', [('embedding_model', config.embedding_model), ('embedding_dim', '384'), ('vec_dim', '384'), ('maintenance_epoch', '1')])\n"
        "    return config\n"
        "def resolve_db_path(home, raw):\n"
        "    return Path(home) / raw\n"
        "def resolve_model_fn(*, hermes_home, config):\n"
        "    return 'resolved-memory-model'\n"
    )
    (implementation / "companion.py").write_text(f"IDENTITY = {identity!r}\n")
    (implementation / "embedding.py").write_text("MODEL = 'test-embedding'\n")
    (implementation / "embedding_process.py").write_text(
        "import os\n"
        "def _event(value):\n"
        "    path = os.environ.get('CRON_EVENTS')\n"
        "    if path:\n"
        "        with open(path, 'a') as handle: handle.write(value + '\\n')\n"
        "class EmbeddingSupervisor:\n"
        "    def __init__(self, **kwargs): self.kwargs = kwargs; self.dimension = kwargs.get('dimension', 0); self.generation = 'fake-generation'\n"
        "    def start(self): self.dimension = 384; _event('start'); return 384\n"
        "    def close(self): _event('close')\n"
    )
    (implementation / "embedding_worker.py").write_text("# test worker marker\n")
    (implementation / "log_filter.py").write_text(
        (Path(provider_module.__file__).parent / "log_filter.py").read_text()
    )
    (implementation / "sleep_refactor.py").write_text(
        "import logging\n"
        "import os\n"
        "from pathlib import Path\n"
        "from .companion import IDENTITY\n"
        "from . import embedding\n"
        "def _generate_dream():\n"
        "    raise RuntimeError('CONTENT-CANARY /private/profile SECRET-CANARY')\n"
        "def run_sleep_cycle(**kwargs):\n"
        "    path = os.environ.get('CRON_EVENTS')\n"
        "    if path:\n"
        "        with open(path, 'a') as handle: handle.write('run\\n')\n"
        "    if os.environ.get('CRON_RAISE'):\n"
        "        raise RuntimeError('synthetic cron failure')\n"
        "    if os.environ.get('CRON_GENERATE_DREAM_FAILURE'):\n"
        "        root = logging.getLogger()\n"
        "        core = logging.getLogger('core')\n"
        "        root.addHandler(logging.StreamHandler())\n"
        "        core.addHandler(logging.StreamHandler())\n"
        "        try:\n"
        "            _generate_dream()\n"
        "        except RuntimeError as error:\n"
        "            root.warning('dream failed: %s', error, exc_info=True)\n"
        "            core.error('dream failed: %s', error, exc_info=True)\n"
        "    assert kwargs['background_dream'] is False\n"
        "    Path(os.environ['PHASE_MARKER']).write_text(IDENTITY)\n"
        "    return {\n"
        "        'installation': IDENTITY,\n"
        "        'db_path': kwargs['db_path'],\n"
        "        'limit': kwargs['limit'],\n"
        "        'model_fn': kwargs['model_fn'],\n"
        "        'background_dream': kwargs['background_dream'],\n"
        "        'embedding_model': kwargs['embedding_model'],\n"
        "        'embedding_device': kwargs['embedding_device'],\n"
        "        'embedding_client': type(kwargs['embedding_client']).__name__,\n"
        "    }\n"
    )


def _generate_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    **config_changes: object,
) -> tuple[Path, Path, Path]:
    """Drive registration itself so tests execute the emitted script."""
    _install_fake_cron(monkeypatch)
    hermes_home = tmp_path / "profile"
    if kind == "flat":
        implementation = (
            hermes_home / "plugins" / "cashew" / "plugins" / "memory" / "cashew"
        )
        _write_installation(implementation, "flat")
    else:
        external = tmp_path / "external-checkout" / "plugins" / "memory" / "cashew"
        _write_installation(external, "development")
        implementation = hermes_home / "hermes-agent" / "plugins" / "memory" / "cashew"
        implementation.parent.mkdir(parents=True)
        implementation.symlink_to(external, target_is_directory=True)

    monkeypatch.setattr(
        provider_module, "__file__", str(implementation / "__init__.py")
    )
    provider = CashewMemoryProvider()
    provider._hermes_home = hermes_home
    provider._config = replace(
        CashewConfig(), sleep_schedule="every 1h", **config_changes
    )
    provider._embedding_identity_ready = True
    provider._register_sleep_cron()

    return (
        hermes_home,
        hermes_home / "scripts" / "cashew-sleep-cycle.py",
        implementation,
    )


@pytest.mark.parametrize("kind", ["flat", "development"])
def test_generated_cron_script_runs_from_registered_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    custom_db = "owned/custom-cashew.db"
    hermes_home, script, _implementation = _generate_script(
        tmp_path,
        monkeypatch,
        kind,
        cashew_db_path=custom_db,
        sleep_max_nodes=7,
        embedding_model="test/embedding-model",
        embedding_device="mps",
    )
    marker = tmp_path / "phase-complete"
    events = tmp_path / "cron-events"
    # This JSON deliberately disagrees with registration. The generated
    # process must retain the provider's validated effective snapshot.
    (hermes_home / "cashew.json").write_text(
        json.dumps(
            {
                "cashew_db_path": "ignored/by/snapshot.db",
                "sleep_max_nodes": 1,
                "embedding_model": "ignored/model",
                "embedding_device": "ignored-device",
            }
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={
            "HERMES_HOME": str(hermes_home),
            "PATH": os.defpath,
            "PYTHONPATH": "",
            "PHASE_MARKER": str(marker),
            "CRON_EVENTS": str(events),
        },
    )

    result = json.loads(completed.stdout)
    assert marker.read_text() == kind
    assert result["installation"] == kind
    assert result["db_path"] == str(hermes_home / custom_db)
    assert result["limit"] == 7
    assert result["model_fn"] == "resolved-memory-model"
    assert result["background_dream"] is False
    assert result["embedding_model"] == "test/embedding-model"
    assert result["embedding_device"] == "mps"
    assert result["embedding_client"] == "EmbeddingSupervisor"
    assert events.read_text().splitlines() == ["start", "run", "close"]


def test_generated_cron_script_closes_owned_child_after_sleep_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home, script, _implementation = _generate_script(
        tmp_path, monkeypatch, "flat"
    )
    events = tmp_path / "cron-events"
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={
            "HERMES_HOME": str(hermes_home),
            "PATH": os.defpath,
            "PYTHONPATH": "",
            "CRON_EVENTS": str(events),
            "CRON_RAISE": "1",
        },
    )

    assert completed.returncode != 0
    assert "synthetic cron failure" in completed.stderr
    assert events.read_text().splitlines() == ["start", "run", "close"]


@pytest.mark.parametrize("kind", ["flat", "development"])
def test_generated_script_uses_registered_anchor_without_stale_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    hermes_home, script, _implementation = _generate_script(tmp_path, monkeypatch, kind)
    decoy = tmp_path / "external-decoy" / "plugins" / "memory" / "cashew"
    _write_installation(decoy, "decoy")
    if kind == "flat":
        other_anchor = hermes_home / "hermes-agent" / "plugins" / "memory" / "cashew"
        other_anchor.parent.mkdir(parents=True)
        other_anchor.symlink_to(decoy, target_is_directory=True)
    else:
        other_anchor = (
            hermes_home / "plugins" / "cashew" / "plugins" / "memory" / "cashew"
        )
        _write_installation(other_anchor, "decoy")
    marker = tmp_path / "phase-complete"

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={
            "HERMES_HOME": str(hermes_home),
            "PATH": os.defpath,
            "PYTHONPATH": "",
            "PHASE_MARKER": str(marker),
        },
    )

    assert marker.read_text() == kind
    assert json.loads(completed.stdout)["installation"] == kind


def test_generated_script_rejects_moved_or_reinstalled_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home, script, implementation = _generate_script(
        tmp_path, monkeypatch, "development"
    )
    implementation.unlink()
    implementation.mkdir()

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={"HERMES_HOME": str(hermes_home), "PATH": os.defpath, "PYTHONPATH": ""},
    )

    assert completed.returncode == 1
    assert "reinitialize Cashew" in completed.stderr


def test_copied_generated_script_rejects_different_hermes_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hermes_home, script, _implementation = _generate_script(
        tmp_path, monkeypatch, "flat"
    )
    other_home = tmp_path / "other-profile"
    _write_installation(
        other_home / "plugins" / "cashew" / "plugins" / "memory" / "cashew",
        "other-profile",
    )
    copied_script = other_home / "scripts" / "cashew-sleep-cycle.py"
    copied_script.parent.mkdir(parents=True)
    copied_script.write_text(script.read_text())

    completed = subprocess.run(
        [sys.executable, str(copied_script)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={"HERMES_HOME": str(other_home), "PATH": os.defpath, "PYTHONPATH": ""},
    )

    assert completed.returncode == 1
    assert "reinitialize Cashew" in completed.stderr


@pytest.mark.parametrize("missing_file", ["config.py", "sleep_refactor.py"])
def test_generated_script_rejects_incomplete_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_file: str
) -> None:
    hermes_home, script, implementation = _generate_script(
        tmp_path, monkeypatch, "flat"
    )
    (implementation / missing_file).unlink()

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={"HERMES_HOME": str(hermes_home), "PATH": os.defpath, "PYTHONPATH": ""},
    )

    assert completed.returncode == 1
    assert "installation is incomplete" in completed.stderr


def test_generated_script_reports_missing_imported_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home, script, implementation = _generate_script(
        tmp_path, monkeypatch, "flat"
    )
    (implementation / "embedding.py").unlink()

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={"HERMES_HOME": str(hermes_home), "PATH": os.defpath, "PYTHONPATH": ""},
    )

    assert completed.returncode == 1
    assert "could not load cron dependencies" in completed.stderr
    assert "reinstall or reinitialize Cashew" in completed.stderr


def test_generated_script_rejects_malformed_installation_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home, script, _implementation = _generate_script(
        tmp_path, monkeypatch, "flat"
    )
    script.write_text(
        script.read_text().replace("'kind': 'flat'", "'kind': 'unexpected-layout'", 1)
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={"HERMES_HOME": str(hermes_home), "PATH": os.defpath, "PYTHONPATH": ""},
    )

    assert completed.returncode == 1
    assert "installation marker is malformed" in completed.stderr


def test_main_rejects_invalid_embedded_effective_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cron entry point validates its registered effective snapshot."""
    hermes_home = tmp_path / "profile"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import plugins.memory.cashew.config as profile_config

    monkeypatch.setattr(
        sleep_cron_script,
        "_load_profile_modules",
        lambda _home: (
            profile_config,
            types.SimpleNamespace(__package__="plugins.memory.cashew"),
            {"config": {"sleep_max_nodes": -1}},
        ),
    )
    with pytest.raises(ValueError, match="configuration snapshot is malformed"):
        sleep_cron_script.main()


def test_unmarked_copied_script_requests_reinitialization(tmp_path: Path) -> None:
    script = tmp_path / "cashew-sleep-cycle.py"
    script.write_text(Path(sleep_cron_script.__file__).read_text())

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={
            "HERMES_HOME": str(tmp_path / "profile"),
            "PATH": os.defpath,
            "PYTHONPATH": "",
        },
    )

    assert completed.returncode == 1
    assert "reinitialize Cashew" in completed.stderr


def test_generated_cron_scrubs_caught_dream_failure_at_dynamic_root_and_core_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated script owns and releases its dynamic logging boundary."""
    hermes_home, script, _implementation = _generate_script(
        tmp_path, monkeypatch, "flat"
    )
    marker = tmp_path / "phase-complete"
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=tmp_path,
        env={
            "HERMES_HOME": str(hermes_home),
            "PATH": os.defpath,
            "PYTHONPATH": "",
            "PHASE_MARKER": str(marker),
            "CRON_GENERATE_DREAM_FAILURE": "1",
        },
    )

    emitted = completed.stderr
    for canary in ("CONTENT-CANARY", "SECRET-CANARY", "/private/profile"):
        assert canary not in emitted
    assert emitted.count("cashew local log event error=RuntimeError") >= 2
    assert marker.read_text() == "flat"
