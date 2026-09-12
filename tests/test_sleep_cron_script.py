"""Subprocess contracts for the standalone sleep-cycle cron script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.memory.cashew import sleep_cron_script


def test_cron_process_waits_for_owned_sleep_phases(tmp_path: Path) -> None:
    """The cron process must not delegate work to a disposable daemon thread."""
    hermes_home = tmp_path / "hermes-home"
    agent_root = hermes_home / "hermes-agent"
    package = agent_root / "plugins" / "memory" / "cashew"
    scripts = hermes_home / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir(parents=True)
    for init_file in (
        agent_root / "plugins" / "__init__.py",
        agent_root / "plugins" / "memory" / "__init__.py",
        package / "__init__.py",
    ):
        init_file.write_text("")

    marker = hermes_home / "phase-complete"
    (hermes_home / "cashew.json").write_text(
        json.dumps({"cashew_db_path": "cashew/brain.db"})
    )
    (package / "config.py").write_text(
        "from pathlib import Path\n"
        "class Config:\n"
        "    cashew_db_path = 'cashew/brain.db'\n"
        "    sleep_max_nodes = 2000\n"
        "    embedding_model = 'thenlper/gte-large'\n"
        "    embedding_device = 'cpu'\n"
        "def load_config(home):\n"
        "    return Config()\n"
        "def resolve_db_path(home, raw):\n"
        "    path = Path(home) / raw\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    return path\n"
        "def resolve_model_fn(*, hermes_home, config):\n"
        "    return lambda prompt: 'dream'\n"
    )
    (package / "sleep_refactor.py").write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "def run_sleep_cycle(**kwargs):\n"
        "    assert kwargs['background_dream'] is False\n"
        "    time.sleep(0.15)\n"
        "    Path(os.environ['PHASE_MARKER']).write_text('complete')\n"
        "    return {'dream_pending': False, 'dream_generation': 'ran'}\n"
    )

    source = (
        Path(__file__).parents[1]
        / "plugins"
        / "memory"
        / "cashew"
        / "sleep_cron_script.py"
    )
    script = scripts / "cashew-sleep-cycle.py"
    script.write_text(source.read_text())

    env = os.environ.copy()
    env.update({"HERMES_HOME": str(hermes_home), "PHASE_MARKER": str(marker)})
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert marker.read_text() == "complete"
    assert json.loads(completed.stdout) == {
        "dream_pending": False,
        "dream_generation": "ran",
    }


def test_main_uses_profile_config_and_prints_cycle_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(
        json.dumps(
            {
                "cashew_db_path": "data/brain.db",
                "sleep_max_nodes": 321,
                "embedding_model": "example/model",
                "embedding_device": "mps",
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    calls: list[dict] = []
    model_fn = object()

    import plugins.memory.cashew.config as config_module
    import plugins.memory.cashew.sleep_refactor as refactor_module

    monkeypatch.setattr(
        config_module,
        "resolve_model_fn",
        lambda *, hermes_home, config: model_fn,
    )

    def run_sleep_cycle(**kwargs):
        calls.append(kwargs)
        return {"processed": 4, "dream_pending": False}

    monkeypatch.setattr(refactor_module, "run_sleep_cycle", run_sleep_cycle)

    sleep_cron_script.main()

    assert json.loads(capsys.readouterr().out) == {
        "processed": 4,
        "dream_pending": False,
    }
    assert calls == [
        {
            "db_path": str(hermes_home / "data" / "brain.db"),
            "limit": 321,
            "model_fn": model_fn,
            "background_dream": False,
            "embedding_model": "example/model",
            "embedding_device": "mps",
        }
    ]


def test_main_rejects_invalid_shared_config_before_running_sleep_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "cashew.json").write_text(json.dumps({"sleep_max_nodes": -1}))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(ValueError, match="sleep_max_nodes must be an integer"):
        sleep_cron_script.main()


def test_helpers_require_home_and_default_missing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERMES_HOME", raising=False)
    with pytest.raises(RuntimeError, match="HERMES_HOME is not set"):
        sleep_cron_script._find_hermes_home()

    assert sleep_cron_script._resolve_db_path(tmp_path, "cashew/brain.db") == str(
        tmp_path / "cashew" / "brain.db"
    )
