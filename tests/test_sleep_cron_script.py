"""Subprocess contracts for the standalone sleep-cycle cron script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


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
        "def resolve_db_path(home, raw):\n"
        "    path = Path(home) / raw\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    return path\n"
        "def resolve_model_fn(*, hermes_home):\n"
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
