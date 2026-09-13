#!/usr/bin/env python3
"""Exercise the flat Hermes loader and cron registration from an sdist tree."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def _load_flat_entrypoint(root: Path) -> types.ModuleType:
    parent_name = "_hermes_user_memory"
    parent = types.ModuleType(parent_name)
    parent.__path__ = []  # type: ignore[attr-defined]
    sys.modules[parent_name] = parent
    name = f"{parent_name}.cashew"
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load flat Cashew entry point")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_cron_stub() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    cron = types.ModuleType("cron")
    cron.__path__ = []  # type: ignore[attr-defined]
    cron_jobs = types.ModuleType("cron.jobs")
    cron_jobs.list_jobs = lambda *, include_disabled=False: []
    cron_jobs.remove_job = lambda _job_id: None
    cron_jobs.parse_schedule = lambda schedule: schedule
    cron_jobs.update_job = lambda _job_id, _updates: None
    cron_jobs.use_cron_store = lambda _home: nullcontext()

    def create_job(**kwargs: Any) -> dict[str, str]:
        jobs.append(kwargs)
        return {"id": "clean-flat-cron"}

    cron_jobs.create_job = create_job
    sys.modules["cron"] = cron
    sys.modules["cron.jobs"] = cron_jobs
    return jobs


def _install_bounded_runtime_fakes(implementation: Path, marker: Path) -> None:
    """Replace only model/sleep dependencies so the generated script can run offline."""
    (implementation / "embedding_process.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "class EmbeddingSupervisor:\n"
        "    def __init__(self, **_kwargs):\n"
        "        self.dimension = 384\n"
        "        self.generation = 'clean-flat-smoke'\n"
        "    def start(self):\n"
        "        return None\n"
        "    def close(self):\n"
        "        Path(os.environ['CASHEW_CRON_SMOKE_MARKER']).write_text('closed')\n"
    )
    (implementation / "sleep_adapter.py").write_text(
        "from pathlib import Path\n"
        "import json\n"
        "import os\n"
        "def run_sleep_cycle(**kwargs):\n"
        "    Path(os.environ['CASHEW_CRON_SMOKE_MARKER']).write_text('ran')\n"
        "    return {'status': 'ok', 'db_path': kwargs['db_path']}\n"
    )
    marker.unlink(missing_ok=True)


def _seed_runtime_identity(hermes_home: Path) -> Path:
    db_path = hermes_home / "cashew" / "brain.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE hermes_provider_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO hermes_provider_meta VALUES (?, ?)",
            [
                ("embedding_model", "thenlper/gte-large"),
                ("embedding_dim", "384"),
                ("vec_dim", "384"),
                ("maintenance_epoch", "1"),
            ],
        )
    return db_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flat_root", type=Path)
    parser.add_argument("hermes_home", type=Path)
    args = parser.parse_args()
    flat_root = args.flat_root.resolve()
    hermes_home = args.hermes_home.resolve()
    jobs = _install_cron_stub()
    module = _load_flat_entrypoint(flat_root)
    provider = module.CashewMemoryProvider()
    config = sys.modules["_hermes_cashew_impl.config"].CashewConfig
    provider._hermes_home = hermes_home
    provider._config = config(sleep_cycles=True, sleep_schedule="every 12h")
    provider._embedding_identity_ready = True
    provider._register_sleep_cron()

    script = hermes_home / "scripts" / "cashew-sleep-cycle.py"
    if not script.is_file():
        raise RuntimeError("flat loader did not register the cron script")
    source = script.read_text()
    if (
        "'kind': 'flat'" not in source
        or str(flat_root / "plugins/memory/cashew") not in source
    ):
        raise RuntimeError("generated cron script did not pin the flat installation")
    compile(source, str(script), "exec")
    if [job.get("script") for job in jobs] != ["cashew-sleep-cycle.py"]:
        raise RuntimeError(f"unexpected cron registrations: {jobs!r}")
    runtime_marker = hermes_home / "cron-runtime-marker"
    implementation = flat_root / "plugins" / "memory" / "cashew"
    _install_bounded_runtime_fakes(implementation, runtime_marker)
    _seed_runtime_identity(hermes_home)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=hermes_home.parent,
        env={
            **os.environ,
            "HERMES_HOME": str(hermes_home),
            "PYTHONPATH": "",
            "CASHEW_CRON_SMOKE_MARKER": str(runtime_marker),
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"generated cron script failed: {completed.stderr}")
    result = json.loads(completed.stdout)
    if result.get("status") != "ok" or runtime_marker.read_text() != "closed":
        raise RuntimeError(f"generated cron runtime did not complete: {result!r}")
    print(f"flat loader and cron runtime verified from {flat_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
