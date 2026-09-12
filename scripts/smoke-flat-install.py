#!/usr/bin/env python3
"""Exercise the flat Hermes loader and cron registration from an sdist tree."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
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
    cron_jobs.list_jobs = lambda: []
    cron_jobs.remove_job = lambda _job_id: None

    def create_job(**kwargs: Any) -> dict[str, str]:
        jobs.append(kwargs)
        return {"id": "clean-flat-cron"}

    cron_jobs.create_job = create_job
    sys.modules["cron"] = cron
    sys.modules["cron.jobs"] = cron_jobs
    return jobs


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
    print(f"flat loader and cron registration verified from {flat_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
