#!/usr/bin/env python3
"""Run the cron lifecycle contract against a pinned Hermes source checkout.

The provider is intentionally not coupled to Hermes as a production dependency.
This harness supplies only the import surface needed by the real ``cron.jobs``
module, while keeping that module and ``agent`` implementation from the pinned
Hermes checkout under test.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _make_shim(hermes_root: Path, shim: Path) -> None:
    _link(hermes_root / "cron" / "jobs.py", shim / "cron" / "jobs.py")
    _link(hermes_root / "cron" / "env_settings.py", shim / "cron" / "env_settings.py")
    _link(hermes_root / "hermes_constants.py", shim / "hermes_constants.py")
    _link(hermes_root / "hermes_time.py", shim / "hermes_time.py")
    _link(hermes_root / "utils.py", shim / "utils.py")
    _link(hermes_root / "agent", shim / "agent")
    (shim / "cron" / "__init__.py").write_text(
        "from .jobs import JOBS_FILE, create_job, get_job, list_jobs, remove_job, update_job\n",
        encoding="utf-8",
    )
    # cron.jobs delegates owner-only permissions to this helper. Importing the
    # full Hermes CLI would add unrelated CLI dependencies to this contract
    # lane, so keep the security boundary deterministic and minimal.
    (shim / "hermes_cli").mkdir(parents=True)
    (shim / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (shim / "hermes_cli" / "config.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def _secure_dir(path: Path) -> None:\n"
        "    path.mkdir(parents=True, exist_ok=True)\n"
        "    path.chmod(0o700)\n"
        "\n"
        "def _secure_file(path: Path) -> None:\n"
        "    path.chmod(0o600)\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-root", type=Path, required=True)
    args = parser.parse_args()
    hermes_root = args.hermes_root.resolve()
    if not (hermes_root / "cron" / "jobs.py").is_file():
        parser.error(f"not an Hermes source checkout: {hermes_root}")

    repo_root = Path(__file__).resolve().parents[1]
    shim = Path(tempfile.mkdtemp(prefix="hermes-cashew-host-shim-"))
    try:
        _make_shim(hermes_root, shim)
        env = os.environ.copy()
        env["HERMES_CASHEW_REAL_HERMES"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root), str(shim), env.get("PYTHONPATH", "")]
        )
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rs",
                "tests/test_sleep_cron_lifecycle.py",
            ],
            cwd=repo_root,
            env=env,
            check=False,
        ).returncode
    finally:
        shutil.rmtree(shim, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
