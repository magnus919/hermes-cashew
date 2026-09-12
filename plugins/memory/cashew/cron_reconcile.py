"""Profile-scoped desired-state reconciliation for Cashew's sleep cron job.

Hermes owns the scheduler.  This module only makes one Cashew job's desired
state explicit and serializes changes that target the same Hermes profile.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Iterator
from typing import Any

from .config import CashewConfig, effective_config_snapshot

CRON_JOB_NAME = "cashew-sleep-cycle"
CRON_SCRIPT_NAME = "cashew-sleep-cycle.py"
_MARKER_SENTINEL = "_INSTALLATION_MARKER = None"


def profile_identity(hermes_home: pathlib.Path) -> str:
    """Return an opaque stable identity for one Hermes profile."""
    return hashlib.sha256(str(hermes_home.resolve()).encode("utf-8")).hexdigest()[:16]


def config_identity(snapshot: dict[str, Any]) -> str:
    """Hash the full effective config without exposing it in scheduler metadata."""
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def cron_prompt(profile_id: str) -> str:
    """Scheduler-visible ownership tag; never includes the profile path."""
    return f"hermes-cashew sleep cycle [{profile_id}]"


@contextlib.contextmanager
def profile_cron_lock(hermes_home: pathlib.Path) -> Iterator[None]:
    """Serialize job/script reconciliation for one profile across processes."""
    import fcntl

    state_dir = hermes_home / "cashew"
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_dir / ".sleep-cron.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def installation_marker(
    hermes_home: pathlib.Path, implementation: pathlib.Path, config: CashewConfig
) -> dict[str, Any]:
    """Build the generated-script marker after validating a supported layout."""
    flat_anchor = hermes_home / "plugins" / "cashew"
    dev_anchor = hermes_home / "hermes-agent" / "plugins" / "memory" / "cashew"
    if (flat_anchor / "plugins" / "memory" / "cashew").resolve() == implementation:
        kind, anchor = "flat", "plugins/cashew"
    elif dev_anchor.resolve() == implementation:
        kind, anchor = "development", "hermes-agent/plugins/memory/cashew"
    else:
        raise RuntimeError(
            "Cashew must be installed at the selected HERMES_HOME flat or "
            "development anchor before its cron job can be registered"
        )
    snapshot = effective_config_snapshot(config)
    return {
        "version": 2,
        "kind": kind,
        "anchor": anchor,
        "implementation": str(implementation),
        "profile_id": profile_identity(hermes_home),
        "config_id": config_identity(snapshot),
        "config": snapshot,
    }


def render_script(template: str, marker: dict[str, Any]) -> str:
    """Render exactly one registration marker into the cron entry-point."""
    if template.count(_MARKER_SENTINEL) != 1:
        raise RuntimeError(
            "Cashew cron script template is invalid; reinstall or reinitialize "
            "Cashew before registering its cron job"
        )
    return template.replace(_MARKER_SENTINEL, f"_INSTALLATION_MARKER = {marker!r}", 1)


def stage_script(destination: pathlib.Path, content: str) -> bool:
    """Atomically replace ``destination`` using a unique same-directory stage."""
    if destination.exists() and destination.read_text(encoding="utf-8") == content:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    staged = pathlib.Path(staged_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staged, 0o755)
        os.replace(staged, destination)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return True


def read_script_marker(destination: pathlib.Path) -> dict[str, Any] | None:
    """Read the literal marker from a managed script without executing it."""
    try:
        tree = ast.parse(destination.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "_INSTALLATION_MARKER"
        ):
            try:
                marker = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                return None
            return marker if isinstance(marker, dict) else None
    return None


def owns_job(job: dict[str, Any], profile_id: str) -> bool:
    """Whether scheduler metadata proves this job belongs to this profile."""
    return (
        job.get("name") == CRON_JOB_NAME
        and job.get("script") == CRON_SCRIPT_NAME
        and job.get("prompt") == cron_prompt(profile_id)
        and job.get("no_agent") is True
        and job.get("repeat") is None
    )


def compatible_job(
    job: dict[str, Any],
    profile_id: str,
    schedule: str,
    marker: dict[str, Any],
    script: pathlib.Path,
) -> bool:
    """Whether a persisted job and managed script may be adopted unchanged."""
    return (
        owns_job(job, profile_id)
        and job.get("schedule") == schedule
        and read_script_marker(script) == marker
    )
