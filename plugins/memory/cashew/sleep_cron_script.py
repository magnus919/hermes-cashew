"""Cron script template for Cashew sleep cycle.

Installed at ``$HERMES_HOME/scripts/cashew-sleep-cycle.py`` during provider
initialize(). Runs as a ``no_agent=True`` cron job — the Hermes scheduler
executes this script on a schedule with zero LLM overhead per tick.

Reads ``cashew.json`` at runtime. Registration embeds the selected profile
installation identity, so reinitialize the provider after moving or
reinstalling it to refresh the generated script.
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

_INSTALLATION_MARKER = None


def _find_hermes_home() -> Path:
    """Locate the Hermes home directory from the environment.

    Returns:
        Path to the Hermes home directory.

    Raises:
        RuntimeError: If ``$HERMES_HOME`` is not set. This should never
            occur when running under the Hermes cron scheduler — the gateway
            always passes ``HERMES_HOME`` to subprocesses.
    """
    env_val = os.environ.get("HERMES_HOME")
    if env_val:
        return Path(env_val)
    raise RuntimeError(
        "$HERMES_HOME is not set. This script runs under the Hermes cron "
        "scheduler which always provides HERMES_HOME. For manual debugging, "
        "set the environment variable: HERMES_HOME=~/.hermes python3 ..."
    )


def _read_config(hermes_home: Path) -> dict:
    """Read cashew.json, returning {} if absent."""
    cfg_path = hermes_home / "cashew.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {}


def _load_profile_modules(hermes_home: Path):
    """Load cron dependencies from the installation that registered this script."""
    marker = _INSTALLATION_MARKER
    if not isinstance(marker, dict):
        raise RuntimeError(
            "Cashew cron script has no installation marker; reinitialize Cashew "
            "to refresh the generated script."
        )
    kind = marker.get("kind")
    anchors = {
        "flat": "plugins/cashew",
        "development": "hermes-agent/plugins/memory/cashew",
    }
    anchor = marker.get("anchor")
    expected = marker.get("implementation")
    if kind not in anchors or anchor != anchors[kind] or not isinstance(expected, str):
        raise RuntimeError(
            "Cashew cron installation marker is malformed; reinitialize Cashew "
            "to refresh the generated script."
        )
    anchor_path = hermes_home.joinpath(*anchors[kind].split("/"))
    implementation = (
        anchor_path / "plugins" / "memory" / "cashew" if kind == "flat" else anchor_path
    )
    if implementation.resolve() != Path(expected).resolve():
        raise RuntimeError(
            "Cashew installation no longer matches this cron registration; "
            "reinitialize Cashew to refresh the generated script."
        )
    if (
        not (implementation / "config.py").is_file()
        or not (implementation / "sleep_refactor.py").is_file()
    ):
        raise RuntimeError(
            "Cashew installation is incomplete for this cron job; reinstall or "
            "reinitialize Cashew."
        )
    package_name = "_hermes_cashew_cron_impl"
    package = types.ModuleType(package_name)
    package.__path__ = [str(implementation)]
    sys.modules[package_name] = package
    config_module = importlib.import_module(f"{package_name}.config")
    sleep_module = importlib.import_module(f"{package_name}.sleep_refactor")
    return config_module, sleep_module


def _resolve_db_path(hermes_home: Path, config: dict, config_module=None) -> str:
    """Resolve the DB path through the provider's profile-isolation guard."""
    raw = config.get("cashew_db_path") or "cashew/brain.db"
    if config_module is None:
        # Retain the direct helper contract used by in-process callers. The
        # generated cron entry point always supplies the profile-pinned module.
        from plugins.memory.cashew.config import resolve_db_path

        return str(resolve_db_path(hermes_home, raw))
    return str(config_module.resolve_db_path(hermes_home, raw))


def main() -> None:
    """Discover config, import sleep_refactor, run one cycle, print JSON."""
    hermes_home = _find_hermes_home()
    config = _read_config(hermes_home)
    limit = config.get("sleep_max_nodes", 2000)
    embedding_model = config.get("embedding_model", "thenlper/gte-large")
    embedding_device = config.get("embedding_device", "cpu")

    config_module, sleep_module = _load_profile_modules(hermes_home)
    db_path = _resolve_db_path(hermes_home, config, config_module)

    # Resolve the LLM callable from auxiliary config for dream generation.
    model_fn = config_module.resolve_model_fn(hermes_home=hermes_home)

    result = sleep_module.run_sleep_cycle(
        db_path=db_path,
        limit=limit,
        model_fn=model_fn,
        # This process owns the scheduled cycle and exits immediately after
        # printing the result. Keep dream generation and orphan embedding
        # synchronous so they complete before interpreter shutdown.
        background_dream=False,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
