"""Cron script template for Cashew sleep cycle.

Installed at ``$HERMES_HOME/scripts/cashew-sleep-cycle.py`` during provider
initialize(). Runs as a ``no_agent=True`` cron job — the Hermes scheduler
executes this script on a schedule with zero LLM overhead per tick.

Reads ``cashew.json`` at runtime so it survives path changes without
regenerating the script.
"""

# ruff: noqa: E402 (import after sys.path setup is intentional)

import json
import os
import sys
from pathlib import Path


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


def _resolve_db_path(hermes_home: Path, config: dict) -> str:
    """Resolve db path: absolute stays, relative rooted at hermes_home."""
    default_db = str(hermes_home / "cashew" / "brain.db")
    raw = config.get("cashew_db_path", "")
    if not raw:
        return default_db
    if Path(raw).is_absolute():
        return raw
    return str(hermes_home / raw)


def main() -> None:
    """Discover config, import sleep_refactor, run one cycle, print JSON."""
    hermes_home = _find_hermes_home()
    config = _read_config(hermes_home)
    db_path = _resolve_db_path(hermes_home, config)
    limit = config.get("sleep_max_nodes", 2000)
    embedding_model = config.get("embedding_model", "thenlper/gte-large")

    # Ensure the Hermes agent root is on sys.path so imports like
    # plugins.memory.cashew.sleep_refactor resolve correctly.
    # When run as a cron script, sys.executable is the Hermes venv Python,
    # which may or may not have the Hermes root on sys.path.
    agent_root_candidates = [
        hermes_home / "hermes-agent",
        Path(sys.executable).parent.parent / "hermes-agent",
        Path(sys.executable).parent.parent.parent / "hermes-agent",
    ]
    for candidate in agent_root_candidates:
        if (candidate / "plugins").is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            break

    try:
        from plugins.memory.cashew.sleep_refactor import run_sleep_cycle
    except ImportError:
        # Fallback: try the installed package (PyPI install path)
        from cashew_sleep_refactor import run_sleep_cycle  # type: ignore[import-not-found]

    result = run_sleep_cycle(
        db_path=db_path,
        limit=limit,
        model_fn=None,
        background_dream=True,
        embedding_model=embedding_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
