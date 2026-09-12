"""Smoke test for hermes-cashew.

Exercises the full CashewMemoryProvider lifecycle against a temporary profile so
end users can verify their install is working without any external infrastructure.

Usage:
    python -m plugins.memory.cashew.verify

Exit codes:
    0  — all checks passed
    1  — any check failed

Error convention: all failure messages are prefixed with "[cashew]" so CI can
grep for ^\\[cashew\\] to distinguish cashew errors from Python/Hermes errors
(CI-06 smoke test requirement).
"""

from __future__ import annotations

import json
import logging
import os as _os
import pathlib
import shutil
import sys
import tempfile
from typing import Any

_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logger = logging.getLogger(__name__)
_PROVIDER_NAME = "cashew"


def _error(msg: str) -> None:
    """Print a cashew-prefixed error and exit with code 1."""
    print(f"[cashew] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_health_status(
    provider: Any, hermes_home: pathlib.Path, tmp_dir: str
) -> None:
    """Validate the diagnostic snapshot without exposing profile details."""
    status = provider.health_status()
    if status.get("state") not in {"ready", "degraded"}:
        _error(
            "health_status() did not report an operational state: "
            f"{status.get('state')}"
        )
    if status.get("generation") != 1:
        _error(
            "health_status() reported unexpected generation: "
            f"{status.get('generation')}"
        )
    runtime = status.get("runtime", {})
    if not runtime.get("config_loaded") or not runtime.get("retriever_ready"):
        _error("health_status() did not report initialized runtime")
    if not status.get("work", {}).get("reconciled"):
        _error("health_status() reported unreconciled work")
    encoded_status = json.dumps(status, sort_keys=True, allow_nan=False)
    if str(hermes_home) in encoded_status or tmp_dir in encoded_status:
        _error("health_status() leaked a temporary profile path")


def _check_tool_health(provider: Any) -> None:
    """Confirm synchronous tool work appears in diagnostics."""
    status = provider.health_status()
    if status.get("tools", {}).get("extract_completed") != 1:
        _error("health_status() did not account for cashew_extract")


def _check_stopped_health(provider: Any) -> None:
    """Confirm shutdown publishes the terminal diagnostic state."""
    status = provider.health_status()
    if status.get("state") != "stopped":
        _error(f"health_status() did not report stopped state: {status.get('state')}")


def main() -> int:  # noqa: C901 - verifier keeps its user-facing failure prefixes together
    """Run the smoke test.

    Creates a temp hermes_home, initializes the provider, runs through
    initialize → cashew_query → cashew_extract → shutdown, then tears down.

    Returns 0 on success, 1 on any failure.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    tmp_dir = tempfile.mkdtemp(prefix="cashew-verify-")
    hermes_home = pathlib.Path(tmp_dir)
    try:
        from plugins.memory.cashew import CashewMemoryProvider

        provider = CashewMemoryProvider()

        from plugins.memory.cashew.config import save_config

        save_config(
            {
                "cashew_db_path": "cashew/brain.db",
                "recall_k": 5,
                "sync_queue_timeout": 30,
            },
            hermes_home=str(hermes_home),
        )

        try:
            provider.initialize(
                session_id="verify-session", hermes_home=str(hermes_home)
            )
        except Exception as exc:
            _error(f"initialize raised {type(exc).__name__}: {exc}")

        if not provider.is_available():
            _error("is_available() returned False after successful initialize")

        try:
            _check_health_status(provider, hermes_home, tmp_dir)
        except SystemExit:
            raise
        except Exception as exc:
            _error(f"health_status raised {type(exc).__name__}: {exc}")

        try:
            result = provider.prefetch("test query")
            if not isinstance(result, str):
                _error(f"prefetch returned {type(result).__name__}, expected str")
        except Exception as exc:
            _error(f"prefetch raised {type(exc).__name__}: {exc}")

        try:
            raw = provider.handle_tool_call(
                "cashew_extract",
                {"user_content": "", "assistant_content": ""},
            )
            envelope = json.loads(raw)
            if not envelope.get("ok"):
                _error(f"cashew_extract envelope not ok: {raw}")
        except Exception as exc:
            _error(f"cashew_extract raised {type(exc).__name__}: {exc}")

        try:
            schemas = provider.get_tool_schemas()
            names = {s["name"] for s in schemas}
            if "cashew_query" not in names:
                _error(f"cashew_query not in tool schemas: {names}")
        except Exception as exc:
            _error(f"get_tool_schemas raised {type(exc).__name__}: {exc}")

        try:
            _check_tool_health(provider)
        except SystemExit:
            raise
        except Exception as exc:
            _error(f"health_status after tools raised {type(exc).__name__}: {exc}")

        try:
            provider.shutdown()
        except Exception as exc:
            _error(f"shutdown raised {type(exc).__name__}: {exc}")

        try:
            _check_stopped_health(provider)
        except SystemExit:
            raise
        except Exception as exc:
            _error(f"health_status after shutdown raised {type(exc).__name__}: {exc}")

        print("[cashew] verify: all checks passed")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _error(f"unexpected {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
