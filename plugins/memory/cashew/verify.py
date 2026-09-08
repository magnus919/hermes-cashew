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

_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logger = logging.getLogger(__name__)
_PROVIDER_NAME = "cashew"


def _error(msg: str) -> None:
    """Print a cashew-prefixed error and exit with code 1."""
    print(f"[cashew] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    """Run the smoke test.

    Creates a temp hermes_home, initializes the provider, runs through
    initialize → cashew_query → cashew_extract → shutdown, then tears down.

    Returns 0 on success, 1 on any failure.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    tmp_dir = tempfile.mkdtemp(prefix="cashew-verify-")
    hermes_home = pathlib.Path(tmp_dir)
    previous_embedding_cache = _os.environ.get("CASHEW_EMBD_CACHE")
    _os.environ["CASHEW_EMBD_CACHE"] = str(hermes_home / "embedding-cache.db")

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
            provider.shutdown()
        except Exception as exc:
            _error(f"shutdown raised {type(exc).__name__}: {exc}")

        print("[cashew] verify: all checks passed")
        return 0

    except SystemExit:
        raise
    except Exception as exc:
        _error(f"unexpected {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if previous_embedding_cache is None:
            _os.environ.pop("CASHEW_EMBD_CACHE", None)
        else:
            _os.environ["CASHEW_EMBD_CACHE"] = previous_embedding_cache


if __name__ == "__main__":
    sys.exit(main())
