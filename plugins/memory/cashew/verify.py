"""Smoke test for hermes-cashew.

Exercises the full CashewMemoryProvider lifecycle against a self-seeded temporary
database so end users can verify their install is working without any external
infrastructure.

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
import pathlib
import shutil
import sys
import tempfile

import os as _os
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

    try:
        from plugins.memory.cashew import CashewMemoryProvider

        provider = CashewMemoryProvider()

        from plugins.memory.cashew.config import save_config
        save_config(
            {
                "cashew_db_path": "cashew/brain.db",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "recall_k": 5,
                "sync_queue_timeout": 30,
            },
            hermes_home=str(hermes_home),
        )
        (hermes_home / "cashew").mkdir(exist_ok=True)
        db_path = hermes_home / "cashew" / "brain.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                confidence REAL,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                last_updated TEXT,
                mood_state TEXT,
                permanent INTEGER DEFAULT 0,
                tags TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
            )
        """)
        conn.commit()
        conn.close()

        try:
            provider.initialize(session_id="verify-session", hermes_home=str(hermes_home))
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
                {"user_content": "hello", "assistant_content": "hi there"},
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


if __name__ == "__main__":
    sys.exit(main())
