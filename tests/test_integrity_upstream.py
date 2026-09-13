"""Offline subprocess proof against the exact upstream PR #137 API fixture."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


_FIXTURE = Path(__file__).parent / "fixtures" / "cashew-pr137"
_PROJECT_ROOT = Path(__file__).parents[1]
_UPSTREAM_SOURCE = _FIXTURE / "core" / "integrity.py"
_EXPECTED_HEAD = "cb940f34c15460b87831748b2e702334c1c5fbd0"
_EXPECTED_SOURCE_SHA256 = (
    "80e928c4a073aadb9340095a27b65f03512393d9c1e03cfd025e9aa70d99b9ba"
)


def test_fixture_records_immutable_upstream_provenance() -> None:
    provenance = json.loads((_FIXTURE / "PROVENANCE.json").read_text())
    assert provenance["commit"] == _EXPECTED_HEAD
    assert provenance["source"] == "core/integrity.py"
    assert provenance["sha256"] == _EXPECTED_SOURCE_SHA256
    assert hashlib.sha256(_UPSTREAM_SOURCE.read_bytes()).hexdigest() == (
        _EXPECTED_SOURCE_SHA256
    )


def test_fixture_is_excluded_from_project_ruff_configuration() -> None:
    config = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text())
    assert (
        "tests/fixtures/cashew-pr137/core/integrity.py"
        in config["tool"]["ruff"]["exclude"]
    )


def test_adapter_delegates_to_exact_upstream_api_in_subprocess(tmp_path: Path) -> None:
    script = (
        r'''
import json
import sqlite3
from plugins.memory.cashew.integrity import apply_integrity_repairs, inspect_integrity

path = r"'''
        + str(tmp_path / "brain.db")
        + r'''"
conn = sqlite3.connect(path)
conn.executescript("""
CREATE TABLE thought_nodes (
    id TEXT PRIMARY KEY, content TEXT, node_type TEXT,
    decayed INTEGER DEFAULT 0, permanent INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0, last_updated TEXT
);
CREATE TABLE embeddings (
    node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT
);
CREATE TABLE derivation_edges (
    parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, timestamp TEXT
);
INSERT INTO thought_nodes VALUES ('live', 'content', 'fact', 0, 0, 0, 'now');
INSERT INTO embeddings VALUES ('live', X'0000803F000000000000000000000000', 'model-a', 'now');
INSERT INTO embeddings VALUES ('ghost', X'0000803F000000000000000000000000', 'model-a', 'now');
""")
conn.commit()
conn.execute("BEGIN")
audit = inspect_integrity(conn, expected_model="model-a", expected_dimension=4)
result = apply_integrity_repairs(
    conn=conn,
    confirm=True,
    expected_model="model-a",
    expected_dimension=4,
    actions={"remove_orphan_embeddings"},
    require_vec_parity=False,
)
payload = {
    "audit_delegate": audit.get("adapter", {}).get("delegated_to"),
    "repair_delegate": result.get("adapter", {}).get("delegated_to"),
    "status": result.get("status"),
    "removed": result.get("repairs", {}).get("orphan_embeddings_removed"),
    "remaining": result.get("remaining"),
    "in_transaction": conn.in_transaction,
    "rows_after": conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
}
conn.rollback()
payload["rows_after_rollback"] = conn.execute(
    "SELECT COUNT(*) FROM embeddings"
).fetchone()[0]
print(json.dumps(payload, sort_keys=True))
'''
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_FIXTURE), str(Path(__file__).parents[1]), env.get("PYTHONPATH", "")]
    )
    env["HOME"] = str(tmp_path / "home")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "audit_delegate": "core.integrity.inspect_integrity",
        "repair_delegate": "core.integrity.repair_integrity",
        "status": "completed",
        "removed": 1,
        "remaining": {},
        "in_transaction": True,
        "rows_after": 1,
        "rows_after_rollback": 2,
    }
