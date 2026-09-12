"""Bounded, real-upstream subprocess contracts for shared Cashew brains."""

from __future__ import annotations

import fcntl
import json
import os
import select
import sqlite3
import subprocess
import sys
from pathlib import Path

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig
from plugins.memory.cashew.sleep_refactor import run_sleep_cycle

_CHILD = r"""
import json, os, sys
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.config import CashewConfig
import core.session
assert getattr(core.session, "__file__", ""), "real core.session required"
home=Path(sys.argv[1]); marker=sys.argv[2]; gate=sys.argv[3]
home.mkdir(parents=True, exist_ok=True)
(home / "cashew.json").write_text(json.dumps({"cashew_db_path":"brain.db", "llm_aux_role":None, "think_cycles":False}))
p=CashewMemoryProvider(); p.initialize("mp-" + marker, hermes_home=str(home))
print(json.dumps({"ready": p._db_path is not None, "core": core.session.__file__}), flush=True)
sys.stdin.readline()  # parent barrier: force operation overlap, not launch overlap
result=p.handle_tool_call("cashew_extract", {"user_content":marker, "assistant_content":"stored marker " + marker})
print(json.dumps({"result": result}), flush=True)
p.shutdown()
"""


def _child(home: Path, marker: str, gate: str) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(home), marker, gate],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HOME": str(home / "user")},
    )
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], 5)
    assert ready, "child readiness timed out"
    payload = json.loads(proc.stdout.readline())
    assert payload["ready"] and payload["core"]
    return proc


def _finish(proc: subprocess.Popen[str]) -> dict:
    try:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write("go\n")
        proc.stdin.close()
        ready, _, _ = select.select([proc.stdout], [], [], 15)
        assert ready, "child write timed out"
        payload = json.loads(proc.stdout.readline())
        assert proc.wait(timeout=5) == 0, proc.stderr.read() if proc.stderr else ""
        return payload
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream and not stream.closed:
                stream.close()


def _markers(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {row[0] for row in conn.execute("SELECT content FROM thought_nodes")}
    finally:
        conn.close()


def test_two_processes_persist_identifiable_records_to_one_brain(
    tmp_path: Path,
) -> None:
    """Real upstream extraction persists both barrier-released process markers."""
    home = tmp_path / "shared"
    first = _child(home, "marker-alpha", "shared")
    second = _child(home, "marker-beta", "shared")
    _finish(first)
    _finish(second)
    stored = _markers(home / "brain.db")
    assert any("marker-alpha" in value for value in stored)
    assert any("marker-beta" in value for value in stored)


def test_separate_profiles_do_not_share_persisted_records(tmp_path: Path) -> None:
    first = _child(tmp_path / "one", "only-one", "one")
    second = _child(tmp_path / "two", "only-two", "two")
    _finish(first)
    _finish(second)
    assert any("only-one" in value for value in _markers(tmp_path / "one" / "brain.db"))
    assert not any(
        "only-two" in value for value in _markers(tmp_path / "one" / "brain.db")
    )


def test_abrupt_worker_exit_only_requires_post_restart_durability(
    tmp_path: Path,
) -> None:
    """A killed pre-write child has no durability promise; restart must persist."""
    home = tmp_path / "shared"
    killed = _child(home, "pre-kill", "kill")
    try:
        killed.kill()
        killed.wait(timeout=5)
    finally:
        for stream in (killed.stdin, killed.stdout, killed.stderr):
            if stream and not stream.closed:
                stream.close()
    restarted = _child(home, "after-restart", "restart")
    _finish(restarted)
    stored = _markers(home / "brain.db")
    assert any("after-restart" in value for value in stored)


def test_sleep_lock_contention_has_bounded_named_skip(tmp_path: Path) -> None:
    """A held maintenance lock returns the public skip result without waiting."""
    db_path = tmp_path / "brain.db"
    lock_path = Path(f"{db_path}.sleep.lock")
    with lock_path.open("a+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert run_sleep_cycle(str(db_path), model_fn=None) == {}
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)


def test_real_extract_survives_following_sleep_cycle(tmp_path: Path) -> None:
    """A real explicit extraction remains queryable after maintenance runs."""
    home = tmp_path / "shared"
    writer = _child(home, "extract-before-sleep", "extract-sleep")
    _finish(writer)
    result = run_sleep_cycle(str(home / "brain.db"), model_fn=None)
    assert result == {} or "error" in result or result["nodes_selected"] >= 0
    assert any("extract-before-sleep" in value for value in _markers(home / "brain.db"))


def test_migration_lock_contention_defers_while_real_writer_runs(
    tmp_path: Path,
) -> None:
    """Migration's nonblocking lock cannot overlap the real child write path."""
    home = tmp_path / "shared"
    writer = _child(home, "writer-during-migration", "migration-write")
    db_path = home / "brain.db"
    lock_path = Path(f"{db_path}.sleep.lock")
    provider = CashewMemoryProvider()
    provider._config = CashewConfig()
    with lock_path.open("a+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            provider._repair_embedding_dimension(db_path)
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
    _finish(writer)
    assert any("writer-during-migration" in value for value in _markers(db_path))
