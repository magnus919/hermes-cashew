"""Cross-process regression tests for the shared maintenance advisory lock."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from plugins.memory.cashew import CashewMemoryProvider, locking
from plugins.memory.cashew.locking import lock_path_for_db, try_maintenance_lock

_HOLDER = """
import sys
from plugins.memory.cashew.locking import try_maintenance_lock

with try_maintenance_lock(sys.argv[1]) as lock_fd:
    assert lock_fd is not None
    print("ready", flush=True)
    sys.stdin.readline()
"""


def _start_lock_holder(db_path: Path) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"
    return process


def _stop_lock_holder(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        assert process.stdin is not None
        process.stdin.write("release\n")
        process.stdin.close()
    assert process.wait(timeout=5) == 0


def test_active_old_lock_keeps_identity_and_blocks_second_process(
    tmp_path: Path,
) -> None:
    """An old mtime cannot turn a live holder into a second lock inode."""
    db_path = tmp_path / "state" / "custom.sqlite"
    db_path.parent.mkdir()
    holder = _start_lock_holder(db_path)
    lock_path = lock_path_for_db(db_path)
    original_inode = lock_path.stat().st_ino
    old = time.time() - 3601
    os.utime(lock_path, (old, old))
    try:
        with try_maintenance_lock(db_path) as contender:
            assert contender is None
        assert lock_path.stat().st_ino == original_inode
    finally:
        _stop_lock_holder(holder)

    with try_maintenance_lock(db_path) as recovered:
        assert recovered is not None


def test_terminated_holder_releases_without_lock_file_deletion(tmp_path: Path) -> None:
    """Process death releases flock while the durable lock pathname remains."""
    db_path = tmp_path / "brain.db"
    holder = _start_lock_holder(db_path)
    lock_path = lock_path_for_db(db_path)
    original_inode = lock_path.stat().st_ino
    holder.terminate()
    assert holder.wait(timeout=5) != 0

    with try_maintenance_lock(db_path) as recovered:
        assert recovered is not None
    assert lock_path.exists()
    assert lock_path.stat().st_ino == original_inode


def test_release_closes_descriptor_without_masking_protected_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed unlock cannot obscure the work failure or skip descriptor close."""
    real_open = Path.open
    opened = []

    class TrackedFile:
        def __init__(self, handle):
            self._handle = handle
            self.closed = False

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def close(self):
            self.closed = True
            return self._handle.close()

    def tracked_open(path, *args, **kwargs):
        handle = TrackedFile(real_open(path, *args, **kwargs))
        opened.append(handle)
        return handle

    real_flock = locking.fcntl.flock

    def unlock_failure(handle, operation):
        if operation == locking.fcntl.LOCK_UN:
            raise OSError("unlock")
        return real_flock(handle, operation)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(locking.fcntl, "flock", unlock_failure)

    with pytest.raises(RuntimeError, match="protected work"):
        with try_maintenance_lock(tmp_path / "brain.db"):
            raise RuntimeError("protected work")
    assert len(opened) == 1
    assert opened[0].closed is True


def test_unexpected_lock_acquisition_error_is_not_reported_as_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only EAGAIN/EWOULDBLOCK means a cycle should defer or skip."""
    real_flock = locking.fcntl.flock

    def denied_lock(handle, operation):
        if operation == locking.fcntl.LOCK_EX | locking.fcntl.LOCK_NB:
            raise OSError(errno.EPERM, "permission denied")
        return real_flock(handle, operation)

    monkeypatch.setattr(locking.fcntl, "flock", denied_lock)

    with pytest.raises(PermissionError):
        with try_maintenance_lock(tmp_path / "brain.db"):
            pass


def test_initialize_respects_held_custom_lock_without_removing_it(
    tmp_path: Path, monkeypatch
) -> None:
    """Initialization defers migration while a different process holds the DB lock."""
    import plugins.memory.cashew as cashew_module

    (tmp_path / "cashew.json").write_text(
        json.dumps({"cashew_db_path": "state/custom.sqlite"})
    )
    db_path = tmp_path / "state" / "custom.sqlite"
    db_path.parent.mkdir()
    holder = _start_lock_holder(db_path)
    lock_path = lock_path_for_db(db_path)
    original_inode = lock_path.stat().st_ino
    old = time.time() - 3601
    os.utime(lock_path, (old, old))
    migration_called = False

    def migration_locked(_self: CashewMemoryProvider, _db_path: Path) -> None:
        nonlocal migration_called
        migration_called = True

    monkeypatch.setattr(cashew_module, "_patch_upstream_embedding", lambda *_: None)
    monkeypatch.setattr(CashewMemoryProvider, "_ensure_db_schema", lambda *_: None)
    monkeypatch.setattr(
        CashewMemoryProvider, "_repair_embedding_dimension_locked", migration_locked
    )
    monkeypatch.setattr(cashew_module, "ContextRetriever", lambda **_: object())
    monkeypatch.setattr(CashewMemoryProvider, "_build_model_fn", lambda _: None)
    monkeypatch.setattr(CashewMemoryProvider, "_start_sync_worker", lambda _: None)

    provider = CashewMemoryProvider()
    try:
        provider.initialize("session", hermes_home=str(tmp_path))
        assert provider._retriever is not None
        assert migration_called is False
        assert lock_path.stat().st_ino == original_inode
    finally:
        provider.shutdown()
        _stop_lock_holder(holder)
