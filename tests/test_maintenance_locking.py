"""Cross-process regression tests for the shared maintenance advisory lock."""

from __future__ import annotations

import errno
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from plugins.memory.cashew import CashewMemoryProvider, locking
from plugins.memory.cashew.locking import (
    MaintenanceLockAcquisitionError,
    lock_path_for_db,
    try_maintenance_lock,
)

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
    try:
        assert process.stdout is not None
        readable, _, _ = select.select([process.stdout], [], [], 5)
        assert readable, "lock holder did not signal readiness within 5 seconds"
        assert process.stdout.readline().strip() == "ready"
        return process
    except BaseException:
        _stop_lock_holder(process, expect_success=False)
        raise


def _stop_lock_holder(
    process: subprocess.Popen[str], *, expect_success: bool = True
) -> None:
    try:
        if process.poll() is None:
            assert process.stdin is not None
            process.stdin.write("release\n")
            process.stdin.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
        if expect_success:
            assert process.returncode == 0
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


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
    try:
        holder.terminate()
        assert holder.wait(timeout=5) != 0
    finally:
        _stop_lock_holder(holder, expect_success=False)

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

    with pytest.raises(MaintenanceLockAcquisitionError) as error:
        with try_maintenance_lock(tmp_path / "brain.db"):
            pass
    assert isinstance(error.value.__cause__, PermissionError)


def test_initialize_survives_acquisition_failure_when_inspection_needs_no_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A healthy database remains usable when only maintenance lock access fails."""
    import plugins.memory.cashew as cashew_module

    (tmp_path / "cashew.json").write_text("{}")
    real_flock = locking.fcntl.flock

    def denied_lock(handle, operation):
        if operation == locking.fcntl.LOCK_EX | locking.fcntl.LOCK_NB:
            raise OSError(errno.EPERM, "permission denied")
        return real_flock(handle, operation)

    monkeypatch.setattr(locking.fcntl, "flock", denied_lock)
    monkeypatch.setattr(
        cashew_module, "_patch_upstream_embedding", lambda *_, **__: None
    )
    monkeypatch.setattr(CashewMemoryProvider, "_ensure_db_schema", lambda *_: None)
    monkeypatch.setattr(
        CashewMemoryProvider, "_embedding_dimensions", lambda *_: (set(), None)
    )
    monkeypatch.setattr(
        CashewMemoryProvider,
        "_repair_embedding_dimension_locked",
        lambda *_: pytest.fail("migration must not run without the maintenance lock"),
    )
    monkeypatch.setattr(cashew_module, "ContextRetriever", lambda **_: object())
    monkeypatch.setattr(CashewMemoryProvider, "_build_model_fn", lambda _: None)
    monkeypatch.setattr(CashewMemoryProvider, "_start_sync_worker", lambda _: None)

    provider = CashewMemoryProvider()
    try:
        provider.initialize("session", hermes_home=str(tmp_path))
        assert provider._config is not None
        assert provider._retriever is not None
        assert "migration not needed; unable to acquire" in caplog.text
    finally:
        provider.shutdown()


def test_initialize_degrades_when_acquisition_failure_hides_required_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Initialization fails closed if a lock failure prevents needed migration."""
    import core.embedding_service

    import plugins.memory.cashew as cashew_module

    (tmp_path / "cashew.json").write_text("{}")
    real_flock = locking.fcntl.flock

    def denied_lock(handle, operation):
        if operation == locking.fcntl.LOCK_EX | locking.fcntl.LOCK_NB:
            raise OSError(errno.EPERM, "permission denied")
        return real_flock(handle, operation)

    monkeypatch.setattr(locking.fcntl, "flock", denied_lock)
    monkeypatch.setattr(
        cashew_module, "_patch_upstream_embedding", lambda *_, **__: None
    )
    monkeypatch.setattr(CashewMemoryProvider, "_ensure_db_schema", lambda *_: None)
    monkeypatch.setattr(
        CashewMemoryProvider, "_embedding_dimensions", lambda *_: ({384}, 384)
    )
    monkeypatch.setattr(core.embedding_service, "resolve_embedding_dim", lambda _: 1024)
    monkeypatch.setattr(cashew_module, "ContextRetriever", lambda **_: object())
    monkeypatch.setattr(CashewMemoryProvider, "_build_model_fn", lambda _: None)
    monkeypatch.setattr(CashewMemoryProvider, "_start_sync_worker", lambda _: None)

    provider = CashewMemoryProvider()
    try:
        provider.initialize("session", hermes_home=str(tmp_path))
        assert provider._config is None
        assert provider._db_path is None
        assert provider._retriever is None
        assert "provider will report unavailable" in caplog.text
    finally:
        provider.shutdown()


@pytest.mark.parametrize("configured_path", ["cashew/brain.db", "state/custom.sqlite"])
def test_initialize_respects_held_aged_lock_without_removing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_path: str
) -> None:
    """Initialization defers migration while a different process holds the DB lock."""
    import plugins.memory.cashew as cashew_module

    (tmp_path / "cashew.json").write_text(
        json.dumps({"cashew_db_path": configured_path})
    )
    db_path = tmp_path / configured_path
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

    monkeypatch.setattr(
        cashew_module, "_patch_upstream_embedding", lambda *_, **__: None
    )
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
