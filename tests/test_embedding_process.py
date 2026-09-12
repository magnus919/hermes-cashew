"""Process-boundary contracts for native embedding work."""

from __future__ import annotations

import socket
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from plugins.memory.cashew import embedding_process
from plugins.memory.cashew.embedding_process import (
    EmbeddingFailure,
    EmbeddingSupervisor,
    EmbeddingUnavailable,
)


@pytest.fixture
def child_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_root = tmp_path / "fake-imports"
    package = fake_root / "sentence_transformers"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text(
        textwrap.dedent(
            """
            import os
            import time
            import numpy as np

            class SentenceTransformer:
                def __init__(self, model_name, device=None):
                    self.model_name = model_name
                    self.device = "cuda:7" if device is None else device
                    if model_name == "startup-exit" and device == "mps":
                        os._exit(24)
                    if device == "mps":
                        raise RuntimeError("simulated device failure")
                    if model_name == "startup-hang":
                        time.sleep(60)

                def get_embedding_dimension(self):
                    return 0 if self.model_name == "invalid-dimension" else 4

                def encode(self, texts, **kwargs):
                    if texts and texts[0] == "hang":
                        time.sleep(60)
                    if texts and texts[0] == "slow":
                        time.sleep(0.2)
                    if texts and texts[0].startswith("exit"):
                        os._exit(23)
                    return np.asarray(
                        [[float(len(text)), 2.0, 3.0, 4.0] for text in texts],
                        dtype=np.float32,
                    )
            """
        )
    )
    launcher = tmp_path / "child-python"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(fake_root)!r})\n"
        "script = sys.argv.pop(1)\n"
        "runpy.run_path(script, run_name='__main__')\n"
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(embedding_process.sys, "executable", str(launcher))
    return launcher


def _supervisor(tmp_path: Path, **kwargs: object) -> EmbeddingSupervisor:
    return EmbeddingSupervisor(
        model=str(kwargs.pop("model", "fake-model")),
        device=str(kwargs.pop("device", "cpu")),
        dimension=4,
        cache_dir=tmp_path / "model-cache",
        startup_timeout=float(kwargs.pop("startup_timeout", 1.0)),
        wait_timeout=float(kwargs.pop("wait_timeout", 1.0)),
        active_timeout=float(kwargs.pop("active_timeout", 1.0)),
        teardown_timeout=float(kwargs.pop("teardown_timeout", 0.2)),
        backoff_base=float(kwargs.pop("backoff_base", 0.0)),
    )


def test_production_worker_returns_validated_float32_vectors(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(tmp_path)
    try:
        vectors = supervisor.encode(["one", "three"])
        assert vectors.dtype == np.float32
        assert vectors.shape == (2, 4)
        assert vectors[:, 0].tolist() == [3.0, 5.0]
        assert supervisor._process is not None
        assert supervisor._process.poll() is None
    finally:
        supervisor.close()
    assert supervisor._process is None


def test_unknown_dimension_comes_only_from_worker_handshake(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = EmbeddingSupervisor(
        model="fake-model",
        device="cpu",
        dimension=0,
        cache_dir=tmp_path / "model-cache",
        startup_timeout=1.0,
        wait_timeout=1.0,
        active_timeout=1.0,
        teardown_timeout=0.2,
        backoff_base=0.0,
    )
    try:
        assert supervisor.start() == 4
        assert supervisor.dimension == 4
        assert supervisor.encode([]).shape == (0, 4)
    finally:
        supervisor.close()


def test_invalid_worker_dimension_fails_before_schema_can_use_it(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = EmbeddingSupervisor(
        model="invalid-dimension",
        device="cpu",
        dimension=0,
        cache_dir=tmp_path / "model-cache",
        startup_timeout=1.0,
        wait_timeout=1.0,
        active_timeout=1.0,
        teardown_timeout=0.2,
        backoff_base=0.0,
    )
    try:
        with pytest.raises(EmbeddingUnavailable) as raised:
            supervisor.start()
        assert raised.value.reason is EmbeddingFailure.EXITED
        assert supervisor.dimension == 0
    finally:
        supervisor.close()


def test_worker_retries_non_cpu_initialization_on_cpu(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(tmp_path, device="mps")
    try:
        assert supervisor.encode(["one"]).shape == (1, 4)
        assert supervisor.effective_device == "cpu"
    finally:
        supervisor.close()


def test_auto_device_handshake_records_worker_effective_device(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(tmp_path, device="auto")
    try:
        assert supervisor.encode(["one"]).shape == (1, 4)
        assert supervisor.effective_device == "cuda:7"
    finally:
        supervisor.close()


def test_native_non_cpu_startup_exit_retries_once_on_cpu(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(tmp_path, model="startup-exit", device="mps")
    try:
        assert supervisor.encode(["one"]).shape == (1, 4)
        assert supervisor.effective_device == "cpu"
        assert supervisor._launch_device == "cpu"
        assert supervisor._cpu_retry_used
    finally:
        supervisor.close()


def test_non_cpu_runtime_exit_changes_next_generation_to_cpu(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(tmp_path, device="cuda")
    try:
        with pytest.raises(EmbeddingUnavailable) as raised:
            supervisor.encode(["exit"])
        assert raised.value.reason is EmbeddingFailure.EXITED
        assert supervisor._launch_device == "cpu"
        assert supervisor.encode(["recovery"]).shape == (1, 4)
        assert supervisor.effective_device == "cpu"
    finally:
        supervisor.close()


@pytest.mark.parametrize(("text", "active_timeout"), [("exit", 1.0), ("hang", 0.05)])
def test_exit_and_hang_are_contained_and_reaped(
    tmp_path: Path,
    child_python: Path,
    text: str,
    active_timeout: float,
) -> None:
    supervisor = _supervisor(
        tmp_path,
        active_timeout=active_timeout,
        wait_timeout=0.5 if text == "hang" else 1.0,
    )
    try:
        started = time.monotonic()
        with pytest.raises(EmbeddingUnavailable):
            supervisor.encode([text])
        assert time.monotonic() - started < 1.0
        assert supervisor._process is None or supervisor._reaping
    finally:
        supervisor.close()


def test_request_send_stall_obeys_active_deadline(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, teardown_timeout=0.01)
    parent, child = socket.socketpair()
    parent.setblocking(False)
    supervisor._socket = parent
    try:
        started = time.monotonic()
        with pytest.raises(EmbeddingUnavailable) as raised:
            supervisor._send(b"x" * (16 * 1024 * 1024), started + 0.02)
        assert raised.value.reason is EmbeddingFailure.TIMEOUT
        assert time.monotonic() - started < 0.5
    finally:
        child.close()
        supervisor.close()


def test_request_thread_start_failure_releases_slot_and_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _supervisor(tmp_path)
    real_start = threading.Thread.start

    def fail_request_start(thread: threading.Thread) -> None:
        if thread.name == "cashew-embedding-request":
            raise RuntimeError("simulated request thread start failure")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_request_start)
    with pytest.raises(EmbeddingUnavailable) as raised:
        supervisor.encode(["one"])
    assert raised.value.reason is EmbeddingFailure.STARTUP
    assert supervisor._request_lock.acquire(blocking=False)
    supervisor._request_lock.release()
    supervisor.close()
    replacement = _supervisor(tmp_path)
    replacement.close()


def test_cleanup_thread_start_failure_still_releases_owner(
    tmp_path: Path,
    child_python: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, active_timeout=1.0)
    supervisor.start()
    request_errors: list[Exception] = []

    def request() -> None:
        try:
            supervisor.encode(["slow"])
        except Exception as exc:
            request_errors.append(exc)

    caller = threading.Thread(target=request)
    caller.start()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not supervisor._request_lock.acquire(blocking=False):
            break
        supervisor._request_lock.release()
        time.sleep(0.005)
    else:
        pytest.fail("embedding request did not acquire its active slot")

    real_start = threading.Thread.start

    def fail_cleanup_start(thread: threading.Thread) -> None:
        if thread.name == "cashew-embedding-close":
            raise RuntimeError("simulated cleanup thread start failure")
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_cleanup_start)
    supervisor.close(timeout=0.0)
    caller.join(timeout=1.0)
    assert not caller.is_alive()
    assert request_errors
    assert supervisor._owner_released.wait(timeout=1.0)
    replacement = _supervisor(tmp_path)
    replacement.close()


def test_second_provider_fails_closed_even_with_identical_identity(
    tmp_path: Path, child_python: Path
) -> None:
    owner = _supervisor(tmp_path)
    try:
        with pytest.raises(EmbeddingUnavailable) as raised:
            _supervisor(tmp_path)
        assert raised.value.reason is EmbeddingFailure.OWNED
    finally:
        owner.close()


def test_short_waiter_does_not_kill_active_request(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(tmp_path, active_timeout=1.0)
    result: list[np.ndarray] = []

    request = threading.Thread(
        target=lambda: result.append(supervisor.encode(["slow"], wait_timeout=1.0))
    )
    request.start()
    try:
        deadline = time.monotonic() + 1.0
        while supervisor._process is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert supervisor._process is not None
        process = supervisor._process

        with pytest.raises(EmbeddingUnavailable) as raised:
            supervisor.encode(["one"], wait_timeout=0.01)
        assert raised.value.reason is EmbeddingFailure.BUSY
        assert process.poll() is None
        assert not supervisor._reaping

        request.join(timeout=1.0)
        assert not request.is_alive()
        assert result[0].shape == (1, 4)
        assert supervisor._process is process
        assert process.poll() is None
    finally:
        supervisor.close()


def test_reaping_generation_blocks_replacement_until_exit_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _supervisor(tmp_path, teardown_timeout=0.01)
    release_reaper = threading.Event()

    class DelayedProcess:
        pid = 43210
        returncode = -9

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake-worker", timeout)
            release_reaper.wait(timeout=1.0)
            return 0

    process = DelayedProcess()
    supervisor._process = process  # type: ignore[assignment]
    monkeypatch.setattr(embedding_process.os, "killpg", lambda *_args: None)
    try:
        supervisor._terminate()
        assert supervisor._reaping
        assert supervisor._process is process
        with pytest.raises(EmbeddingUnavailable) as raised:
            supervisor._start()
        assert raised.value.reason is EmbeddingFailure.CLOSED
        supervisor.close()
        assert supervisor._reaping
        assert supervisor._process is process
    finally:
        release_reaper.set()
        deadline = time.monotonic() + 1.0
        while supervisor._reaping and time.monotonic() < deadline:
            time.sleep(0.005)
    assert not supervisor._reaping
    assert supervisor._process is None


def test_close_during_startup_is_bounded_and_reaps_child(
    tmp_path: Path, child_python: Path
) -> None:
    supervisor = _supervisor(
        tmp_path,
        model="startup-hang",
        startup_timeout=1.0,
        wait_timeout=1.0,
        teardown_timeout=0.05,
    )
    with pytest.raises(EmbeddingUnavailable) as raised:
        supervisor.encode(["one"], wait_timeout=0.01)
    assert raised.value.reason is EmbeddingFailure.BUSY
    started = time.monotonic()
    supervisor.close(timeout=0.1)
    assert time.monotonic() - started < 0.5
    deadline = time.monotonic() + 1.0
    while supervisor._reaping and time.monotonic() < deadline:
        time.sleep(0.005)
    assert supervisor._process is None
    assert not supervisor._reaping


def test_close_before_process_publication_keeps_owner_until_request_exits(
    tmp_path: Path,
    child_python: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    entered_popen = threading.Event()
    release_popen = threading.Event()
    request_failure: list[Exception] = []
    real_popen = embedding_process.subprocess.Popen

    def delayed_popen(*args: object, **kwargs: object):
        entered_popen.set()
        release_popen.wait(timeout=1.0)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(embedding_process.subprocess, "Popen", delayed_popen)

    def request() -> None:
        try:
            supervisor.encode(["one"])
        except Exception as exc:
            request_failure.append(exc)

    thread = threading.Thread(target=request)
    thread.start()
    assert entered_popen.wait(timeout=1.0)
    supervisor.close(timeout=0.0)
    assert supervisor._close_waiting
    with pytest.raises(EmbeddingUnavailable) as raised:
        _supervisor(tmp_path)
    assert raised.value.reason is EmbeddingFailure.OWNED

    release_popen.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    deadline = time.monotonic() + 1.0
    while supervisor._close_waiting and time.monotonic() < deadline:
        time.sleep(0.005)
    assert request_failure
    assert not supervisor._close_waiting
    replacement = _supervisor(tmp_path)
    replacement.close()


def test_child_environment_excludes_credentials_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("CASHEW_MEMORY_SENTINEL", "private-memory")
    supervisor = _supervisor(tmp_path)
    try:
        child_env = supervisor._child_environment()
        assert "secret-api-key" not in repr(child_env)
        assert "private-memory" not in repr(child_env)
        assert child_env["HF_HOME"] == str(tmp_path / "model-cache" / "huggingface")
    finally:
        supervisor.close()


def test_failure_log_contains_only_bounded_metadata(
    tmp_path: Path,
    child_python: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = _supervisor(tmp_path)
    private_text = "exit private-memory-sentinel"
    try:
        with pytest.raises(EmbeddingUnavailable):
            supervisor.encode([private_text])
        assert private_text not in caplog.text
        assert "operation=encode" in caplog.text
        assert "reason=exited" in caplog.text
        assert "count=1" in caplog.text
    finally:
        supervisor.close()
