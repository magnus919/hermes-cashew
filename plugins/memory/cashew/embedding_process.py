"""Owned subprocess boundary for SentenceTransformer loading and encoding."""

from __future__ import annotations

import contextlib
import contextvars
import enum
import hashlib
import json
import logging
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np

from .embedding import DEFAULT_EMBEDDING_DEVICE, normalize_embedding_device

logger = logging.getLogger(__name__)
_LENGTH = struct.Struct("!I")
_MAX_FRAME = 16 * 1024 * 1024
_MAX_DIMENSION = 65536
_OWNER_LOCK = threading.Lock()
_OWNER: EmbeddingSupervisor | None = None
_CALLER_WAIT: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "cashew_embedding_caller_wait", default=None
)


class EmbeddingFailure(str, enum.Enum):
    BUSY = "busy"
    STARTUP = "startup"
    TIMEOUT = "timeout"
    EXITED = "exited"
    PROTOCOL = "protocol"
    CLOSED = "closed"
    OWNED = "owned"


class EmbeddingUnavailable(RuntimeError):  # noqa: N818 - public boundary term
    """Payload-free failure that may cross the embedding service boundary."""

    def __init__(self, reason: EmbeddingFailure) -> None:
        self.reason = reason
        super().__init__(reason.value)


class NoInProcessEmbeddingBackend:
    def __init__(self, dimension: int) -> None:
        self.dim = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        raise EmbeddingUnavailable(EmbeddingFailure.CLOSED)


class ProcessEmbeddingBackend:
    def __init__(self, supervisor: "EmbeddingSupervisor") -> None:
        self.supervisor = supervisor

    @property
    def dim(self) -> int:
        return self.supervisor.dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.supervisor.encode(texts, wait_timeout=_CALLER_WAIT.get())


@contextlib.contextmanager
def embedding_caller_wait(seconds: float) -> Iterator[None]:
    """Set caller patience without changing the active worker deadline."""
    token = _CALLER_WAIT.set(seconds)
    try:
        yield
    finally:
        _CALLER_WAIT.reset(token)


class EmbeddingSupervisor:
    """Exclusive, bounded owner of one embedding worker generation."""

    def __init__(
        self,
        *,
        model: str,
        device: object,
        dimension: int,
        cache_dir: Path,
        startup_timeout: float = 60.0,
        wait_timeout: float = 30.0,
        active_timeout: float = 30.0,
        teardown_timeout: float = 0.25,
        backoff_base: float = 1.0,
    ) -> None:
        if os.name != "posix":
            raise EmbeddingUnavailable(EmbeddingFailure.STARTUP)
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or not 0 <= dimension <= _MAX_DIMENSION
        ):
            raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
        self.model = model
        self._model_log_id = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
        self.device = normalize_embedding_device(device)
        self._launch_device = self.device
        self._cpu_retry_used = self.device == DEFAULT_EMBEDDING_DEVICE
        self.effective_device: str | None = None
        self.dimension = dimension
        self.startup_timeout = startup_timeout
        self.wait_timeout = wait_timeout
        self.active_timeout = active_timeout
        self.teardown_timeout = teardown_timeout
        self.cache_dir = Path(cache_dir)
        self.backoff_base = backoff_base
        self.generation = uuid.uuid4().hex
        self._request_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._closed = False
        self._close_waiting = False
        self._reaping = False
        self._failure_count = 0
        self._next_start = 0.0
        self._last_exit_code: int | None = None
        self._owner_released = threading.Event()
        self._close_callbacks_lock = threading.Lock()
        self._close_callbacks: list[Callable[[], None]] = []
        self._claim_owner()

    def _child_environment(self) -> dict[str, str]:
        allowed = {
            "PATH",
            "TMPDIR",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "CUDA_VISIBLE_DEVICES",
        }
        child_env = {key: value for key, value in os.environ.items() if key in allowed}
        child_env["HF_HOME"] = str(self.cache_dir / "huggingface")
        return child_env

    def _claim_owner(self) -> None:
        global _OWNER
        with _OWNER_LOCK:
            if _OWNER is not None:
                raise EmbeddingUnavailable(EmbeddingFailure.OWNED)
            _OWNER = self

    def _release_owner(self) -> None:
        global _OWNER
        released = False
        with _OWNER_LOCK:
            if _OWNER is self:
                _OWNER = None
                released = True
        if not released:
            return
        with self._close_callbacks_lock:
            self._owner_released.set()
            callbacks, self._close_callbacks = self._close_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.warning("embedding close callback failed", exc_info=True)

    def _when_closed(self, callback: Callable[[], None]) -> None:
        """Run *callback* once this supervisor no longer owns process state."""
        call_now = False
        with self._close_callbacks_lock:
            if self._owner_released.is_set():
                call_now = True
            else:
                self._close_callbacks.append(callback)
        if call_now:
            callback()

    def _read_exact(self, size: int, deadline: float) -> bytes:
        channel = self._socket
        if channel is None:
            raise EmbeddingUnavailable(EmbeddingFailure.EXITED)
        chunks: list[bytes] = []
        while size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([channel], [], [], remaining)[0]:
                raise EmbeddingUnavailable(EmbeddingFailure.TIMEOUT)
            try:
                chunk = channel.recv(size)
            except BlockingIOError:
                continue
            if not chunk:
                raise EmbeddingUnavailable(EmbeddingFailure.EXITED)
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def _receive(self, deadline: float) -> bytes:
        size = _LENGTH.unpack(self._read_exact(_LENGTH.size, deadline))[0]
        if size > _MAX_FRAME:
            raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
        return self._read_exact(size, deadline)

    def _send(self, payload: bytes, deadline: float) -> None:
        channel = self._socket
        if len(payload) > _MAX_FRAME or channel is None:
            raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
        pending = memoryview(_LENGTH.pack(len(payload)) + payload)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [channel], [], remaining)[1]:
                raise EmbeddingUnavailable(EmbeddingFailure.TIMEOUT)
            try:
                sent = channel.send(pending)
            except BlockingIOError:
                continue
            if sent <= 0:
                raise EmbeddingUnavailable(EmbeddingFailure.EXITED)
            pending = pending[sent:]

    def _start_once(self, device: str) -> None:
        self._last_exit_code = None
        parent, child = socket.socketpair()
        worker = Path(__file__).with_name("embedding_worker.py")
        command = [
            sys.executable,
            str(worker),
            "--fd",
            str(child.fileno()),
            "--generation",
            self.generation,
            "--model",
            self.model,
            "--device",
            device,
            "--dimension",
            str(self.dimension),
        ]
        try:
            process = subprocess.Popen(
                command,
                pass_fds=(child.fileno(),),
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                env=self._child_environment(),
            )
        except Exception as exc:
            parent.close()
            raise EmbeddingUnavailable(EmbeddingFailure.STARTUP) from exc
        finally:
            child.close()
        parent.setblocking(False)
        self._process = process
        self._socket = parent
        if self._closed:
            self._terminate()
            raise EmbeddingUnavailable(EmbeddingFailure.CLOSED)
        try:
            hello = json.loads(self._receive(time.monotonic() + self.startup_timeout))
            reported_device = hello.get("device") if isinstance(hello, dict) else None
            reported_dimension = (
                hello.get("dimension") if isinstance(hello, dict) else None
            )
            if not isinstance(hello, dict) or (
                hello.get("op") != "hello"
                or hello.get("version") != 1
                or hello.get("generation") != self.generation
                or hello.get("model") != self.model
                or hello.get("requested_device") != device
                or not isinstance(reported_dimension, int)
                or isinstance(reported_dimension, bool)
                or not 0 < reported_dimension <= _MAX_DIMENSION
                or (self.dimension > 0 and reported_dimension != self.dimension)
                or not isinstance(reported_device, str)
                or not reported_device
                or len(reported_device) > 64
                or (
                    device != "auto"
                    and reported_device not in {device, DEFAULT_EMBEDDING_DEVICE}
                )
            ):
                raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
            if self.dimension == 0:
                self.dimension = reported_dimension
            self.effective_device = reported_device
            if self.effective_device == DEFAULT_EMBEDDING_DEVICE:
                self._launch_device = DEFAULT_EMBEDDING_DEVICE
                self._cpu_retry_used = True
        except EmbeddingUnavailable:
            self._terminate()
            raise
        except (ConnectionError, OSError) as exc:
            self._terminate()
            raise EmbeddingUnavailable(EmbeddingFailure.EXITED) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._terminate()
            raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL) from exc

    def _start(self) -> None:
        if self._closed or self._reaping:
            raise EmbeddingUnavailable(EmbeddingFailure.CLOSED)
        if time.monotonic() < self._next_start:
            raise EmbeddingUnavailable(EmbeddingFailure.BUSY)
        try:
            self._start_once(self._launch_device)
        except EmbeddingUnavailable as exc:
            may_retry_cpu = (
                not self._cpu_retry_used
                and self._launch_device != DEFAULT_EMBEDDING_DEVICE
                and not self._reaping
                and self._process is None
                and exc.reason
                in {
                    EmbeddingFailure.STARTUP,
                    EmbeddingFailure.EXITED,
                    EmbeddingFailure.TIMEOUT,
                }
            )
            if not may_retry_cpu:
                self._record_failure()
                self._log_failure("startup", exc.reason)
                raise
            logger.warning(
                "embedding worker startup failed; retrying once on CPU: "
                "generation=%s model_id=%s requested_device=%s reason=%s",
                self.generation,
                self._model_log_id,
                self.device,
                exc.reason.value,
            )
            self._cpu_retry_used = True
            self._launch_device = DEFAULT_EMBEDDING_DEVICE
            try:
                self._start_once(DEFAULT_EMBEDDING_DEVICE)
            except EmbeddingUnavailable as cpu_exc:
                self._record_failure()
                self._log_failure("startup", cpu_exc.reason)
                raise

    def start(self) -> int:
        """Start and verify the child without issuing an encode request."""
        if not self._request_lock.acquire(timeout=max(0.0, self.wait_timeout)):
            raise EmbeddingUnavailable(EmbeddingFailure.BUSY)
        try:
            if self._process is None:
                self._start()
            return self.dimension
        finally:
            self._request_lock.release()
            if self._close_waiting:
                self._finish_close_after_request()

    def _record_failure(self) -> None:
        self._failure_count += 1
        delay = min(30.0, self.backoff_base * (2 ** min(self._failure_count - 1, 5)))
        self._next_start = time.monotonic() + delay

    def _log_failure(
        self,
        operation: str,
        reason: EmbeddingFailure,
        *,
        request_id: str = "-",
        count: int = 0,
        started: float | None = None,
    ) -> None:
        elapsed = 0.0 if started is None else max(0.0, time.monotonic() - started)
        logger.warning(
            "embedding worker failure: operation=%s generation=%s model_id=%s "
            "requested_device=%s effective_device=%s request_id=%s count=%d "
            "reason=%s exit_code=%s elapsed_s=%.3f",
            operation,
            self.generation,
            self._model_log_id,
            self.device,
            self.effective_device or "unknown",
            request_id,
            count,
            reason.value,
            self._last_exit_code,
            elapsed,
        )

    def _encode_owned(
        self, texts: list[str], *, request_id: str, request_payload: bytes
    ) -> np.ndarray:
        import numpy as np

        try:
            if self._closed or self._reaping:
                raise EmbeddingUnavailable(EmbeddingFailure.CLOSED)
            if self._process is None:
                self._start()
            started = time.monotonic()
            deadline = time.monotonic() + self.active_timeout
            try:
                self._send(request_payload, deadline)
                header = json.loads(self._receive(deadline))
                raw = self._receive(deadline)
                expected_bytes = len(texts) * self.dimension * 4
                if not isinstance(header, dict) or (
                    header.get("op") != "vectors"
                    or header.get("version") != 1
                    or header.get("generation") != self.generation
                    or header.get("request_id") != request_id
                    or header.get("count") != len(texts)
                    or header.get("dimension") != self.dimension
                    or header.get("device") != self.effective_device
                    or header.get("bytes") != expected_bytes
                    or len(raw) != expected_bytes
                ):
                    raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
                vectors = np.frombuffer(raw, dtype="<f4").reshape(
                    len(texts), self.dimension
                )
                if not np.isfinite(vectors).all():
                    raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
                self._failure_count = 0
                self._next_start = 0.0
                return cast(np.ndarray, vectors.copy())
            except EmbeddingUnavailable as exc:
                if self.effective_device != DEFAULT_EMBEDDING_DEVICE:
                    self._cpu_retry_used = True
                    self._launch_device = DEFAULT_EMBEDDING_DEVICE
                self._terminate()
                self._record_failure()
                self._log_failure(
                    "encode",
                    exc.reason,
                    request_id=request_id,
                    count=len(texts),
                    started=started,
                )
                raise
            except (
                BrokenPipeError,
                ConnectionError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                if self.effective_device != DEFAULT_EMBEDDING_DEVICE:
                    self._cpu_retry_used = True
                    self._launch_device = DEFAULT_EMBEDDING_DEVICE
                self._terminate()
                self._record_failure()
                failure = EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
                self._log_failure(
                    "encode",
                    failure.reason,
                    request_id=request_id,
                    count=len(texts),
                    started=started,
                )
                raise failure from exc
        finally:
            self._request_lock.release()

    def encode(
        self, texts: list[str], *, wait_timeout: float | None = None
    ) -> np.ndarray:
        if not texts:
            import numpy as np

            return np.zeros((0, self.dimension), dtype=np.float32)
        if len(texts) > 100 or any(
            not isinstance(text, str) or len(text.encode("utf-8")) > 1048576
            for text in texts
        ):
            raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
        request_id = uuid.uuid4().hex
        request_payload = json.dumps(
            {
                "op": "encode",
                "version": 1,
                "generation": self.generation,
                "request_id": request_id,
                "texts": texts,
            },
            separators=(",", ":"),
        ).encode()
        if len(request_payload) > _MAX_FRAME:
            raise EmbeddingUnavailable(EmbeddingFailure.PROTOCOL)
        wait = self.wait_timeout if wait_timeout is None else wait_timeout
        caller_deadline = time.monotonic() + max(0.0, wait)
        if not self._request_lock.acquire(timeout=max(0.0, wait)):
            raise EmbeddingUnavailable(EmbeddingFailure.BUSY)

        result: list[np.ndarray] = []
        failure: list[Exception] = []
        done = threading.Event()

        def execute() -> None:
            try:
                result.append(
                    self._encode_owned(
                        texts,
                        request_id=request_id,
                        request_payload=request_payload,
                    )
                )
            except Exception as exc:
                failure.append(exc)
            finally:
                done.set()
                if self._close_waiting:
                    self._finish_close_after_request()

        try:
            request_thread = threading.Thread(
                target=execute, daemon=True, name="cashew-embedding-request"
            )
            request_thread.start()
        except Exception as exc:
            self._request_lock.release()
            raise EmbeddingUnavailable(EmbeddingFailure.STARTUP) from exc
        if not done.wait(timeout=max(0.0, caller_deadline - time.monotonic())):
            raise EmbeddingUnavailable(EmbeddingFailure.BUSY)
        if failure:
            raise failure[0]
        return result[0]

    def _terminate(self, *, deadline: float | None = None) -> None:
        def wait_budget() -> float:
            if deadline is None:
                return self.teardown_timeout
            return min(self.teardown_timeout, max(0.0, deadline - time.monotonic()))

        channel, self._socket = self._socket, None
        if channel is not None:
            channel.close()
        if self._reaping:
            return
        process = self._process
        if process is None:
            return

        def signal_process(sig: signal.Signals) -> None:
            try:
                os.killpg(process.pid, sig)
            except PermissionError:
                try:
                    process.send_signal(sig)
                except (PermissionError, ProcessLookupError):
                    pass
            except ProcessLookupError:
                pass

        if process.poll() is None:
            try:
                signal_process(signal.SIGTERM)
                process.wait(timeout=wait_budget())
            except subprocess.TimeoutExpired:
                signal_process(signal.SIGKILL)
        try:
            process.wait(timeout=wait_budget())
        except subprocess.TimeoutExpired:
            self._reaping = True
            try:
                reaper = threading.Thread(
                    target=self._reap,
                    args=(process,),
                    daemon=True,
                    name="cashew-embedding-reaper",
                )
                reaper.start()
            except Exception:
                # Retain the process and owner for a later close() retry. It is
                # unsafe to publish a replacement until wait() actually reaps
                # this generation.
                self._reaping = False
                logger.warning("embedding reaper unavailable; close remains pending")
            return
        self._last_exit_code = process.returncode
        self._process = None

    def _reap(self, process: subprocess.Popen[bytes]) -> None:
        process.wait()
        self._last_exit_code = process.returncode
        self._process = None
        self._reaping = False
        if self._closed:
            self._release_owner()

    def _finish_close_after_request(self) -> None:
        with self._request_lock:
            self._terminate()
        self._close_waiting = False
        if not self._reaping and self._process is None:
            self._release_owner()

    def close(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)

        def wait_budget() -> float:
            if deadline is None:
                return self.teardown_timeout
            return min(self.teardown_timeout, max(0.0, deadline - time.monotonic()))

        self._closed = True
        # A caller-provided timeout is the owner's total shutdown budget. Give
        # an admitted request that full remaining window to release its slot;
        # teardown operations themselves retain their shorter phase cap.
        request_wait = (
            self.teardown_timeout
            if deadline is None
            else max(0.0, deadline - time.monotonic())
        )
        if self._request_lock.acquire(timeout=request_wait):
            try:
                if self._socket is not None:
                    try:
                        self._send(
                            json.dumps(
                                {"op": "shutdown", "generation": self.generation}
                            ).encode(),
                            time.monotonic() + wait_budget(),
                        )
                    except Exception:
                        pass
                self._terminate(deadline=deadline)
            finally:
                self._request_lock.release()
        else:
            self._close_waiting = True
            self._terminate(deadline=deadline)
            try:
                cleanup = threading.Thread(
                    target=self._finish_close_after_request,
                    daemon=True,
                    name="cashew-embedding-close",
                )
                cleanup.start()
            except Exception:
                # The request thread also observes _close_waiting in its
                # finally block. Cover the narrow race where it finished just
                # before that flag was published by claiming the now-free slot.
                if self._request_lock.acquire(blocking=False):
                    try:
                        self._terminate(deadline=deadline)
                    finally:
                        self._request_lock.release()
                    self._close_waiting = False
                    if not self._reaping and self._process is None:
                        self._release_owner()
            return
        if not self._reaping and self._process is None:
            self._release_owner()


def _close_owner_for_tests() -> None:
    """Release process state between isolated pytest cases."""
    with _OWNER_LOCK:
        owner = _OWNER
    if owner is not None:
        owner.close()
