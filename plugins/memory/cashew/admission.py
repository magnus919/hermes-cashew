"""Immutable graph/cache admission tokens with one ordered deadline."""

from __future__ import annotations

import contextvars
import dataclasses
import pathlib
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator


class OperationAdmissionError(RuntimeError):
    """Operation could not be admitted or its identity was stale."""


class AdmissionLeaseOwner:
    """Transferable owner for the descriptors held by one admission."""

    def __init__(self, stack: ExitStack) -> None:
        self.stack = stack
        self.transferred = False
        self.closed = False
        self._lock = threading.Lock()

    def transfer(self) -> None:
        with self._lock:
            if self.closed:
                raise OperationAdmissionError("admission lease is already closed")
            if self.transferred:
                raise OperationAdmissionError("admission lease was already transferred")
            self.transferred = True

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self.stack.close()


@dataclasses.dataclass(frozen=True)
class OperationAdmission:
    graph_path: pathlib.Path | None
    cache_path: pathlib.Path | None
    model: str | None
    embedding_dim: int | None
    vec_dim: int | None
    epoch: int | None
    supervisor: Any
    embedding_generation: str | int | None
    graph_lease: Any
    cache_lease: Any
    exclusive: bool = False
    cache_exclusive: bool = False
    lease_owner: AdmissionLeaseOwner | None = dataclasses.field(
        default=None, compare=False, repr=False
    )


_CURRENT: contextvars.ContextVar[OperationAdmission | None] = contextvars.ContextVar(
    "hermes_cashew_operation_admission", default=None
)


def current_admission() -> OperationAdmission | None:
    return _CURRENT.get()


@contextmanager
def install_admission(admission: OperationAdmission) -> Iterator[OperationAdmission]:
    """Install an already-owned immutable token in a transferred worker."""
    token = _CURRENT.set(admission)
    try:
        yield admission
    finally:
        _CURRENT.reset(token)


@contextmanager
def transfer_admission(admission: OperationAdmission) -> Iterator[OperationAdmission]:
    """Transfer lease ownership to a worker that outlives the caller."""
    owner = admission.lease_owner
    if owner is None:
        raise OperationAdmissionError("admission has no transferable lease owner")
    owner.transfer()
    try:
        yield admission
    except BaseException:
        owner.close()
        raise


def _canonical(path: str | pathlib.Path | None) -> pathlib.Path | None:
    return pathlib.Path(path).resolve(strict=False) if path is not None else None


@contextmanager
def admit_operation(  # noqa: C901 - bounded two-resource admission state machine
    *,
    graph_path: str | pathlib.Path | None = None,
    cache_path: str | pathlib.Path | None = None,
    model: str | None = None,
    embedding_dim: int | None = None,
    vec_dim: int | None = None,
    epoch: int | None = None,
    supervisor: Any = None,
    embedding_generation: str | int | None = None,
    exclusive: bool = False,
    cache_exclusive: bool = False,
    deadline: float = 5.0,
) -> Iterator[OperationAdmission]:
    """Acquire graph then cache, or cache alone, without lock upgrades."""
    # The flat-entry loader preloads sibling modules before registering the
    # synthetic package.  Resolve the lock helpers at call time so this module
    # remains importable in both the normal package and flat plugin paths.
    from .locking import try_maintenance_lock, try_shared_lock

    graph = _canonical(graph_path)
    cache = _canonical(cache_path)
    if graph is None and cache is None:
        raise OperationAdmissionError("an operation requires graph or cache identity")
    if graph is not None and cache == graph:
        raise OperationAdmissionError("graph and cache must have distinct lock files")
    parent = current_admission()
    if parent is not None:
        if parent.graph_path != graph or parent.cache_path != cache:
            raise OperationAdmissionError("nested admission identity mismatch")
        if model is not None and parent.model != model:
            raise OperationAdmissionError("nested admission model mismatch")
        if embedding_dim is not None and parent.embedding_dim != embedding_dim:
            raise OperationAdmissionError("nested admission dimension mismatch")
        if vec_dim is not None and parent.vec_dim != vec_dim:
            raise OperationAdmissionError("nested admission vector dimension mismatch")
        if epoch is not None and parent.epoch != epoch:
            raise OperationAdmissionError("nested admission epoch mismatch")
        if supervisor is not None and parent.supervisor is not supervisor:
            raise OperationAdmissionError("nested admission supervisor mismatch")
        if (
            embedding_generation is not None
            and parent.embedding_generation != embedding_generation
        ):
            raise OperationAdmissionError("nested admission generation mismatch")
        if cache_exclusive and not parent.cache_exclusive:
            raise OperationAdmissionError("nested cache lease upgrade is forbidden")
        if exclusive and not parent.exclusive:
            raise OperationAdmissionError("nested graph lease upgrade is forbidden")
        yield parent
        return

    end = time.monotonic() + max(0.0, deadline)
    while True:
        stack = ExitStack()
        owner: AdmissionLeaseOwner | None = None
        try:
            graph_lease = None
            if graph is not None:
                graph_lease = stack.enter_context(
                    try_maintenance_lock(graph) if exclusive else try_shared_lock(graph)
                )
                if graph_lease is None:
                    pass
                else:
                    cache_lease = None
                    if cache is not None:
                        cache_lease = stack.enter_context(
                            try_maintenance_lock(cache)
                            if cache_exclusive
                            else try_shared_lock(cache)
                        )
                    if cache is None or cache_lease is not None:
                        owner = AdmissionLeaseOwner(stack)
                        token_value = OperationAdmission(
                            graph,
                            cache,
                            model,
                            embedding_dim,
                            vec_dim,
                            epoch,
                            supervisor,
                            embedding_generation,
                            graph_lease,
                            cache_lease,
                            exclusive,
                            cache_exclusive,
                            owner,
                        )
                        token = _CURRENT.set(token_value)
                        try:
                            yield token_value
                        finally:
                            _CURRENT.reset(token)
                            if not owner.transferred:
                                owner.close()
                        return
            else:
                assert cache is not None
                cache_lease = stack.enter_context(
                    try_maintenance_lock(cache)
                    if cache_exclusive or exclusive
                    else try_shared_lock(cache)
                )
                if cache_lease is not None:
                    owner = AdmissionLeaseOwner(stack)
                    token_value = OperationAdmission(
                        None,
                        cache,
                        model,
                        embedding_dim,
                        vec_dim,
                        epoch,
                        supervisor,
                        embedding_generation,
                        None,
                        cache_lease,
                        False,
                        cache_exclusive or exclusive,
                        owner,
                    )
                    token = _CURRENT.set(token_value)
                    try:
                        yield token_value
                    finally:
                        _CURRENT.reset(token)
                        if not owner.transferred:
                            owner.close()
                    return
        except BaseException:
            if owner is None or not owner.transferred:
                stack.close()
            raise
        finally:
            if owner is None:
                stack.close()
        if time.monotonic() >= end:
            raise OperationAdmissionError("operation admission deadline exceeded")
        time.sleep(min(0.01, max(0.0, end - time.monotonic())))
