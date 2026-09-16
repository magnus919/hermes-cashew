"""Hermes boundary for the upstream Cashew sleep cycle.

The consolidation algorithm belongs to ``cashew-brain``.  This module only
owns the Hermes concerns that upstream cannot know about: profile admission,
the stable maintenance lease, journal safety, and the configured embedding
worker.  The selected upstream commit is the implementation under test.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import sqlite3
from typing import Any, Iterator, cast

from core.sleep import run_sleep_cycle as _upstream_run_sleep_cycle

from .admission import OperationAdmissionError, current_admission
from .locking import (
    MaintenanceLockAcquisitionError,
    guard_sqlite_journal,
    try_maintenance_lock,
)

logger = logging.getLogger(__name__)


def _owned_admission(db_path: str) -> bool:
    admission = current_admission()
    if admission is None:
        return False
    if admission.graph_path != pathlib.Path(db_path).resolve(strict=False):
        raise OperationAdmissionError("sleep admission graph identity mismatch")
    if not admission.exclusive:
        raise OperationAdmissionError("sleep requires an exclusive graph admission")
    if admission.cache_path is not None and admission.cache_lease is None:
        raise OperationAdmissionError("sleep admission cache lease is missing")
    return True


@contextlib.contextmanager
def _maintenance_lease(db_path: str) -> Iterator[None]:
    """Reuse an outer admission or acquire the standalone graph lease."""
    if _owned_admission(db_path):
        yield
        return
    with try_maintenance_lock(db_path) as lock_fd:
        if lock_fd is None:
            raise MaintenanceLockAcquisitionError(db_path)
        yield


def run_sleep_cycle(
    db_path: str,
    limit: int | None = 2_000,
    max_edges: int = 100_000,
    model_fn: Any = None,
    background_dream: bool = False,
    embedding_model: str | None = "thenlper/gte-large",
    embedding_device: str = "cpu",
    embedding_client: Any = None,
    *,
    cross_source_only: bool = False,
    orphan_limit: int | None = None,
    orphan_batch_size: int = 10,
) -> dict[str, Any]:
    """Run the pinned upstream cycle under Hermes coordination.

    ``embedding_device`` remains part of the adapter contract so the flat and
    package cron layouts can pass their configured profile unchanged. Device
    isolation is implemented by ``embedding_client``; upstream receives the
    already-owned worker and never loads a model in the cron process.
    """
    del embedding_device  # Device selection belongs to the Hermes worker.
    if background_dream:
        # The upstream daemon outlives this function, so it cannot retain the
        # Hermes file lease. Scheduled Hermes work is synchronous by design.
        logger.warning("sleep: background dream requires a retained Hermes lease")
        return {"status": "rejected", "error": "background_dream_unsupported"}
    try:
        with _maintenance_lease(db_path):
            # The lease/admission is established before this probe. Preserve
            # the journal selected during profile bootstrap; upstream's
            # ``manage`` mode could otherwise toggle WAL during a concurrent
            # Hermes session.
            with sqlite3.connect(db_path) as conn:
                guard_sqlite_journal(conn)
            expected_dimension = getattr(embedding_client, "dimension", None)
            if expected_dimension is None:
                expected_dimension = getattr(embedding_client, "dim", None)
            admission = current_admission()
            if admission is not None and embedding_client is not None:
                if admission.model not in (None, embedding_model):
                    raise OperationAdmissionError("sleep admission model identity mismatch")
                if admission.embedding_dim not in (None, expected_dimension):
                    raise OperationAdmissionError(
                        "sleep admission dimension identity mismatch"
                    )
            # Upstream treats a model without its embedding client as a
            # malformed partial contract. A no-client sleep still performs
            # graph maintenance and simply skips orphan repair.
            effective_embedding_model = (
                embedding_model if embedding_client is not None else None
            )
            return cast(dict[str, Any], _upstream_run_sleep_cycle(
                db_path=db_path,
                limit=limit,
                model_fn=model_fn,
                background_dream=background_dream,
                max_edges=max_edges,
                cross_source_only=cross_source_only,
                embedding_client=embedding_client,
                embedding_model=effective_embedding_model,
                expected_dimension=expected_dimension,
                journal_policy="preserve",
                orphan_limit=orphan_limit,
                orphan_batch_size=orphan_batch_size,
            ))
    except MaintenanceLockAcquisitionError:
        logger.info("sleep: maintenance lease is busy; skipping")
        return {}
    except Exception:
        logger.warning("sleep: upstream cycle failed", exc_info=True)
        return {}
