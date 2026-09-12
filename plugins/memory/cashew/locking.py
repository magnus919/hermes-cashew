"""Stable, nonblocking advisory locks for Cashew maintenance operations.

The lock file names the configured database and is intentionally never removed:
``flock`` ownership belongs to the open descriptor, so unlinking a file while a
holder is alive would let a new pathname refer to a different inode.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import pathlib
import sys
from contextlib import contextmanager
from typing import Iterator, TextIO

logger = logging.getLogger(__name__)


def lock_path_for_db(db_path: str | pathlib.Path) -> pathlib.Path:
    """Return the canonical, stable advisory-lock path for a configured DB."""
    canonical_db = pathlib.Path(db_path).resolve(strict=False)
    return pathlib.Path(f"{canonical_db}.sleep.lock")


@contextmanager
def try_maintenance_lock(db_path: str | pathlib.Path) -> Iterator[TextIO | None]:
    """Yield a nonblocking lock descriptor, or ``None`` when it is contended.

    Unexpected open or flock errors remain errors: only EAGAIN/EWOULDBLOCK means
    another participant currently owns the lock. Every successfully opened
    descriptor is closed, including when unlock itself fails. If the protected
    body already raised, release errors are logged without masking that original
    exception.
    """
    lock_path = lock_path_for_db(db_path)
    lock_fd = lock_path.open("a+")
    acquired = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            yield None
            return
        acquired = True
        yield lock_fd
    finally:
        original_exception = sys.exc_info()[0] is not None
        try:
            if acquired:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except BaseException:
            if original_exception:
                logger.warning(
                    "failed to release maintenance lock %s", lock_path, exc_info=True
                )
            else:
                raise
        finally:
            try:
                lock_fd.close()
            except BaseException:
                if original_exception:
                    logger.warning(
                        "failed to close maintenance lock %s", lock_path, exc_info=True
                    )
                else:
                    raise
