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
import re
import sqlite3
import sys
from contextlib import contextmanager
from typing import Iterator, TextIO, cast

logger = logging.getLogger(__name__)


class MaintenanceLockAcquisitionError(OSError):
    """Opening or acquiring the maintenance lock failed unexpectedly."""


class SQLiteWALUnsupportedError(RuntimeError):
    """An affected SQLite runtime cannot safely write an existing WAL DB."""


def lock_path_for_db(db_path: str | pathlib.Path) -> pathlib.Path:
    """Return the canonical, stable advisory-lock path for a configured DB."""
    canonical_db = pathlib.Path(db_path).resolve(strict=False)
    return pathlib.Path(f"{canonical_db}.sleep.lock")


def sqlite_wal_reset_vulnerable(version: tuple[int, int, int]) -> bool:
    """Classify the documented WAL reset defect and its backport fixes."""
    major, minor, patch = version
    if major != 3:
        return major < 3
    if minor < 44:
        return minor >= 7
    if minor == 44:
        return patch < 6
    if 45 <= minor < 50:
        return True
    if minor == 50:
        return patch < 7
    if minor == 51:
        return patch < 3
    return False


def guard_sqlite_journal(conn: object) -> str:
    """Verify journal safety without live mode churn."""
    import sqlite3

    if not hasattr(conn, "execute"):
        raise TypeError("guard_sqlite_journal requires a sqlite3 connection")
    mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    if sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info):
        if mode == "wal":
            raise SQLiteWALUnsupportedError(
                f"SQLite {sqlite3.sqlite_version} cannot safely write an existing WAL database"
            )
        if mode != "delete":
            mode = str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower()
        if mode != "delete":
            raise SQLiteWALUnsupportedError("could not establish DELETE journal mode")
    return mode


def bootstrap_sqlite_journal(conn: object) -> str:
    """Establish WAL only from exclusive, quiescent bootstrap."""
    import sqlite3

    if not hasattr(conn, "execute"):
        raise TypeError("bootstrap_sqlite_journal requires a sqlite3 connection")
    if sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info):
        return guard_sqlite_journal(conn)
    mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    if mode != "wal":
        mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
    if mode != "wal":
        raise SQLiteWALUnsupportedError("could not establish WAL journal mode")
    return mode


def open_readonly_verified(
    db_path: str | pathlib.Path,
) -> tuple[sqlite3.Connection, str]:
    """Open an existing DB through URI read-only and verify query_only."""
    path = pathlib.Path(db_path).resolve(strict=False)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        conn.close()
        raise SQLiteWALUnsupportedError("query-only verification failed")
    mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    return conn, mode


def verify_readonly_profile(conn: object) -> dict[str, str]:  # noqa: C901
    """Validate an affected WAL profile without executing an application write."""

    if not hasattr(conn, "execute"):
        raise TypeError("verify_readonly_profile requires a sqlite3 connection")
    conn = cast(sqlite3.Connection, conn)
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise SQLiteWALUnsupportedError("query-only verification failed")
    user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if user_version != 3:
        raise SQLiteWALUnsupportedError(
            f"read-only Cashew schema version {user_version} is unsupported"
        )
    integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0]).lower()
    if integrity != "ok":
        raise SQLiteWALUnsupportedError("read-only integrity check failed")
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    required = {
        "thought_nodes",
        "embeddings",
        "derivation_edges",
        "hermes_provider_meta",
    }
    if not required.issubset(tables):
        raise SQLiteWALUnsupportedError("read-only Cashew schema is incomplete")
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(thought_nodes)").fetchall()
    }
    required_keyword_columns = {
        "id",
        "content",
        "node_type",
        "domain",
        "timestamp",
        "access_count",
        "last_accessed",
        "source_file",
        "decayed",
        "metadata",
        "last_updated",
        "mood_state",
        "permanent",
        "tags",
        "referent_time",
    }
    if not required_keyword_columns.issubset(columns):
        raise SQLiteWALUnsupportedError("read-only keyword columns are incomplete")
    embedding_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(embeddings)").fetchall()
    }
    if not {"node_id", "vector", "model", "updated_at"}.issubset(embedding_columns):
        raise SQLiteWALUnsupportedError("read-only embedding columns are incomplete")
    edge_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(derivation_edges)").fetchall()
    }
    if not {
        "parent_id",
        "child_id",
        "weight",
        "reasoning",
        "timestamp",
    }.issubset(edge_columns):
        raise SQLiteWALUnsupportedError("read-only derivation columns are incomplete")

    meta = dict(
        conn.execute(
            "SELECT key, value FROM hermes_provider_meta WHERE key IN "
            "('embedding_model','embedding_dim','vec_dim','maintenance_epoch')"
        ).fetchall()
    )
    required_meta = {"embedding_model", "embedding_dim", "vec_dim", "maintenance_epoch"}
    if not required_meta.issubset(meta):
        raise SQLiteWALUnsupportedError("read-only provider identity is incomplete")
    try:
        expected_dim = int(meta["embedding_dim"])
        expected_vec_dim = int(meta["vec_dim"])
        int(meta["maintenance_epoch"])
    except (TypeError, ValueError) as exc:
        raise SQLiteWALUnsupportedError(
            "read-only provider identity is invalid"
        ) from exc
    if (
        expected_dim <= 0
        or expected_vec_dim <= 0
        or expected_dim != expected_vec_dim
        or not meta["embedding_model"]
    ):
        raise SQLiteWALUnsupportedError("read-only provider identity is invalid")
    rows = conn.execute(
        "SELECT node_id, model, LENGTH(vector) FROM embeddings"
    ).fetchall()
    for node_id, model, byte_length in rows:
        if (
            not node_id
            or not model
            or byte_length is None
            or int(byte_length) != expected_dim * 4
        ):
            raise SQLiteWALUnsupportedError(
                "read-only embedding identity is inconsistent"
            )
        if str(model) != meta["embedding_model"]:
            raise SQLiteWALUnsupportedError("read-only embedding model is stale")
    vec_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
    ).fetchone()
    if vec_table is not None:
        # Loading sqlite-vec into a query-only connection only registers the
        # virtual-table module; it does not alter the DB, WAL, or SHM files.
        try:
            conn.enable_load_extension(True)
            try:
                import sqlite_vec

                sqlite_vec.load(conn)
            except (ImportError, AttributeError):
                conn.load_extension("vec0")
        except Exception as exc:
            raise SQLiteWALUnsupportedError(
                "read-only vec parity could not be verified"
            ) from exc
        finally:
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
        vec_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(vec_embeddings)").fetchall()
        }
        vec_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='vec_embeddings'"
        ).fetchone()[0]
        declared_dimension_match = re.search(
            r"(?:float|int8)\s*\[\s*(\d+)\s*\]", str(vec_sql), re.IGNORECASE
        )
        if declared_dimension_match is None:
            raise SQLiteWALUnsupportedError("read-only vec dimension is unavailable")
        if int(declared_dimension_match.group(1)) != expected_vec_dim:
            raise SQLiteWALUnsupportedError("read-only vec dimension is inconsistent")
        # A virtual vec table has no PRAGMA columns on some sqlite-vec releases;
        # parity is checked when its read-only query is available.
        if "node_id" in vec_columns:
            try:
                vec_ids = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT node_id FROM vec_embeddings"
                    ).fetchall()
                }
                ordinary_ids = {str(row[0]) for row in rows}
                vec_lengths = {
                    str(row[0]): row[1]
                    for row in conn.execute(
                        "SELECT node_id, LENGTH(embedding) FROM vec_embeddings"
                    ).fetchall()
                }
            except Exception as exc:
                raise SQLiteWALUnsupportedError(
                    "read-only vec parity could not be verified"
                ) from exc
            if vec_ids != ordinary_ids:
                raise SQLiteWALUnsupportedError("read-only vec and ordinary IDs differ")
            if any(
                length is None or int(length) != expected_vec_dim * 4
                for length in vec_lengths.values()
            ):
                raise SQLiteWALUnsupportedError("read-only vec blob dimensions differ")

    source_id = str(conn.execute("SELECT sqlite_source_id()").fetchone()[0])
    return {
        "sqlite_source_id": source_id,
        "user_version": str(user_version),
        "provider_model": meta["embedding_model"],
        "provider_embedding_dim": str(expected_dim),
        "provider_vec_dim": str(expected_vec_dim),
        "provider_epoch": meta["maintenance_epoch"],
    }


def sqlite_journal_report(db_path: str | pathlib.Path) -> dict[str, str]:
    """Return linked-runtime and journal evidence without changing the DB."""
    import sqlite3

    path = pathlib.Path(db_path).resolve(strict=False)
    conn = sqlite3.connect(str(path))
    try:
        return {
            "sqlite_version": sqlite3.sqlite_version,
            "sqlite_source_id": str(
                conn.execute("SELECT sqlite_source_id()").fetchone()[0]
            ),
            "journal_mode": str(
                conn.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower(),
        }
    finally:
        conn.close()


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
    try:
        lock_fd = lock_path.open("a+")
    except OSError as exc:
        raise MaintenanceLockAcquisitionError(lock_path) from exc
    acquired = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise MaintenanceLockAcquisitionError(lock_path) from exc
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


@contextmanager
def try_shared_lock(db_path: str | pathlib.Path) -> Iterator[TextIO | None]:
    """Acquire the canonical stable file lease in shared mode."""
    lock_path = lock_path_for_db(db_path)
    try:
        lock_fd = lock_path.open("a+")
    except OSError as exc:
        raise MaintenanceLockAcquisitionError(lock_path) from exc
    acquired = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise MaintenanceLockAcquisitionError(lock_path) from exc
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
                    "failed to release shared lock %s", lock_path, exc_info=True
                )
            else:
                raise
        finally:
            try:
                lock_fd.close()
            except BaseException:
                if original_exception:
                    logger.warning(
                        "failed to close shared lock %s", lock_path, exc_info=True
                    )
                else:
                    raise
