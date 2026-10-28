"""Behavior and provenance contracts for the selected Cashew source baseline."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from plugins.memory.cashew._vendor.core import integrity  # noqa: E402
from plugins.memory.cashew._vendor.core.db import (  # noqa: E402, I001
    NODE_COLUMNS,
    ensure_schema,
    schema_version,
)

CASHEW_ARCHIVE_URL = (
    "https://github.com/magnus919/true/archive/"
    "fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz"
)
CASHEW_SESSION_SHA256 = (
    "0ce60cc63adf4fb7136581aee722bb10e9a344e556b6fb98f4d46855d53c36cd"
)
CASHEW_INTEGRITY_SHA256 = (
    "80e928c4a073aadb9340095a27b65f03512393d9c1e03cfd025e9aa70d99b9ba"
)
CASHEW_ARCHIVE_SHA256 = (
    "23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3"
)
MINIMUM_SQLITE = (3, 35, 0)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_installed_cashew_has_selected_source_provenance() -> None:
    from plugins.memory.cashew._vendor.verify import verify

    assert verify()["commit"] == "fcb4919ac37144bfbeb822eaafc668a4bdceb791"


def test_installed_cashew_exposes_connection_owned_integrity_api() -> None:
    assert callable(integrity.inspect_integrity)
    assert callable(integrity.repair_integrity)


def test_sqlite_supports_selected_upstream_legacy_migrations() -> None:
    assert sqlite3.sqlite_version_info >= MINIMUM_SQLITE, (
        f"SQLite {sqlite3.sqlite_version} is unsupported by this baseline; "
        "upstream legacy v1 migration requires SQLite >= 3.35"
    )


def test_current_schema_check_does_not_wait_for_held_writer(tmp_path: Path) -> None:
    db_path = tmp_path / "brain.db"
    ensure_schema(str(db_path))

    with closing(sqlite3.connect(db_path)) as setup:
        assert setup.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"

    writer = sqlite3.connect(db_path, timeout=0)
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.perf_counter()
        ensure_schema(str(db_path))
        elapsed = time.perf_counter() - started
    finally:
        writer.rollback()
        writer.close()

    assert elapsed < 0.5, f"current schema check waited {elapsed:.3f}s for a writer"


def test_selected_upstream_creates_current_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    ensure_schema(str(db_path))

    with closing(sqlite3.connect(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"thought_nodes", "derivation_edges", "embeddings", "hotspots"} <= (
            tables
        )
        assert set(NODE_COLUMNS) <= _columns(conn, "thought_nodes")
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema_version()


def test_selected_upstream_migrates_legacy_schema_without_data_loss(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "CREATE TABLE thought_nodes ("
            "id TEXT PRIMARY KEY, content TEXT NOT NULL, node_type TEXT NOT NULL, "
            "timestamp TEXT, confidence REAL)"
        )
        conn.execute(
            "INSERT INTO thought_nodes VALUES (?, ?, ?, ?, ?)",
            ("legacy-1", "preserve me", "seed", "2024-01-01T00:00:00Z", 0.9),
        )
        conn.commit()

    ensure_schema(str(db_path))

    with closing(sqlite3.connect(db_path)) as conn:
        assert set(NODE_COLUMNS) <= _columns(conn, "thought_nodes")
        assert "confidence" not in _columns(conn, "thought_nodes")
        assert conn.execute(
            "SELECT id, content, node_type, timestamp, permanent "
            "FROM thought_nodes WHERE id='legacy-1'"
        ).fetchone() == (
            "legacy-1",
            "preserve me",
            "seed",
            "2024-01-01T00:00:00Z",
            1,
        )
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema_version()
