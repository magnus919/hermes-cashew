#!/usr/bin/env python3
"""Verify that cashew-brain is the selected source build, not PyPI 1.2.1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sqlite3
from contextlib import closing
from pathlib import Path

ARCHIVE_URL = (
    "https://github.com/rajkripal/cashew/archive/"
    "dd57ef029cf9a6dce0b8145d335a55202dd1bac4.tar.gz"
)
SESSION_SHA256 = "0ce60cc63adf4fb7136581aee722bb10e9a344e556b6fb98f4d46855d53c36cd"
MINIMUM_SQLITE = (3, 35, 0)


def main() -> None:
    distribution = importlib.metadata.distribution("cashew-brain")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise SystemExit(
            "cashew-brain has no direct_url.json; version 1.2.1 does not prove "
            "the selected source is installed"
        )
    actual_url = json.loads(direct_url_text)["url"]
    if actual_url != ARCHIVE_URL:
        raise SystemExit(f"unexpected cashew-brain source: {actual_url}")

    session_path = Path(distribution.locate_file("core/session.py"))
    actual_hash = hashlib.sha256(session_path.read_bytes()).hexdigest()
    if actual_hash != SESSION_SHA256:
        raise SystemExit(
            f"unexpected core/session.py SHA-256: {actual_hash}; "
            f"expected {SESSION_SHA256}"
        )
    print(f"cashew-brain source verified: {actual_url}")
    print(f"core/session.py SHA-256: {actual_hash}")

    with closing(sqlite3.connect(":memory:")) as connection:
        source_id = connection.execute("SELECT sqlite_source_id()").fetchone()[0]
    print(f"SQLite version: {sqlite3.sqlite_version}")
    print(f"SQLite source ID: {source_id}")
    if sqlite3.sqlite_version_info < MINIMUM_SQLITE:
        minimum = ".".join(str(part) for part in MINIMUM_SQLITE)
        raise SystemExit(
            f"SQLite {sqlite3.sqlite_version} is unsupported by this baseline; "
            f"SQLite >= {minimum} is required for upstream legacy v1 migrations"
        )


if __name__ == "__main__":
    main()
