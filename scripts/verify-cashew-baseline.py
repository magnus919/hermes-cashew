#!/usr/bin/env python3
"""Verify that cashew-brain is the selected immutable source build."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sqlite3
from contextlib import closing
from pathlib import Path

ARCHIVE_URL = (
    "https://github.com/magnus919/true/archive/"
    "fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz"
)
ARCHIVE_SHA256 = (
    "23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3"
)
SESSION_SHA256 = "0ce60cc63adf4fb7136581aee722bb10e9a344e556b6fb98f4d46855d53c36cd"
MINIMUM_SQLITE = (3, 35, 0)


def main() -> None:
    distribution = importlib.metadata.distribution("cashew-brain")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise SystemExit("cashew-brain has no direct_url.json; source is unverified")
    direct_url = json.loads(direct_url_text)
    actual_url = direct_url["url"]
    if actual_url != ARCHIVE_URL:
        raise SystemExit(f"unexpected cashew-brain source: {actual_url}")
    expected_archive_hash = f"sha256={ARCHIVE_SHA256}"
    actual_archive_hash = direct_url.get("archive_info", {}).get("hash")
    # uv and pip may omit archive_info when installing a direct URL.  The
    # source requirement and frozen lockfile enforce this digest at install;
    # reject it only when the installer reports a conflicting value.
    if actual_archive_hash is not None and actual_archive_hash != expected_archive_hash:
        raise SystemExit(
            f"unexpected cashew-brain archive hash: {actual_archive_hash}; "
            f"expected {expected_archive_hash}"
        )

    session_path = Path(distribution.locate_file("core/session.py"))
    actual_session_hash = hashlib.sha256(session_path.read_bytes()).hexdigest()
    if actual_session_hash != SESSION_SHA256:
        raise SystemExit(
            f"unexpected core/session.py SHA-256: {actual_session_hash}; "
            f"expected {SESSION_SHA256}"
        )
    print(f"cashew-brain source verified: {actual_url}")
    print(f"cashew-brain archive SHA-256: {ARCHIVE_SHA256}")
    print(f"core/session.py SHA-256: {actual_session_hash}")

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
