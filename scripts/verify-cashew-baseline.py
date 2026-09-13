#!/usr/bin/env python3
"""Verify that cashew-brain is the selected source build, not PyPI 1.2.1."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

ARCHIVE_URL = (
    "https://github.com/rajkripal/cashew/archive/"
    "ac090ce75ffd2e97dac257cee9430628c68aa241.tar.gz"
)
ARCHIVE_SHA256 = "0777dcb89bde8e0d6103786358c93c0ad3fd4ffb7f8a210194c37b211ae4c28b"
SESSION_SHA256 = "0ce60cc63adf4fb7136581aee722bb10e9a344e556b6fb98f4d46855d53c36cd"
MINIMUM_SQLITE = (3, 35, 0)
_SHA256_RE = re.compile(r"^(?:sha256[=:])?([0-9a-fA-F]{64})$")


def _normalise_sha256(value: object) -> str | None:
    """Return a bare SHA-256 digest from a PEP 610 hash value."""
    if not isinstance(value, str):
        return None
    match = _SHA256_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


def _direct_url_sha256(payload: object) -> str | None:
    """Extract a PEP 610 archive digest, accepting pip's hash shapes.

    pip currently records both ``archive_info.hash`` as ``sha256=...`` and
    ``archive_info.hashes.sha256``. uv may emit an empty ``archive_info`` for
    a locked direct URL, so callers use the lockfile as a verified fallback in
    that installer-specific case.
    """
    if not isinstance(payload, dict):
        return None
    archive_info = payload.get("archive_info")
    if not isinstance(archive_info, dict):
        return None

    candidates: list[str] = []
    raw_hash = archive_info.get("hash")
    digest = _normalise_sha256(raw_hash)
    if raw_hash is not None and digest is None:
        raise ValueError("PEP 610 archive_info.hash is not a SHA-256 value")
    if digest is not None:
        candidates.append(digest)
    hashes = archive_info.get("hashes")
    if isinstance(hashes, dict):
        raw_sha256 = hashes.get("sha256")
        digest = _normalise_sha256(raw_sha256)
        if raw_sha256 is not None and digest is None:
            raise ValueError("PEP 610 archive_info.hashes.sha256 is invalid")
        if digest is not None:
            candidates.append(digest)
    if not candidates:
        if archive_info:
            raise ValueError("PEP 610 archive_info has no SHA-256 value")
        return None
    if any(candidate != candidates[0] for candidate in candidates[1:]):
        raise ValueError("PEP 610 archive_info contains conflicting SHA-256 values")
    return candidates[0]


def _locked_archive_sha256() -> str | None:
    """Read the Cashew archive digest from uv.lock for uv's empty metadata."""
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    try:
        text = lock_path.read_text(encoding="utf-8")
    except OSError:
        return None
    pattern = re.compile(
        r'name = "cashew-brain".*?'
        rf'source = \{{ url = "{re.escape(ARCHIVE_URL)}" \}}.*?'
        r'sdist = \{ hash = "sha256:([0-9a-fA-F]{64})" \}',
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).lower() if match else None


def main() -> None:
    distribution = importlib.metadata.distribution("cashew-brain")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise SystemExit(
            "cashew-brain has no direct_url.json; version 1.2.1 does not prove "
            "the selected source is installed"
        )
    direct_url = json.loads(direct_url_text)
    actual_url = direct_url["url"]
    if actual_url != ARCHIVE_URL:
        raise SystemExit(f"unexpected cashew-brain source: {actual_url}")
    try:
        archive_hash = _direct_url_sha256(direct_url)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if archive_hash is not None:
        if archive_hash != ARCHIVE_SHA256:
            raise SystemExit(
                f"unexpected Cashew archive SHA-256: {archive_hash}; "
                f"expected {ARCHIVE_SHA256}"
            )
        print(f"cashew-brain archive SHA-256 verified: {archive_hash}")
    else:
        locked_hash = _locked_archive_sha256()
        if locked_hash != ARCHIVE_SHA256:
            raise SystemExit(
                "cashew-brain direct_url.json has no archive_info SHA-256 and "
                "uv.lock does not record the expected digest"
            )
        print(
            "cashew-brain archive SHA-256 verified from uv.lock; "
            "uv omitted archive_info.hash"
        )

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
