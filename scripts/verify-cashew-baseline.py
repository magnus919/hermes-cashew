#!/usr/bin/env python3
"""Verify the bundled Cashew source and SQLite migration capability."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

MINIMUM_SQLITE = (3, 35, 0)


def main() -> None:
    distribution = importlib.metadata.distribution("hermes-cashew")
    vendor = Path(distribution.locate_file("plugins/memory/cashew/_vendor"))
    # Editable installs need the source package directory; use the selected
    # provider package only when metadata does not locate regular wheel files.
    if not vendor.is_dir():
        import plugins.memory.cashew

        vendor = Path(plugins.memory.cashew.__file__).resolve().parent / "_vendor"

    spec = importlib.util.spec_from_file_location(
        "_cashew_provenance", vendor / "verify.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    manifest = verifier.verify(vendor)
    print(f"bundled Cashew source verified: {manifest['commit']}")
    print(f"upstream archive SHA-256: {manifest['archive_sha256']}")

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
