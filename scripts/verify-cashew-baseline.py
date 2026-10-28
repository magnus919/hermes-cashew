#!/usr/bin/env python3
"""Verify the bundled Cashew source and SQLite migration capability."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

MINIMUM_SQLITE = (3, 35, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installed",
        action="store_true",
        help="audit the installed wheel/sdist rather than the adjacent flat source",
    )
    args = parser.parse_args()
    if args.installed:
        distribution = importlib.metadata.distribution("hermes-cashew")
        vendor = Path(distribution.locate_file("plugins/memory/cashew/_vendor"))
    else:
        vendor = Path(__file__).resolve().parents[1] / "plugins/memory/cashew/_vendor"
    print(f"auditing bundled source: {vendor}")

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
