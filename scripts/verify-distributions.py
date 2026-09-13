#!/usr/bin/env python3
"""Verify release artifacts contain only the intended source tree."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import PurePosixPath

REQUIRED = PurePosixPath("plugins/memory/cashew/__init__.py")
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "droid-wiki",
        "graphify-out",
        "cashew-pr137",
    }
)
FORBIDDEN_SUFFIXES = (".db", ".db-journal", ".db-shm", ".db-wal")


def _normalized(names: list[str], *, sdist: bool) -> set[PurePosixPath]:
    paths = [PurePosixPath(name) for name in names if name and not name.endswith("/")]
    if not sdist:
        return set(paths)
    if not paths:
        return set()
    roots = {path.parts[0] for path in paths if path.parts}
    if len(roots) != 1:
        raise ValueError(
            f"sdist must have one top-level directory, got {sorted(roots)}"
        )
    return {PurePosixPath(*path.parts[1:]) for path in paths if len(path.parts) > 1}


def _forbidden(paths: set[PurePosixPath]) -> list[str]:
    return sorted(
        path.as_posix()
        for path in paths
        if any(part in FORBIDDEN_PARTS for part in path.parts)
        or path.name.endswith(FORBIDDEN_SUFFIXES)
    )


def verify_distribution(path: PurePosixPath | str) -> None:
    """Raise ValueError when a wheel or sdist is incomplete or contaminated."""
    artifact = PurePosixPath(path)
    artifact_path = str(artifact)
    is_sdist = artifact_path.endswith((".tar.gz", ".tgz"))
    if artifact_path.endswith(".whl"):
        with zipfile.ZipFile(artifact_path) as archive:
            paths = _normalized(archive.namelist(), sdist=False)
            symlinks: list[str] = []
    elif is_sdist:
        with tarfile.open(artifact_path, "r:gz") as archive:
            members = archive.getmembers()
            paths = _normalized([member.name for member in members], sdist=True)
            symlinks = [
                member.name for member in members if member.issym() or member.islnk()
            ]
    else:
        raise ValueError(f"unsupported artifact: {artifact_path}")

    problems: list[str] = []
    if REQUIRED not in paths:
        problems.append(f"missing {REQUIRED}")
    forbidden = _forbidden(paths)
    if forbidden:
        problems.append(f"forbidden paths: {', '.join(forbidden)}")
    if symlinks:
        problems.append(f"symlink entries: {', '.join(symlinks)}")
    if problems:
        raise ValueError(f"{artifact_path}: {'; '.join(problems)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=PurePosixPath)
    args = parser.parse_args()
    for artifact in args.artifacts:
        verify_distribution(artifact)
        print(f"verified clean distribution: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
