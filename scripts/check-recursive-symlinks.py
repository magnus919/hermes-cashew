#!/usr/bin/env python3
"""Reject symlinks that make a recursive source-tree walk unsafe.

The checker is deliberately read-only.  It is intended to run before tools
such as mypy that follow directory symlinks, so a local development symlink
cannot turn a normal quality command into an opaque path explosion.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def find_recursive_symlinks(root: Path) -> list[tuple[Path, str]]:
    """Return directory links whose target is an ancestor of the link."""
    root = root.resolve()
    findings: list[tuple[Path, str]] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        for name in [*directories, *filenames]:
            link = Path(current, name)
            if not link.is_symlink():
                continue
            try:
                target = link.resolve(strict=False)
            except RuntimeError:
                findings.append((link, "cannot be resolved (symlink loop)"))
                continue
            if target.is_dir() and _is_relative_to(link.parent.resolve(), target):
                findings.append((link, f"resolves to ancestor {target}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    findings: list[tuple[Path, str]] = []
    for root in args.paths:
        if not root.is_dir():
            raise SystemExit(
                f"workspace path does not exist or is not a directory: {root}"
            )
        findings.extend(find_recursive_symlinks(root))

    if not findings:
        print("No recursive workspace symlinks found.")
        return 0

    print("Recursive symlink contamination detected:")
    for link, reason in findings:
        print(f"  - {link}: {reason}")
    print(
        "Remove or retarget the listed link, then rerun the quality command. "
        "This checker never removes files."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
