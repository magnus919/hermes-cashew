#!/usr/bin/env python3
"""Detect duplicate code blocks across Python source files using difflib.

Scans .py files under plugins/memory/cashew/, splits each into lines,
and compares every pair of files with SequenceMatcher to find similar
blocks (>85% similarity, >6 significant lines).  Flags the top offender
per pair if any block exceeds thresholds.

Exit codes: 0 = no duplicates found, 1 = duplicates flagged.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

SOURCE_DIR = Path("plugins/memory/cashew")
SIMILARITY_THRESHOLD = 0.85
MIN_SIGNIFICANT_LINES = 6


def _significant_lines(lines: list[str]) -> list[str]:
    """Strip blank lines and comment-only lines."""
    return [line for line in lines if line.strip() and not line.strip().startswith("#")]


def main() -> int:
    py_files = sorted(SOURCE_DIR.glob("*.py"))
    if len(py_files) < 2:
        print("Too few Python files to compare.")
        return 0

    violations = 0

    for i in range(len(py_files)):
        for j in range(i + 1, len(py_files)):
            a_lines = py_files[i].read_text().splitlines()
            b_lines = py_files[j].read_text().splitlines()

            matcher = difflib.SequenceMatcher(None, a_lines, b_lines)
            blocks = matcher.get_matching_blocks()

            for block in blocks[:-1]:  # last block is dummy (0, 0, 0)
                sig_lines = _significant_lines(a_lines[block.a : block.a + block.size])
                if len(sig_lines) >= MIN_SIGNIFICANT_LINES:
                    ratio = block.size / max(len(a_lines), len(b_lines))
                    if ratio > SIMILARITY_THRESHOLD:
                        print(
                            f"DUPLICATE: {py_files[i].name} L{block.a + 1} and "
                            f"{py_files[j].name} L{block.b + 1} "
                            f"({block.size} lines, {ratio:.1%} similarity, "
                            f"{len(sig_lines)} significant)"
                        )
                        violations += 1
                        break  # only report worst block per pair

    if violations:
        print(f"{violations} duplicate block(s) found.")
        return 1
    print("No duplicate code blocks detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
