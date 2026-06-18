#!/usr/bin/env python3
"""Detect dead feature flags — flags defined in DEFAULTS but never checked.

Scans the _features dict in plugins/memory/cashew/config.py, then greps
the plugin source for is_feature_enabled() calls referencing each flag.
Flags with zero call sites are reported as dead.

Exit codes: 0 = no dead flags, 1 = dead flags found.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

CONFIG_FILE = Path("plugins/memory/cashew/config.py")
SOURCE_DIR = Path("plugins/memory/cashew")


def _parse_feature_flags(config_path: Path) -> dict[str, bool]:
    """Extract _features dict keys from DEFAULTS using AST parsing.

    DEFAULTS is a single dict literal with string keys and a type
    annotation (``DEFAULTS: dict[str, Any] = {...}``), so it appears
    as ``ast.AnnAssign`` in the AST. We find the ``_features`` key
    within it and return its inner dict keys.
    """
    tree = ast.parse(config_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULTS":
                    if isinstance(node.value, ast.Dict):
                        for key_node, val_node in zip(
                            node.value.keys, node.value.values
                        ):
                            if (
                                isinstance(key_node, ast.Constant)
                                and key_node.value == "_features"
                                and isinstance(val_node, ast.Dict)
                            ):
                                flags: dict[str, bool] = {}
                                for fk, fv in zip(val_node.keys, val_node.values):
                                    if isinstance(fk, ast.Constant):
                                        flags[fk.value] = (
                                            fv.value
                                            if isinstance(fv, ast.Constant)
                                            else False
                                        )
                                return flags
    return {}


def _find_flag_usages(flags: dict[str, bool], source_dir: Path) -> dict[str, int]:
    """Count is_feature_enabled() calls for each flag name in source files."""
    usage: dict[str, int] = {flag: 0 for flag in flags}
    for py_file in sorted(source_dir.glob("*.py")):
        text = py_file.read_text()
        for flag in flags:
            # Match: is_feature_enabled(..., "...")
            pattern = (
                rf'is_feature_enabled\s*\(\s*\S+\s*,\s*["\']({re.escape(flag)})["\']'
            )
            usage[flag] += len(re.findall(pattern, text))
    return usage


def main() -> int:
    if not CONFIG_FILE.exists():
        print(f"ERROR: config file not found: {CONFIG_FILE}")
        return 1

    flags = _parse_feature_flags(CONFIG_FILE)
    if not flags:
        print("No _features found in DEFAULTS.")
        return 0

    usage = _find_flag_usages(flags, SOURCE_DIR)
    dead_flags = [flag for flag, count in usage.items() if count == 0]
    live_flags = [flag for flag, count in usage.items() if count > 0]

    print(f"Feature flags defined: {len(flags)}")
    for flag in live_flags:
        print(f"  LIVE: {flag} ({usage[flag]} call site(s))")
    for flag in dead_flags:
        print(
            f"  DEAD: {flag} — defined in DEFAULTS but never checked via is_feature_enabled()"
        )

    if dead_flags:
        print(
            f"\n{len(dead_flags)} dead flag(s) found. Remove them from DEFAULTS or wire into a code path."
        )
        return 1

    print("\nNo dead flags detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
