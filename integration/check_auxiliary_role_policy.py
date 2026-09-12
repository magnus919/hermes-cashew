#!/usr/bin/env python3
"""Check Cashew's explicit auxiliary-role policy against a pinned Hermes tree.

Run with the pinned Hermes interpreter and source documented in integration/README.md:

    HERMES_PINNED_SOURCE=/tmp/hermes-agent-<revision> \
      /tmp/hermes-agent-<revision>-venv/bin/python \
      integration/check_auxiliary_role_policy.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def _load_config(plugin_source: Path):
    spec = importlib.util.spec_from_file_location(
        "cashew_pinned_role_policy", plugin_source / "plugins/memory/cashew/config.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check(mapping: str, expected: bool, config_module, home: Path) -> None:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    (home / "config.yaml").write_text(mapping, encoding="utf-8")
    token = set_hermes_home_override(home)
    try:
        assert config_module._raw_role_mapping("memory") is expected
    finally:
        reset_hermes_home_override(token)


def main() -> None:
    source = Path(os.environ["HERMES_PINNED_SOURCE"])
    if not (source / "agent/auxiliary_client.py").is_file():
        raise SystemExit("HERMES_PINNED_SOURCE must be a Hermes source tree")
    plugin = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))
    config_module = _load_config(plugin)
    with tempfile.TemporaryDirectory(prefix="cashew-role-policy-") as raw_home:
        home = Path(raw_home)
        _check("auxiliary:\n  memory:\n    provider: auto\n", True, config_module, home)
        _check(
            "auxiliary:\n  memory:\n    provider: auto\n    base_url: null\n"
            "    api_key: null\n    key_env: null\n",
            True,
            config_module,
            home,
        )
        _check("auxiliary:\n  memory:\n    model: null\n", False, config_module, home)
        _check(
            "auxiliary:\n  memory:\n    provider: auto\n    base_url: 42\n",
            False,
            config_module,
            home,
        )
    print("PASS pinned Hermes auxiliary role policy")


if __name__ == "__main__":
    main()
