"""Installation surfaces must resolve the same runtime dependency contracts."""

from __future__ import annotations

from pathlib import Path

import tomllib
import yaml


def test_plugin_manifests_match_project_runtime_dependencies() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    expected = set(project["project"]["dependencies"])

    for relative in ("plugin.yaml", "plugins/memory/cashew/plugin.yaml"):
        manifest = yaml.safe_load((root / relative).read_text())
        assert set(manifest["pip_dependencies"]) == expected
