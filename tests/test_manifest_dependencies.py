"""Installation surfaces must resolve the same runtime dependency contracts."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 CI floor
    import tomli as tomllib
import yaml


def test_plugin_manifests_match_project_runtime_dependencies() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    expected = set(project["project"]["dependencies"])

    for relative in ("plugin.yaml", "plugins/memory/cashew/plugin.yaml"):
        manifest = yaml.safe_load((root / relative).read_text())
        assert set(manifest["pip_dependencies"]) == expected

    assert all(" @ " not in requirement for requirement in expected)


def test_lock_has_no_separate_cashew_distribution() -> None:
    root = Path(__file__).parents[1]
    lock = tomllib.loads((root / "uv.lock").read_text())
    assert not any(p["name"] == "cashew-brain" for p in lock["package"])


def test_embedding_runtime_dependencies_are_direct_project_requirements() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependency_names = {
        re.match(r"[A-Za-z0-9_.-]+", dependency).group()
        for dependency in project["project"]["dependencies"]
    }

    assert {"numpy", "sentence-transformers"} <= dependency_names
    assert {"scikit-learn", "PyYAML"} <= dependency_names
