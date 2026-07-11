"""Regression checks for release and contributor documentation contracts."""

from __future__ import annotations

import re
from pathlib import Path

from plugins.memory.cashew.config import DEFAULTS, get_config_schema

ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_syncs_both_manifests_from_pyproject() -> None:
    release = (ROOT / ".github/workflows/release.yml").read_text()
    assert re.search(r"VERSION=.*\^version = .*pyproject\.toml", release, re.MULTILINE)
    assert "for f in plugin.yaml plugins/memory/cashew/plugin.yaml" in release
    assert 'sed -i "s/^version: .*/version: ${VERSION}/" "$f"' in release


def test_documented_config_surface_matches_runtime() -> None:
    readme = (ROOT / "README.md").read_text()
    agents = (ROOT / "AGENTS.md").read_text()
    assert len(DEFAULTS) == 37
    assert len(get_config_schema()) == 16
    assert "37 persisted configuration fields" in readme
    assert "16 fields backed by current" in readme
    assert "37 compatibility defaults" in agents
    assert "16-field runtime-backed setup schema" in agents


def test_contributor_docs_match_threading_and_release_workflows() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    claude = (ROOT / "CLAUDE.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    release = (ROOT / ".github/workflows/release.yml").read_text()

    assert "non-daemon" not in agents + claude + contributing
    assert "single daemon worker" in agents
    assert "single **daemon** worker" in contributing
    assert "TestPyPI" not in release
    assert "There is no TestPyPI publication job" in contributing
    assert "not committed" in agents
    assert "not committed" in contributing
    assert "single source of truth" in claude
