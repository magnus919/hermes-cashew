"""Regression checks for release and contributor documentation contracts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from plugins.memory.cashew.config import DEFAULTS, get_config_schema

ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_syncs_both_manifests_from_pyproject() -> None:
    release = (ROOT / ".github/workflows/release.yml").read_text()
    assert re.search(r"VERSION=.*\^version = .*pyproject\.toml", release, re.MULTILINE)
    assert "for f in plugin.yaml plugins/memory/cashew/plugin.yaml" in release
    assert 'sed -i "s/^version: .*/version: ${VERSION}/" "$f"' in release


def test_release_workflow_blocks_direct_dependencies_before_tests() -> None:
    release_path = ROOT / ".github/workflows/release.yml"
    release_text = release_path.read_text()
    workflow = yaml.safe_load(release_text)
    jobs = workflow["jobs"]

    assert "direct URL dependencies are not accepted by PyPI" in release_text
    assert "release-policy" in jobs
    assert jobs["test"]["needs"] == "release-policy"
    assert jobs["build"]["needs"] == "test"


def test_documented_config_surface_matches_runtime() -> None:
    readme = (ROOT / "README.md").read_text()
    plugin_readme = (ROOT / "plugins/memory/cashew/README.md").read_text()
    agents = (ROOT / "AGENTS.md").read_text()
    assert len(DEFAULTS) == 17
    assert len(get_config_schema()) == 17
    assert "all 17 persisted configuration fields" in readme
    assert "backed by current runtime behavior" in readme
    assert "17-field runtime-backed setup schema" in agents
    for documentation in (readme, plugin_readme):
        assert "Starting with v0.11.0" in documentation
        assert "removed keys are ignored and pruned" in documentation


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


def test_security_policy_provides_private_reporting_path() -> None:
    policy = (ROOT / "SECURITY.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()

    assert "security/advisories/new" in policy
    assert "latest published" in policy
    assert "Older releases are not supported" in policy
    assert "five business days" in policy
    assert "[security policy](./SECURITY.md)" in contributing
    assert "See the Security section below" not in contributing


def test_ci_and_contributor_docs_use_frozen_uv_lock() -> None:
    gitignore = (ROOT / ".gitignore").read_text()
    tests_workflow = (ROOT / ".github/workflows/tests.yml").read_text()
    release_workflow = (ROOT / ".github/workflows/release.yml").read_text()
    readme = (ROOT / "README.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()

    assert (ROOT / "uv.lock").is_file()
    ignore_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "uv.lock"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert ignore_check.returncode == 1, ignore_check.stderr or gitignore
    for workflow in (tests_workflow, release_workflow):
        assert 'python -m pip install "uv==0.11.18"' in workflow
        assert "uv sync --frozen --extra dev" in workflow
        assert ".venv/bin/pytest" in workflow
        assert ".venv/bin/pytest -xvs" not in workflow
        assert "uv pip install --system" not in workflow
        assert "scripts/verify-cashew-baseline.py" in workflow
        assert 'python-version: "3.11"' not in workflow
    assert "python-version: ${{ matrix.setup }}" in tests_workflow
    assert 'python-version: "3.12"' in release_workflow
    assert "uv sync --frozen --extra dev" in readme
    assert "uv sync --frozen --extra dev" in contributing
    assert (
        "uv run --frozen --extra dev python scripts/check-recursive-symlinks.py"
        in contributing
    )
    assert "--cov-fail-under=75" in tests_workflow
    tests_config = yaml.safe_load(tests_workflow)
    for job in tests_config["jobs"].values():
        assert "${{ runner.temp }}" not in " ".join(job.get("env", {}).values())
    assert 'WHEEL_SMOKE_VENV="$RUNNER_TEMP/' in tests_workflow
    assert 'FLAT_SMOKE_VENV="$RUNNER_TEMP/' in tests_workflow
    assert 'SDIST_SMOKE_VENV="$RUNNER_TEMP/' in tests_workflow
    assert "scripts/check-recursive-symlinks.py" in tests_workflow
    assert "scripts/verify-distributions.py dist/*" in tests_workflow
    assert "scripts/smoke-flat-install.py" in tests_workflow


def test_ci_uses_scoped_managed_python_for_sqlite_baseline() -> None:
    expected_install = 'uv python install --managed-python "3.12.11"'
    expected_sync = 'uv sync --frozen --extra dev --managed-python --python "3.12.11"'
    expected_env = {
        "UV_PYTHON_INSTALL_DIR": "${{ runner.temp }}/uv-python",
        "UV_PYTHON_BIN_DIR": "${{ runner.temp }}/uv-python-bin",
    }

    tests = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    test_install = next(
        step
        for step in tests["jobs"]["test-python"]["steps"]
        if step.get("name") == "Install"
    )
    assert test_install["env"] == expected_env
    assert (
        'uv python install --managed-python "${{ matrix.python }}"'
        in test_install["run"]
    )
    assert (
        'uv sync --frozen --extra dev --managed-python --python "${{ matrix.python }}"'
        in test_install["run"]
    )

    managed_steps = (
        next(
            step
            for step in tests["jobs"]["wheel-smoke"]["steps"]
            if step.get("name") == "Build wheel"
        ),
        next(
            step
            for step in release["jobs"]["test"]["steps"]
            if step.get("name") == "Install"
        ),
        next(
            step
            for step in release["jobs"]["build"]["steps"]
            if step.get("name") == "Install build dependencies"
        ),
    )
    for step in managed_steps:
        assert step["env"] == expected_env
        assert expected_install in step["run"]
        assert expected_sync in step["run"]

    tests_text = (ROOT / ".github/workflows/tests.yml").read_text()
    assert '.venv/bin/python -m venv --clear "$WHEEL_SMOKE_VENV"' in tests_text
    assert '.venv/bin/python -m venv --clear "$FLAT_SMOKE_VENV"' in tests_text
    assert '.venv/bin/python -m venv --clear "$SDIST_SMOKE_VENV"' in tests_text


def test_ci_covers_declared_minimum_and_current_python() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    matrix = workflow["jobs"]["test-python"]["strategy"]["matrix"]
    assert matrix["include"] == [
        {"setup": "3.10", "python": "3.10.19"},
        {"setup": "3.12", "python": "3.12.11"},
    ]
    assert workflow["jobs"]["test-python"]["strategy"]["fail-fast"] is False
    assert (
        workflow["jobs"]["test-python"]["name"] == "Test (Python ${{ matrix.python }})"
    )
    assert workflow["jobs"]["test"]["name"] == "test"
    assert workflow["jobs"]["test"]["needs"] == "test-python"
    assert workflow["jobs"]["test"]["if"] == "always()"
    assert workflow["jobs"]["test"]["steps"][0]["env"]["MATRIX_RESULT"] == (
        "${{ needs['test-python'].result }}"
    )
    assert workflow["jobs"]["wheel-smoke"]["needs"] == "test"


def test_ci_runs_runtime_tests_on_both_versions_and_quality_once() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    steps = workflow["jobs"]["test-python"]["steps"]
    names = {
        "Lint with ruff",
        "Type check with mypy",
        "Dead code detection with vulture",
        "Duplicate code detection",
        "Dead feature flag detection",
        "Unused dependency detection with deptry",
    }
    by_name = {step["name"]: step for step in steps if "name" in step}
    assert all(by_name[name]["if"] == "matrix.python == '3.12.11'" for name in names)
    assert "if" not in by_name["Verify Cashew source and SQLite migration capability"]
    assert (
        "if"
        not in by_name["AGENTS.md validation — verify install and test commands work"]
    )
    assert (
        "if"
        not in by_name[
            "Run tests with coverage (capture log for offline-download scan)"
        ]
    )
    assert "if" not in by_name["CI-03 — fail if embedding model was downloaded"]


def test_ci_matrix_coverage_artifacts_have_unique_names() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    upload = next(
        step
        for step in workflow["jobs"]["test-python"]["steps"]
        if step.get("name") == "Upload coverage report"
    )
    assert upload["with"]["name"] == "coverage-py${{ matrix.python }}"


def test_droid_tag_uses_the_minimum_oidc_permission_for_its_pinned_action() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/droid.yml").read_text())
    job = workflow["jobs"]["droid"]
    assert job["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
        "issues": "write",
        "actions": "read",
        "id-token": "write",
    }
    assert "github.actor == github.repository_owner" in job["if"]
