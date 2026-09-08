"""Security contracts for secret-bearing GitHub Actions workflows."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github/workflows/opencode.yml"


def _workflow() -> tuple[str, dict[str, Any]]:
    text = WORKFLOW_PATH.read_text()
    return text, yaml.safe_load(text)


def test_opencode_job_requires_repository_owner_and_protected_environment() -> None:
    """Untrusted commenters must not start a secret-bearing runner."""
    _, workflow = _workflow()
    job = workflow["jobs"]["opencode"]

    assert "github.actor == github.repository_owner" in job["if"]
    assert job["environment"] == "opencode"
    assert job["permissions"] == {
        "id-token": "write",
        "contents": "read",
        "pull-requests": "read",
        "issues": "read",
    }


def test_opencode_rejects_fork_pr_before_secret_bearing_step() -> None:
    """Pull-request provenance is checked before checkout or OpenCode runs."""
    text, workflow = _workflow()
    steps = workflow["jobs"]["opencode"]["steps"]
    trust, checkout, opencode = steps

    assert trust["id"] == "trust"
    assert "OPENROUTER_API_KEY" not in str(trust)
    assert "gh api" in trust["run"]
    assert ".head.repo.full_name" in trust["run"]
    assert 'echo "allowed=false"' in trust["run"]
    assert checkout["if"] == "steps.trust.outputs.allowed == 'true'"
    assert opencode["if"] == "steps.trust.outputs.allowed == 'true'"
    assert text.count("${{ secrets.OPENROUTER_API_KEY }}") == 1
    assert opencode["env"]["OPENROUTER_API_KEY"] == "${{ secrets.OPENROUTER_API_KEY }}"


def test_opencode_trust_guard_has_valid_bash_syntax() -> None:
    """The fail-closed guard must be executable, not merely valid YAML."""
    _, workflow = _workflow()
    guard = workflow["jobs"]["opencode"]["steps"][0]["run"]

    result = subprocess.run(
        ["bash", "-n"],
        input=guard,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("pull_request_url", "head_repository", "gh_exit", "expected_exit", "allowed"),
    [
        ("", "unused/fork", 99, 0, "true"),
        ("https://api.github.test/pulls/42", "owner/repo", 0, 0, "true"),
        ("https://api.github.test/pulls/42", "outsider/fork", 0, 0, "false"),
        ("https://api.github.test/pulls/42", "unused/fork", 1, 1, None),
    ],
)
def test_opencode_trust_guard_fails_closed(
    tmp_path: Path,
    pull_request_url: str,
    head_repository: str,
    gh_exit: int,
    expected_exit: int,
    allowed: str | None,
) -> None:
    """The guard allows issues and same-repo PRs, but rejects forks/API errors."""
    _, workflow = _workflow()
    guard = workflow["jobs"]["opencode"]["steps"][0]["run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "${HEAD_REPOSITORY}"\nexit "${GH_EXIT}"\n'
    )
    gh.chmod(0o755)
    github_output = tmp_path / "github-output"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GH_TOKEN": "test-token",
        "REPOSITORY": "owner/repo",
        "PULL_REQUEST_URL": pull_request_url,
        "PULL_REQUEST_NUMBER": "42",
        "GITHUB_OUTPUT": str(github_output),
        "HEAD_REPOSITORY": head_repository,
        "GH_EXIT": str(gh_exit),
    }

    result = subprocess.run(
        ["bash"],
        input=guard,
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == expected_exit, result.stderr
    output = github_output.read_text() if github_output.exists() else ""
    if allowed is None:
        assert "allowed=" not in output
    else:
        assert output == f"allowed={allowed}\n"


def test_opencode_actions_are_pinned_to_full_commit_shas() -> None:
    """Mutable action refs must not enter the secret-bearing workflow."""
    text, _ = _workflow()
    action_refs = re.findall(r"^\s*uses:\s*([^\s#]+)", text, re.MULTILINE)

    assert action_refs
    for action_ref in action_refs:
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action_ref), action_ref

    assert (
        "anomalyco/opencode/github@02a167e048d3bd7299225068d79e4fce5c830d67"
        in action_refs
    )
