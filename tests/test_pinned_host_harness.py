"""Focused guards for the real-Hermes release-candidate harness."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "pinned_host_harness", Path(__file__).parents[1] / "integration" / "run_pinned_host.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_parse_cron_result = _MODULE._parse_cron_result
_verify_wheel_package = _MODULE._verify_wheel_package


@pytest.mark.parametrize(
    "stdout",
    ["{}", "[]", '{"status": "completed", "nodes_selected": 0}', "not json",
     '{"status": "unavailable", "nodes_selected": 1}'],
)
def test_cron_result_rejects_empty_invalid_or_non_effective_output(stdout: str) -> None:
    with pytest.raises(AssertionError):
        _parse_cron_result(stdout)


def test_cron_result_requires_documented_status_and_selected_nodes() -> None:
    assert _parse_cron_result(
        '{"status": "completed", "nodes_selected": 2}'
    ) == {"status": "completed", "nodes_selected": 2}


def test_wheel_package_rejects_stale_installed_bytes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    installed = tmp_path / "installed"
    (candidate / "nested").mkdir(parents=True)
    (installed / "nested").mkdir(parents=True)
    (candidate / "nested" / "provider.py").write_text("candidate", encoding="utf-8")
    (installed / "nested" / "provider.py").write_text("stale", encoding="utf-8")
    digest = base64.urlsafe_b64encode(
        hashlib.sha256((candidate / "nested" / "provider.py").read_bytes()).digest()
    ).rstrip(b"=").decode("ascii")
    record = f"plugins/memory/cashew/nested/provider.py,sha256={digest},{len('candidate')}\n"
    with pytest.raises(SystemExit, match="candidate source"):
        _verify_wheel_package(installed, candidate, record)


@pytest.mark.parametrize("hash_field", [None, "", "sha256="])
def test_wheel_package_requires_record_hash_for_every_file(
    tmp_path: Path, hash_field: str | None,
) -> None:
    candidate = tmp_path / "candidate"
    installed = tmp_path / "installed"
    for directory in (candidate, installed):
        directory.mkdir()
        (directory / "provider.py").write_text("candidate", encoding="utf-8")
    record = (
        "" if hash_field is None
        else f"plugins/memory/cashew/provider.py,{hash_field},9\n"
    )
    with pytest.raises(SystemExit, match="wheel RECORD hash"):
        _verify_wheel_package(installed, candidate, record)
