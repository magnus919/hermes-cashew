"""Regression tests for the pinned Hermes host harness result guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_HARNESS_PATH = Path(__file__).parents[1] / "scripts" / "run-hermes-cron-host-tests.py"
_HARNESS_SPEC = importlib.util.spec_from_file_location("hermes_cron_host_harness", _HARNESS_PATH)
assert _HARNESS_SPEC and _HARNESS_SPEC.loader
_HARNESS = importlib.util.module_from_spec(_HARNESS_SPEC)
_HARNESS_SPEC.loader.exec_module(_HARNESS)
_validate_host_results = _HARNESS._validate_host_results


@pytest.mark.parametrize(
    "summary",
    [
        "13 passed, 1 failed in 0.1s",
        "13 passed, 1 error in 0.1s",
        "13 passed, 1 skipped in 0.1s",
        "13 passed, 1 xfailed in 0.1s",
        "13 passed, 1 xpassed in 0.1s",
        "13 passed, 1 deselected in 0.1s",
    ],
)
def test_host_harness_rejects_non_pass_outcomes(summary: str) -> None:
    with pytest.raises(ValueError):
        _validate_host_results(14, summary)


def test_host_harness_requires_every_collected_test_to_pass() -> None:
    with pytest.raises(ValueError, match="pass/collection mismatch"):
        _validate_host_results(14, "13 passed in 0.1s")


def test_host_harness_accepts_complete_success() -> None:
    _validate_host_results(14, "14 passed in 0.1s")
