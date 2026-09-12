"""Contracts for the user-facing installation verifier."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.memory.cashew import verify


def _fixed_temp_dir(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    def make_temp_dir(*, prefix: str) -> str:
        assert prefix == "cashew-verify-"
        path.mkdir()
        return str(path)

    monkeypatch.setattr(verify.tempfile, "mkdtemp", make_temp_dir)


def test_main_exercises_real_provider_without_leaving_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verify_home = tmp_path / "verify-home"
    _fixed_temp_dir(monkeypatch, verify_home)
    assert verify.main() == 0

    captured = capsys.readouterr()
    assert captured.out == "[cashew] verify: all checks passed\n"
    assert "[cashew] ERROR" not in captured.err
    assert not verify_home.exists()
    assert "CASHEW_EMBD_CACHE" not in os.environ


def test_main_reports_initialize_failure_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verify_home = tmp_path / "verify-home"
    _fixed_temp_dir(monkeypatch, verify_home)

    class FailingProvider:
        def initialize(self, session_id: str, **kwargs) -> None:
            raise RuntimeError("broken provider")

    import plugins.memory.cashew as cashew_module

    monkeypatch.setattr(cashew_module, "CashewMemoryProvider", FailingProvider)

    with pytest.raises(SystemExit, match="1"):
        verify.main()

    captured = capsys.readouterr()
    assert (
        captured.err
        == "[cashew] ERROR: initialize raised RuntimeError: broken provider\n"
    )
    assert not verify_home.exists()
    assert "CASHEW_EMBD_CACHE" not in os.environ


def test_module_entrypoint_reports_unavailable_child_without_path_leakage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "plugins.memory.cashew.verify"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        timeout=30,
    )
    # The subprocess environment deliberately has no cached model.  A real
    # verifier must now report that failed child handshake instead of silently
    # falling back to a parent-local embedding service.
    assert result.returncode == 1
    assert (
        "[cashew] ERROR: health_status() did not report an operational state"
        in result.stderr
    )
    assert "health_status" not in result.stdout
