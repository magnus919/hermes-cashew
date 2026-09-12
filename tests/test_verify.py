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
    previous_cache = "/existing/embedding-cache.db"
    monkeypatch.setenv("CASHEW_EMBD_CACHE", previous_cache)

    assert verify.main() == 0

    captured = capsys.readouterr()
    assert captured.out == "[cashew] verify: all checks passed\n"
    assert "[cashew] ERROR" not in captured.err
    assert not verify_home.exists()
    assert os.environ["CASHEW_EMBD_CACHE"] == previous_cache


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


def test_module_entrypoint_consumes_health_status_without_path_leakage() -> None:
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
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[cashew] verify: all checks passed"
    assert "health_status" not in result.stdout
    assert "Traceback" not in result.stderr
