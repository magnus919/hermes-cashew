"""Regression coverage for clean-source and clean-artifact CI guards."""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sdist(path: Path, names: list[str], *, symlink: bool = False) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            info = tarfile.TarInfo(f"hermes_cashew-0.10.2/{name}")
            info.size = 0
            archive.addfile(info, io.BytesIO())
        if symlink:
            info = tarfile.TarInfo("hermes_cashew-0.10.2/plugins/memory/cashew/loop")
            info.type = tarfile.SYMTYPE
            info.linkname = "plugins/memory/cashew"
            archive.addfile(info)


def test_recursive_symlink_checker_is_read_only_and_explains_remediation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plugins" / "memory" / "cashew"
    source.mkdir(parents=True)
    loop = source / "cashew"
    loop.symlink_to(source, target_is_directory=True)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check-recursive-symlinks.py"),
            str(source),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert result.returncode == 1
    assert "Recursive symlink contamination detected" in result.stdout
    assert str(loop) in result.stdout
    assert "never removes files" in result.stdout
    assert loop.is_symlink()


def test_recursive_symlink_checker_allows_external_development_link(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plugins" / "memory" / "cashew"
    source.mkdir(parents=True)
    external = tmp_path / "checkout"
    external.mkdir()
    (source / "development").symlink_to(external, target_is_directory=True)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check-recursive-symlinks.py"),
            str(source),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stdout


def test_distribution_guard_accepts_clean_wheel_and_sdist(tmp_path: Path) -> None:
    verifier = _load_script("verify-distributions.py")
    wheel = tmp_path / "hermes_cashew-0.10.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("plugins/memory/cashew/__init__.py", "")
    sdist = tmp_path / "hermes_cashew-0.10.2.tar.gz"
    _write_sdist(sdist, ["plugins/memory/cashew/__init__.py"])

    verifier.verify_distribution(wheel)
    verifier.verify_distribution(sdist)


@pytest.mark.parametrize(
    ("name", "writer", "expected"),
    [
        ("nested.db", "wheel", "forbidden paths"),
        ("graphify-out/report.md", "wheel", "forbidden paths"),
        ("tests/fixtures/cashew-pr137/core/integrity.py", "wheel", "forbidden paths"),
        ("plugins/memory/cashew/loop", "sdist-link", "symlink entries"),
    ],
)
def test_distribution_guard_rejects_runtime_and_link_contamination(
    tmp_path: Path, name: str, writer: str, expected: str
) -> None:
    verifier = _load_script("verify-distributions.py")
    if writer == "wheel":
        artifact = tmp_path / "hermes_cashew-0.10.2-py3-none-any.whl"
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr("plugins/memory/cashew/__init__.py", "")
            archive.writestr(name, "")
    else:
        artifact = tmp_path / "hermes_cashew-0.10.2.tar.gz"
        _write_sdist(artifact, ["plugins/memory/cashew/__init__.py"], symlink=True)

    with pytest.raises(ValueError, match=expected):
        verifier.verify_distribution(artifact)


def test_clean_flat_smoke_executes_generated_cron_runtime(tmp_path: Path) -> None:
    """The smoke runs the generated subprocess, not merely its registration."""
    flat_root = tmp_path / "profile" / "plugins" / "cashew"
    shutil.copytree(
        ROOT,
        flat_root,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", "dist", "build", "graphify-out"
        ),
    )
    hermes_home = tmp_path / "profile"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/smoke-flat-install.py"),
            str(flat_root),
            str(hermes_home),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "flat loader and cron runtime verified" in result.stdout
