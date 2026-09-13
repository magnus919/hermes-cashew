"""Installation surfaces must resolve the same runtime dependency contracts."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 CI floor
    import tomli as tomllib
import yaml

CASHEW_ARCHIVE_URL = (
    "https://github.com/magnus919/true/archive/"
    "fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz"
)
CASHEW_ARCHIVE_SHA256 = (
    "23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3"
)
CASHEW_REQUIREMENT = (
    f"cashew-brain @ {CASHEW_ARCHIVE_URL}#sha256={CASHEW_ARCHIVE_SHA256}"
)


def test_plugin_manifests_match_project_runtime_dependencies() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    expected = set(project["project"]["dependencies"])

    for relative in ("plugin.yaml", "plugins/memory/cashew/plugin.yaml"):
        manifest = yaml.safe_load((root / relative).read_text())
        assert set(manifest["pip_dependencies"]) == expected

    assert CASHEW_REQUIREMENT in expected


def test_lock_records_selected_cashew_source_and_archive_hash() -> None:
    root = Path(__file__).parents[1]
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = lock["package"]
    cashew = next(package for package in packages if package["name"] == "cashew-brain")
    wrapper = next(
        package for package in packages if package["name"] == "hermes-cashew"
    )

    assert cashew["source"] == {"url": CASHEW_ARCHIVE_URL}
    assert cashew["sdist"] == {"hash": f"sha256:{CASHEW_ARCHIVE_SHA256}"}
    wrapper_requirement = next(
        requirement
        for requirement in wrapper["metadata"]["requires-dist"]
        if requirement["name"] == "cashew-brain"
    )
    assert wrapper_requirement == {"name": "cashew-brain", "url": CASHEW_ARCHIVE_URL}


def test_source_install_docs_require_reinstall_and_provenance_check() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text()
    integration_readme = (root / "integration/README.md").read_text()

    assert CASHEW_REQUIREMENT in readme
    assert '--reinstall "$CASHEW_PIN" sqlite-vec' in readme
    assert "scripts/verify-cashew-baseline.py" in readme
    assert "hermes plugins update cashew" in readme
    assert "hermes update" in readme
    assert "installed automatically by `hermes plugins install`" not in readme
    assert CASHEW_REQUIREMENT in integration_readme
    assert '--reinstall "$CASHEW_PIN"' in integration_readme
    assert "scripts/verify-cashew-baseline.py" in integration_readme


def test_embedding_runtime_dependencies_are_direct_project_requirements() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependency_names = {
        re.match(r"[A-Za-z0-9_.-]+", dependency).group()
        for dependency in project["project"]["dependencies"]
    }

    assert {"numpy", "sentence-transformers"} <= dependency_names
    assert "scikit-learn" not in dependency_names
