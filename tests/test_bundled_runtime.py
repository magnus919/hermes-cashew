"""Installer and private-engine regressions for discussion #261."""

import shutil
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_all_runtime_requirements_survive_hermes_admission():
    root = Path(__file__).resolve().parents[1]
    specs = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    # Exercise the actual immutable implementation from the reporter's host.
    import ast
    import hashlib

    fixture = root / "tests/fixtures/hermes-pm/plugin_declarations.py"
    source = fixture.read_bytes()
    assert (
        hashlib.sha256(source).hexdigest()
        == "fafbb66a9e0ac3f685c7558bc56403f149e6c1f58066dc7f9b5f56df020a016f"
    )
    tree = ast.parse(source)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "applicable_requirements"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(fixture), "exec"),
        namespace,
    )
    accepted = list(namespace["applicable_requirements"](specs))
    assert accepted == specs, "Hermes discards required direct-URL dependencies"


def test_bundled_provenance_and_import_only_relocation():
    from plugins.memory.cashew._vendor.verify import verify

    manifest = verify()
    assert manifest["commit"] == "fcb4919ac37144bfbeb822eaafc668a4bdceb791"
    assert len(manifest["files"]) == 25


@pytest.mark.parametrize(
    "mutation", ["changed", "missing", "extra", "pristine", "license"]
)
def test_provenance_rejects_changed_or_incomplete_vendor(tmp_path, mutation):
    from plugins.memory.cashew._vendor import verify as verifier

    source = Path(verifier.__file__).resolve().parent
    vendor = tmp_path / "vendor"
    shutil.copytree(source, vendor, ignore=shutil.ignore_patterns("__pycache__"))
    if mutation == "changed":
        with (vendor / "core/session.py").open("a") as stream:
            stream.write("\nchanged = True\n")
    elif mutation == "missing":
        (vendor / "core/session.py").unlink()
    elif mutation == "extra":
        (vendor / "core/unreviewed.py").write_text("pass\n")
    elif mutation == "pristine":
        with (vendor / "pristine-core.tar.gz").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "license":
        (vendor / "LICENSE").write_text("changed")
    with pytest.raises(ValueError):
        verifier.verify(vendor)


def test_bundled_engine_ignores_preimported_ambient_core(tmp_path):
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import sys, types
        from pathlib import Path
        ambient = types.ModuleType("core")
        ambient.__path__ = []
        sys.modules["core"] = ambient
        import plugins.memory.cashew as provider
        from plugins.memory.cashew._vendor.core import db, session, sleep, integrity, context
        path = Path(sys.argv[1]) / "brain.db"
        db.ensure_schema(str(path))
        assert db.schema_version() > 0
        assert provider.ContextRetriever is context.ContextRetriever
        assert callable(session.end_session)
        assert callable(integrity.repair_integrity)
        result = sleep.run_sleep_cycle(str(path), orphan_limit=0)
        assert isinstance(result, dict)
        assert sys.modules["core"] is ambient
        for module in (db, session, sleep, integrity, context):
            assert "_vendor/core" in module.__file__
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
