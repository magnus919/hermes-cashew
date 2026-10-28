"""Installer and private-engine regressions for discussion #261."""

import json
import shutil
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_all_runtime_requirements_survive_hermes_admission() -> None:
    root = Path(__file__).resolve().parents[1]
    specs = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    # Exercise the actual immutable implementation from the reporter's host.
    import ast
    import hashlib

    fixture = root / "tests/fixtures/hermes-pm/plugin_declarations.py"
    source = fixture.read_bytes()
    provenance = json.loads((fixture.parent / "PROVENANCE.json").read_text())
    assert provenance["commit"] == "5b32cfc230078417fc523e12b9aabbc9b59f684b"
    assert provenance["file"] == "pm/plugin_declarations.py"
    assert hashlib.sha256(source).hexdigest() == provenance["sha256"]
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


def test_bundled_provenance_and_import_only_relocation() -> None:
    from plugins.memory.cashew._vendor.verify import verify

    manifest = verify()
    assert manifest["commit"] == "fcb4919ac37144bfbeb822eaafc668a4bdceb791"
    assert len(manifest["files"]) == 25


@pytest.mark.parametrize(
    "mutation", ["changed", "missing", "extra", "pristine", "license"]
)
def test_provenance_rejects_changed_or_incomplete_vendor(
    tmp_path: Path, mutation: str
) -> None:
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


def test_bundled_engine_ignores_preimported_ambient_core(tmp_path: Path) -> None:
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


def test_flat_source_audit_needs_no_installed_package_metadata() -> None:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/verify-cashew-baseline.py"
    result = subprocess.run(
        [sys.executable, "-S", str(script)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bundled Cashew source verified" in result.stdout


def test_relocation_changes_only_allowlisted_import_plumbing() -> None:
    from plugins.memory.cashew._vendor.verify import relocate

    original = b"from core.db import ensure_schema\n    from core.session import end_session\nvalue = 'from core.db import fake'\nfrom .config import config\n"
    assert (
        relocate("core/example.py", original)
        == b"from .db import ensure_schema\n    from .session import end_session\nvalue = 'from core.db import fake'\nfrom .config import config\n"
    )
    bootstrap = b"ROOT = Path(__file__).resolve().parent.parent\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n"
    script = bootstrap + b"from core.embeddings import embed_nodes\nresult = 42\n"
    assert (
        relocate("scripts/migrate_embeddings.py", script)
        == b"from ..core.embeddings import embed_nodes\nresult = 42\n"
    )
    assert relocate("core/example.py", bootstrap) == bootstrap
    assert relocate("core/example.txt", original) == original
