# Root-level conftest.py — tells pytest to ignore the flat-entry loader shim
# (repo-root __init__.py) which uses a relative import that only resolves when
# Hermes loads it as _hermes_user_memory.cashew, not when pytest imports it.
from __future__ import annotations

import pathlib

import pytest

collect_ignore = ["__init__.py"]


def _object_repr_artifacts(repo_root: pathlib.Path) -> set[str]:
    """Return accidental DB paths produced by stringified test doubles."""
    return {
        path.name
        for path in repo_root.iterdir()
        if path.is_file()
        and (
            path.name == "None"
            or path.name.startswith("<MagicMock")
            or path.name.startswith("<sqlite3.Connection object at ")
        )
    }


@pytest.fixture(scope="session", autouse=True)
def reject_new_object_repr_artifacts():
    """Fail the suite if a test writes a database to an object-repr path."""
    repo_root = pathlib.Path(__file__).resolve().parent
    before = _object_repr_artifacts(repo_root)
    yield
    created = sorted(_object_repr_artifacts(repo_root) - before)
    assert not created, f"tests created object-repr artifacts in repo root: {created}"
