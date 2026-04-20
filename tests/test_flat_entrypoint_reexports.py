# tests/test_flat_entrypoint_reexports.py
# Spike-mandated (A2 FLAT-REQUIRED, see .planning/phases/01-scaffolding-discovery-packaging/01-01-SPIKE-REPORT.md):
# `hermes plugins install` git-clones the repo into $HERMES_HOME/plugins/cashew/.
# The Hermes memory loader scans that directory for __init__.py DIRECTLY — it does
# NOT recurse into the nested plugins/memory/cashew/. Therefore the root __init__.py
# must re-export `register` and `CashewMemoryProvider` from the nested implementation.
import importlib.util
import pathlib


def test_root_init_reexports_register_and_provider():
    """Simulates hermes plugins install dropping the repo at $HERMES_HOME/plugins/cashew/.
    The memory loader reads __init__.py directly; it must re-export register + CashewMemoryProvider."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    init_path = repo_root / "__init__.py"
    assert init_path.exists(), f"root __init__.py missing at {init_path}"

    spec = importlib.util.spec_from_file_location("_flat_cashew", init_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "register"), "root __init__.py must re-export register"
    assert callable(mod.register), f"register must be callable, got {type(mod.register)}"
    assert hasattr(mod, "CashewMemoryProvider"), "root __init__.py must re-export CashewMemoryProvider"
    assert mod.CashewMemoryProvider().name == "cashew"
