# Re-export shim for hermes-cashew.
#
# The actual implementation lives under plugins/memory/cashew/ so both
# install paths work:
#   1. `hermes plugins install magnus919/hermes-cashew`  → root __init__.py loads
#   2. Symlink into hermes-agent/plugins/memory/cashew/   → nested __init__.py loads
#
# This shim uses importlib so relative imports in the nested module resolve
# regardless of how the root package was named by the plugin discovery
# system (_hermes_user_memory.cashew, hermes_plugins.cashew, etc.).

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

# Detect flat-entry: __spec__.parent is a Hermes synthetic namespace
# (hermes_plugins or _hermes_user_memory). In pip/test, this root
# __init__.py is NOT loaded (namespace package skips it).
_spec_parent = getattr(__spec__, "parent", "") or ""
_is_flat = (
    _spec_parent.startswith("_hermes_user_memory")
    or _spec_parent.startswith("_hermes")
    or _spec_parent == "hermes_plugins"
    or _spec_parent.startswith("hermes_plugins.")
)

_NESTED = Path(__file__).resolve().parent / "plugins" / "memory" / "cashew"
_IMPL_MODULE = "_hermes_cashew_impl"

if _IMPL_MODULE not in sys.modules:
    _init = _NESTED / "__init__.py"
    if _init.exists():
        _spec = importlib.util.spec_from_file_location(
            _IMPL_MODULE,
            str(_init),
            submodule_search_locations=[str(_NESTED)],
        )
        if _spec and _spec.loader:
            # Pre-register submodules so relative imports inside the
            # nested __init__.py (from .config, from .tools) work.
            for _sf in sorted(_NESTED.glob("*.py")):
                if _sf.name == "__init__.py":
                    continue
                _sn = _sf.stem
                _full = f"{_IMPL_MODULE}.{_sn}"
                if _full not in sys.modules:
                    _ss = importlib.util.spec_from_file_location(_full, str(_sf))
                    if _ss and _ss.loader:
                        _sm = importlib.util.module_from_spec(_ss)
                        sys.modules[_full] = _sm
                        try:
                            _ss.loader.exec_module(_sm)
                        except Exception:
                            pass

            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_IMPL_MODULE] = _mod
            _spec.loader.exec_module(_mod)

            # Re-export everything the discovery system looks for
            CashewMemoryProvider = _mod.CashewMemoryProvider

            def register(ctx) -> None:
                """Hermes discovery entry point."""
                register_fn = getattr(ctx, "register_memory_provider", None)
                if register_fn is not None:
                    register_fn(CashewMemoryProvider())
        else:
            msg = f"hermes-cashew: nested __init__.py found at {_init} but spec_from_file_location returned {_spec}"
            raise RuntimeError(msg)
    else:
        msg = f"hermes-cashew: nested implementation not found at {_NESTED}"
        raise RuntimeError(msg)
else:
    # Already loaded — re-export from cache (shouldn't normally happen)
    from _hermes_cashew_impl import CashewMemoryProvider  # noqa: F811

    def register(ctx) -> None:
        register_fn = getattr(ctx, "register_memory_provider", None)
        if register_fn is not None:
            register_fn(CashewMemoryProvider())
