# __init__.py (repo root — dual-context implementation)
#
# FLAT-ENTRY (Hermes): Loaded as _hermes_user_memory.cashew.
# Python sets __package__ = '_hermes_user_memory' (a non-existent parent).
# We detect flat-entry by __spec__.parent not being a real package in sys.modules.
# We provide the full implementation by pre-populating sys.modules and exec'ing the
# nested __init__.py so relative imports resolve.
#
# PIP / TEST: Loaded as plugins.memory.cashew via namespace package resolution.
# Python's normal import mechanism loads nested/__init__.py directly (this root
# __init__.py is not involved since there's no root __init__.py in the namespace).
from __future__ import annotations

import importlib.util
import sys
import os

# Detect flat-entry: __spec__.parent is '_hermes_user_memory' (no real package).
# In pip/test, this root __init__.py is NOT loaded (namespace package skips it).
_spec_parent = getattr(__spec__, "parent", "") or ""
_is_flat = _spec_parent.startswith("_hermes_user_memory") or _spec_parent.startswith("_hermes")

if _is_flat:
    _root = os.path.dirname(os.path.abspath(__file__))
    _nested = os.path.join(_root, "plugins", "memory", "cashew")

    # Pre-populate sys.modules so nested's relative imports resolve.
    _pkg_name = _spec_parent
    if _pkg_name not in sys.modules:
        _pkg_mod = type(sys)(_pkg_name)
        _pkg_mod.__path__ = [_nested]
        sys.modules[_pkg_name] = _pkg_mod

    # Load config/tools/verify into _pkg_name namespace
    for _name in ("config", "tools", "verify"):
        _path = os.path.join(_nested, f"{_name}.py")
        _spec = importlib.util.spec_from_file_location(f"{_pkg_name}.{_name}", _path)
        if _spec and _spec.loader:
            _m = importlib.util.module_from_spec(_spec)
            sys.modules[f"{_pkg_name}.{_name}"] = _m
            _spec.loader.exec_module(_m)

    # Load agent.memory_provider and core.context (may already be in sys.modules)
    try:
        from agent.memory_provider import MemoryProvider  # noqa: F401,F811
    except ImportError:
        pass
    try:
        from core.context import ContextRetriever  # noqa: F401,F811
    except ImportError:
        pass

    # Now exec the nested __init__.py as plugins.memory.cashew
    _spec = importlib.util.spec_from_file_location(
        "plugins.memory.cashew",
        os.path.join(_nested, "__init__.py"),
        submodule_search_locations=[_nested],
    )
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["plugins.memory.cashew"] = _mod
        _spec.loader.exec_module(_mod)
        for _attr in dir(_mod):
            if not _attr.startswith("__"):
                globals()[_attr] = getattr(_mod, _attr)
else:
    # pip / test — namespace package loads nested/__init__.py directly.
    # This root file is not involved in that path.
    from plugins.memory.cashew import CashewMemoryProvider, register  # noqa: F401
    __all__ = ["CashewMemoryProvider", "register"]
