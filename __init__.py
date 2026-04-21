# __init__.py (repo root)
# When `hermes plugins install` drops this repo at $HERMES_HOME/plugins/cashew/, the
# Hermes memory loader loads THIS file as `_hermes_user_memory.cashew`. We re-export
# CashewMemoryProvider and register from the nested implementation.
#
# Both loading paths are supported:
#   - Hermes flat-entry loader: relative import from _hermes_user_memory.cashew
#   - Direct / pip-installed: absolute import from plugins.memory.cashew
#
# See .planning/phases/01-scaffolding-discovery-packaging/01-01-SPIKE-REPORT.md (A2 FLAT-REQUIRED)
# for why this dual layout is required.
try:
    # Flat-entry loader path — this file is loaded with a package context.
    from .plugins.memory.cashew import CashewMemoryProvider, register
except ImportError:
    # Standalone / pip-installed path — __init__.py is loaded without a package
    # context, or plugins/memory/cashew/ is accessible via sys.path.
    from plugins.memory.cashew import CashewMemoryProvider, register

__all__ = ["CashewMemoryProvider", "register"]
