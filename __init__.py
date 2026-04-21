# __init__.py (repo root)
# When `hermes plugins install` drops this repo at $HERMES_HOME/plugins/cashew/, the
# Hermes memory loader loads THIS file as `_hermes_user_memory.cashew`. We use a
# relative import to reach the nested implementation at plugins/memory/cashew/.
# The nested __init__.py uses relative imports internally (.config, .tools) so it
# can load from either plugins.memory.cashew (bundled-loader path) or
# _hermes_user_memory.cashew.plugins.memory.cashew (flat-entry loader path).
#
# See .planning/phases/01-scaffolding-discovery-packaging/01-01-SPIKE-REPORT.md (A2 FLAT-REQUIRED)
# for why this dual layout is required.
from .plugins.memory.cashew import CashewMemoryProvider, register

__all__ = ["CashewMemoryProvider", "register"]
