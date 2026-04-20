# __init__.py (repo root)
# When `hermes plugins install` drops this repo at $HERMES_HOME/plugins/cashew/, the
# Hermes memory loader scans THIS file looking for `register` or `MemoryProvider`.
# For pip-installed users, this file is not on the import path — they use
# `from plugins.memory.cashew import CashewMemoryProvider` directly via the wheel.
#
# See .planning/phases/01-scaffolding-discovery-packaging/01-01-SPIKE-REPORT.md (A2 FLAT-REQUIRED)
# for why this dual layout is required.
from plugins.memory.cashew import CashewMemoryProvider, register

__all__ = ["CashewMemoryProvider", "register"]
