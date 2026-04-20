# plugins/memory/cashew/__init__.py
# Source: Pattern mirrored from plugins/memory/hindsight/__init__.py (NousResearch/hermes-agent@main)
from __future__ import annotations

import logging
import pathlib
import queue
from typing import Any, Dict, List

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    MemoryProvider = object  # Hermes not installed — allows module to load for discovery/wheel-smoke; is_available() gates real usage

from plugins.memory.cashew.config import (
    CashewConfig,
    get_config_schema as _config_get_config_schema,
    load_config,
    resolve_config_path,
    resolve_db_path,
    save_config as _config_save_config,
)

logger = logging.getLogger(__name__)


class CashewMemoryProvider(MemoryProvider):
    """Cashew thought-graph memory provider for Hermes Agent."""

    def __init__(self) -> None:
        self._retriever = None          # Phase 3
        self._config: dict = {}         # Phase 2
        self._session_id: str = ""      # Phase 2
        # threading state lives in Phase 4

    @property
    def name(self) -> str:
        return "cashew"

    def is_available(self) -> bool:
        """Phase 1: always False (not yet configured). Phase 2 will check for $HERMES_HOME/cashew.json."""
        return False

    def get_config_schema(self) -> Dict[str, Any]:
        """Return the JSON-Schema-shaped dict Hermes uses to drive `hermes memory setup` (CONF-01)."""
        return _config_get_config_schema()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist the four-key Cashew config to $HERMES_HOME/cashew.json (CONF-02).

        Delegates to plugins.memory.cashew.config.save_config which merges over
        DEFAULTS, drops unknown keys, and writes UTF-8 / 2-space-indent / sorted-keys
        JSON. The ABC method returns None; we discard the helper's return path.
        """
        _config_save_config(values, hermes_home)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Phase 3 adds cashew_query, Phase 4 adds cashew_extract
        return []

    # All other ABC methods are inherited as no-ops from the ABC defaults (when Hermes is present).


def register(ctx) -> None:
    """Hermes discovery entry point — filesystem loader calls this."""
    ctx.register_memory_provider(CashewMemoryProvider())
