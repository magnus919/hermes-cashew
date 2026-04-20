# plugins/memory/cashew/__init__.py
# Source: Pattern mirrored from plugins/memory/hindsight/__init__.py (NousResearch/hermes-agent@main)
from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    MemoryProvider = object  # Hermes not installed — allows module to load for discovery/wheel-smoke; is_available() gates real usage

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

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Phase 3 adds cashew_query, Phase 4 adds cashew_extract
        return []

    # All other ABC methods are inherited as no-ops from the ABC defaults (when Hermes is present).


def register(ctx) -> None:
    """Hermes discovery entry point — filesystem loader calls this."""
    ctx.register_memory_provider(CashewMemoryProvider())
