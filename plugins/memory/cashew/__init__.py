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

try:
    from core.context import ContextRetriever
except ImportError:
    # Cashew not installed — plugin still loads (for wheel-smoke + discovery paths).
    # initialize() will silent-degrade if ContextRetriever is unavailable at runtime.
    ContextRetriever = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class CashewMemoryProvider(MemoryProvider):
    """Cashew thought-graph memory provider for Hermes Agent."""

    def __init__(self) -> None:
        # Phase 3 — eager lifecycle: constructed in initialize() once _db_path is
        # known, cleared in shutdown(). None during Phase 2 half-state (corrupt
        # config) so prefetch() + handle_tool_call() can short-circuit uniformly.
        self._retriever: "ContextRetriever | None" = None
        # Phase 2 lifecycle state (None until initialize() runs)
        self._hermes_home: pathlib.Path | None = None
        self._config: CashewConfig | None = None
        self._db_path: pathlib.Path | None = None
        self._sync_queue: queue.Queue | None = None
        self._session_id: str = ""
        # threading state lives in Phase 4 — the worker that drains _sync_queue lives there

    @property
    def name(self) -> str:
        return "cashew"

    def is_available(self) -> bool:
        """Return True iff a Cashew config file exists under the initialized hermes_home.

        Contract (per ROADMAP Phase 2 Success #3 + Phase 1 RESEARCH.md Pitfall 5):
        zero I/O beyond ONE Path.exists() probe. No file content is read here.
        Returns False if initialize() has not yet been called (no _hermes_home set) —
        Hermes calls is_available() before initialize() to decide whether to even
        bother initializing.
        """
        if self._hermes_home is None:
            return False
        return resolve_config_path(self._hermes_home).exists()

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

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Wire the provider to a hermes_home (ABC-04).

        Reads kwargs["hermes_home"] (KeyError if absent — surfaces actionable
        message because Hermes always passes it; the only way it's missing is a
        test / programmer error). Loads config, resolves DB path, creates the
        bounded sync queue. Does NOT start a worker thread — Phase 4 owns that.
        Does NOT touch the Cashew runtime — Phase 3 owns the ContextRetriever.

        Cashew/load failures degrade silently per PROJECT.md Key Decisions:
        WARNING is logged with exc_info=True and self._config stays None;
        is_available() will still answer correctly because it probes the file
        directly, not self._config.
        """
        if "hermes_home" not in kwargs:
            raise KeyError(
                "CashewMemoryProvider.initialize requires hermes_home in kwargs; "
                "Hermes Agent passes it as a keyword. Got: " + repr(sorted(kwargs.keys()))
            )
        self._session_id = session_id
        self._hermes_home = pathlib.Path(kwargs["hermes_home"])
        # Queue is created here so config-driven sizing/timeout values are wired in.
        # Phase 4 adds the non-daemon worker thread that drains it (see CLAUDE.md ### Threading Rule).
        self._sync_queue = queue.Queue(maxsize=16)
        try:
            self._config = load_config(self._hermes_home)
            self._db_path = resolve_db_path(self._hermes_home, self._config.cashew_db_path)
            # Phase 3 eager construction (PHASE_DESIGN_NOTES Decision Point 1).
            # ContextRetriever.__init__ is lazy — no SQLite open, no embedding load yet.
            # Guard against the defensive-import fallback (ContextRetriever = None).
            if ContextRetriever is None:
                raise RuntimeError(
                    "core.context.ContextRetriever unavailable at import time; "
                    "cashew-brain dependency missing"
                )
            self._retriever = ContextRetriever(db_path=str(self._db_path))
        except Exception:
            logger.warning(
                "cashew initialize failed at %s; provider will report unavailable until fixed",
                resolve_config_path(self._hermes_home),
                exc_info=True,
            )
            self._config = None
            self._db_path = None
            self._retriever = None

    def shutdown(self) -> None:
        """Tear down the provider (ABC contract; safe no-op pre-initialize).

        Phase 2: no worker exists yet, so this just clears references. Phase 4
        will (a) signal the worker via a queue sentinel and (b) bounded-join the
        worker using self._config.sync_queue_timeout. Both additions slot in
        without restructuring this method.

        _hermes_home is intentionally NOT reset — is_available() should continue
        to reflect on-disk reality (the cashew.json file persists across init/shutdown
        cycles in the same process; only the in-memory queue/config state is torn down).
        """
        if self._sync_queue is None:
            # initialize() was never called; nothing to tear down.
            return
        # Phase 4 will: self._sync_queue.put(_SHUTDOWN_SENTINEL); self._worker.join(timeout=...)
        self._sync_queue = None
        self._config = None
        self._db_path = None
        self._retriever = None
        logger.debug("cashew provider shutdown complete (Phase 3 — no worker yet, retriever cleared)")

    def prefetch(self, query: str) -> str:
        """Return recalled-context string from Cashew, respecting `recall_k` (RECALL-01).

        Contract (PHASE_DESIGN_NOTES Decision Point 1 + 03-RESEARCH.md §§1, 2, 8):
        - Half-state guard: if `_retriever is None` (initialize was never called OR
          corrupt config / resolve failure left the provider in the silent-degrade
          state), return `""` without logging. `initialize()` already emitted the
          WARNING when it set `_retriever = None`; we do not double-log.
        - Happy path: `retrieve(query, max_nodes=recall_k)` + `format_context(nodes)`.
          Both methods are synchronous per 03-RESEARCH.md §1. Empty result is valid
          (returns `""` without logging — empty graph is not a failure).
        - Failure path: `except Exception` (no Cashew exception taxonomy per
          03-RESEARCH.md §2). Log exactly ONE `WARNING` with `exc_info=True` and
          return `""`. Never raise into Hermes (silent-degrade is a PROJECT.md Key
          Decision).

        Args:
            query: The search query string. Passed through to Cashew's retrieve
                call as a typed parameter — NO string concatenation into SQL (threat
                T-03-01-01 mitigated by Cashew's parameterized queries).

        Returns:
            Formatted context string on success (may be the empty string if Cashew
            has nothing relevant or errored). Never None, never raises.
        """
        if self._retriever is None or self._config is None:
            # Half-state: initialize() never ran OR ran and silent-degraded.
            # The relevant WARNING was already emitted by initialize(); stay quiet.
            return ""
        try:
            nodes = self._retriever.retrieve(query, max_nodes=self._config.recall_k)
            return self._retriever.format_context(nodes)
        except Exception:
            logger.warning(
                "cashew recall failed for query=%r",
                query,
                exc_info=True,
            )
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        # Phase 3 adds cashew_query, Phase 4 adds cashew_extract
        return []

    # All other ABC methods are inherited as no-ops from the ABC defaults (when Hermes is present).


def register(ctx) -> None:
    """Hermes discovery entry point — filesystem loader calls this."""
    ctx.register_memory_provider(CashewMemoryProvider())
