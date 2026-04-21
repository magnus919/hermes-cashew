# plugins/memory/cashew/__init__.py
# Source: Pattern mirrored from plugins/memory/hindsight/__init__.py (NousResearch/hermes-agent@main)
from __future__ import annotations

import logging
import pathlib
import queue
import threading  # Phase 4: non-daemon worker for sync_turn drain
import time       # Phase 4: monotonic-clock polling in on_session_end
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

# Phase 3 Plan 02 — tool-surface helpers live in a dedicated module (PHASE_DESIGN_NOTES
# Decision Point 2). `import json` is redundant with tools.py's own use of json.dumps
# but is kept at module level for forward compatibility with future defensive parsing
# of `args` if Hermes ever passes a JSON string instead of a dict.
import json  # noqa: E402 — intentional: documented Phase 3 forward-compat hook
from plugins.memory.cashew.tools import (  # noqa: E402
    CASHEW_EXTRACT_SCHEMA,
    CASHEW_QUERY_SCHEMA,
    build_error_envelope,
    build_extract_error_envelope,
    build_extract_success_envelope,
    build_success_envelope,
)

logger = logging.getLogger(__name__)


_SHUTDOWN = object()
"""Unique sentinel for graceful sync worker exit.

Compared with `is` (identity), never `==`. Never None — None collides with
legitimate test-code payloads and with Python 3.13's queue.Queue.shutdown()
signal path. See 04-RESEARCH.md §6.4 for rationale.
"""


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
        self._sync_worker: "threading.Thread | None" = None
        # Phase 4: non-daemon worker that drains _sync_queue. Started in
        # initialize() only on happy path (Pitfall 4). Joined in shutdown()
        # with bounded timeout. See 04-RESEARCH.md §§6.5, 6.6.
        self._dropped_turn_count: int = 0
        # Phase 4: monotonic counter of drop-oldest events on the sync queue.
        # Incremented inside sync_turn's overflow branch each time a queued
        # turn is evicted to make room for a new one. Exposed for Plan 04-04's
        # test_burst_default_queue_drops_oldest_cleanly to assert counter
        # matches the WARNING-log count. See B-02 revision in
        # PHASE_DESIGN_NOTES (2026-04-20).

    @property
    def name(self) -> str:
        return "cashew"

    def is_available(self) -> bool:
        """Return True iff a Cashew config file exists under hermes_home.

        Contract (per ROADMAP Phase 2 Success #3 + Phase 1 RESEARCH.md Pitfall 5):
        zero I/O beyond ONE Path.exists() probe. No file content is read here.
        Hermes calls is_available() before initialize using a temporary provider
        instance — we probe the default config location if _hermes_home has not
        been set yet.
        """
        if self._hermes_home is not None:
            return resolve_config_path(self._hermes_home).exists()
        # Fallback for pre-initialize check
        try:
            from hermes_constants import get_hermes_home
            return resolve_config_path(get_hermes_home()).exists()
        except Exception:
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
            # Phase 4: start the sync worker AFTER all worker-read state
            # is populated (db_path, session_id, sync_queue). Pitfall 4.
            self._start_sync_worker()
        except Exception:
            logger.warning(
                "cashew initialize failed at %s; provider will report unavailable until fixed",
                resolve_config_path(self._hermes_home),
                exc_info=True,
            )
            self._config = None
            self._db_path = None
            self._retriever = None
            self._sync_worker = None

    def _start_sync_worker(self) -> None:
        """Launch the non-daemon worker. Called from initialize() only on happy path.

        MUST run AFTER self._db_path / self._session_id / self._sync_queue are set
        (Pitfall 4). daemon=False is load-bearing (PROJECT.md Key Decision).
        """
        self._sync_worker = threading.Thread(
            target=self._worker_loop,
            name=f"cashew-sync-{self._session_id}",
            daemon=False,
        )
        self._sync_worker.start()

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        """Hot-path enqueue of a completed turn (SYNC-01).

        Contract: returns in <10ms. Never raises. If the queue is full, drops the
        OLDEST queued turn, logs a WARNING, and enqueues the new one (drop-oldest
        policy, 04-RESEARCH.md §6.3). If somehow still full after the drop (rare
        worker-draining race), drops the NEW turn with a second WARNING.

        Half-state (_sync_queue is None) is a silent no-op.
        """
        if self._sync_queue is None:
            return  # not initialized or silent-degraded; no worker to feed
        turn = (user_content, assistant_content)
        try:
            self._sync_queue.put_nowait(turn)
        except queue.Full:
            # Drop-oldest (04-RESEARCH.md §6.3 + Pitfall 3).
            try:
                self._sync_queue.get_nowait()
                self._sync_queue.task_done()  # balance the drop (EXACTLY ONCE)
            except queue.Empty:
                pass  # worker drained between Full and get_nowait — rare race; no-op
            self._dropped_turn_count += 1  # per-drop counter — asserted by Plan 04-04 burst test
            logger.warning(
                "cashew sync queue overflow (maxsize=%d); dropped oldest turn",
                self._sync_queue.maxsize,
            )
            try:
                self._sync_queue.put_nowait(turn)
            except queue.Full:
                logger.warning("cashew sync queue still full after drop-oldest; dropping new turn")

    def _worker_loop(self) -> None:
        """Background drain loop. Entry point for self._sync_worker.

        Sentinel check BEFORE try (Pitfall 1 — must not be reachable from the
        exception path). task_done() ALWAYS in finally (Pitfall 2). Per-iteration
        except catches all Cashew failures (SYNC-06) without poisoning the queue.

        Plan 04-04 race fix: binds the queue reference to a local `q` at loop
        entry. If shutdown() times out waiting for this worker, it clears
        `self._sync_queue = None` and abandons the worker. Without this local
        bind, the abandoned worker's `finally: self._sync_queue.task_done()`
        would raise AttributeError on NoneType. Using `q` keeps task_done()
        bound to the queue the worker was actually draining, race-free.
        """
        q = self._sync_queue  # bind once; shutdown may clear self._sync_queue before we exit
        assert q is not None  # invariant: worker only starts when queue exists
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                q.task_done()
                return
            try:
                self._drain_once(item)
            except Exception:
                logger.warning("cashew sync worker: turn failed", exc_info=True)
            finally:
                q.task_done()

    def _drain_once(self, turn: tuple[str, str]) -> None:
        """Persist one turn via Cashew's heuristic extractor.

        Lazy-imports core.session so the plugin module loads even when cashew-brain
        is not installed (matches Phase 1 Pattern 3 for agent.memory_provider).
        model_fn=None uses Cashew's built-in heuristic extractor (04-RESEARCH.md
        §§1, 6.2 — no LLM round-trip).
        """
        from core.session import end_session  # lazy import (see 04-RESEARCH.md §9 + test strategy §11)
        user, assistant = turn
        end_session(
            db_path=str(self._db_path),
            session_id=self._session_id,
            conversation_text=f"User: {user}\nAssistant: {assistant}",
            model_fn=None,
        )

    def on_session_end(self, messages: list) -> None:
        """Best-effort bounded drain; does NOT stop the worker (ABC-06).

        Polls unfinished_tasks with the same sync_queue_timeout that shutdown()
        uses. Worker stays alive for subsequent sessions. No WARNING on timeout —
        this method is advisory; shutdown() is authoritative. See 04-RESEARCH.md
        §6.10 + PHASE_DESIGN_NOTES Decision Point 4.
        """
        if self._sync_queue is None:
            return  # not initialized or silent-degraded
        timeout = self._config.sync_queue_timeout if self._config is not None else 30.0
        deadline = time.monotonic() + timeout
        while self._sync_queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        # Intentional: no WARNING on incomplete drain — shutdown() will log if it also times out.

    def shutdown(self) -> None:
        """Post sentinel, bounded-join worker, clear references (ABC-05 + SYNC-05).

        Order is load-bearing:
          1. If not initialized, return (Phase 2 safe-no-op carryover).
          2. Post _SHUTDOWN sentinel to the queue. put_nowait first; fallback to
             a 1s blocking put if the queue is full (worker is draining fast).
          3. Bounded-join the worker using sync_queue_timeout. WARNING on timeout.
          4. Clear _sync_queue, _sync_worker, _config, _db_path, _retriever.

        _hermes_home is intentionally NOT reset — is_available() must keep
        reflecting on-disk reality (Phase 2 Success #3).
        """
        if self._sync_queue is None:
            return  # safe no-op: initialize() was never called
        timeout = self._config.sync_queue_timeout if self._config is not None else 30.0
        # Post sentinel. put_nowait first; if somehow full, try a brief blocking put.
        try:
            self._sync_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            try:
                self._sync_queue.put(_SHUTDOWN, block=True, timeout=1.0)
            except queue.Full:
                logger.warning("cashew shutdown: could not post sentinel; worker may leak")
        # Bounded join. NEVER raise (silent-degrade Key Decision).
        if self._sync_worker is not None:
            self._sync_worker.join(timeout=timeout)
            if self._sync_worker.is_alive():
                logger.warning(
                    "cashew sync worker did not exit within %ss; abandoning",
                    timeout,
                )
        # Clear state. _hermes_home persists (see Phase 2 Plan 02-02 rationale).
        self._sync_queue = None
        self._sync_worker = None
        self._config = None
        self._db_path = None
        self._retriever = None
        logger.debug("cashew provider shutdown complete (Phase 4 — worker drained)")

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
        """Return the list of LLM tool schemas this provider exposes (RECALL-02 + SYNC-03).

        Phase 3 + Phase 4: two tools — cashew_query (recall) and cashew_extract
        (explicit sync). Schema structure follows Anthropic's input_schema
        convention — see 03-RESEARCH.md §3 / 04-RESEARCH.md §6.9 and
        plugins.memory.cashew.tools.CASHEW_QUERY_SCHEMA /
        plugins.memory.cashew.tools.CASHEW_EXTRACT_SCHEMA.

        The returned list is a fresh list literal each call, but the schema dicts
        themselves are module constants (not copies) — callers must not mutate.
        """
        return [CASHEW_QUERY_SCHEMA, CASHEW_EXTRACT_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """Route an LLM tool call to the Cashew backend.

        Two tools are handled:
          - cashew_query (Phase 3): recall from the thought graph.
          - cashew_extract (Phase 4): explicit, synchronous extraction of
            one turn. Bypasses the sync queue — returns only after Cashew
            completes.

        Silent-degrade paths (per PROJECT.md Key Decision):
          - Unknown tool -> WARNING (no exc_info) + error envelope with
            tool='cashew_query' for historical compatibility.
          - Half-state (initialize never ran or silent-degraded) -> error
            envelope, no log (initialize already warned).
          - Any exception during happy path -> WARNING + exc_info=True +
            error envelope.

        Returns:
            JSON string — NEVER None, NEVER raises into Hermes.
        """
        if name == "cashew_query":
            # -- Phase 3 branch preserved verbatim --
            # Half-state guard (PHASE_DESIGN_NOTES Decision Point 1 + 03-RESEARCH.md §8):
            # match Plan 03-01's prefetch contract. No log here — initialize already
            # emitted the WARNING that set _retriever = None.
            if self._retriever is None or self._config is None:
                return build_error_envelope(
                    query=args.get("query"),
                    error_message="cashew recall failed",
                )
            try:
                query = args["query"]  # KeyError caught below — counts as tool-call failure
                max_nodes = args.get("max_nodes", self._config.recall_k)
                nodes = self._retriever.retrieve(query, max_nodes=max_nodes)
                context = self._retriever.format_context(nodes)
                return build_success_envelope(
                    query=query,
                    context=context,
                    node_count=len(nodes),
                )
            except Exception:
                logger.warning(
                    "cashew tool call %r failed",
                    name,
                    exc_info=True,
                )
                return build_error_envelope(
                    query=args.get("query"),
                    error_message="cashew recall failed",
                )

        elif name == "cashew_extract":
            # Half-state guard (matches Phase 3 cashew_query + 04-RESEARCH.md §6.9).
            # No log — initialize() already warned when it set _db_path / _config to None.
            if self._db_path is None or self._config is None:
                return build_extract_error_envelope()
            try:
                user = args["user_content"]  # KeyError caught below — tool-call failure
                assistant = args["assistant_content"]
                # Lazy import (matches _drain_once in Plan 04-01 — keeps is_available
                # free of core.session side effects).
                from core.session import end_session
                result = end_session(
                    db_path=str(self._db_path),
                    session_id=self._session_id,
                    conversation_text=f"User: {user}\nAssistant: {assistant}",
                    model_fn=None,
                )
                return build_extract_success_envelope(
                    new_nodes=len(result.new_nodes),
                    new_edges=len(result.new_edges),
                )
            except Exception:
                logger.warning(
                    "cashew tool call %r failed",
                    name,
                    exc_info=True,
                )
                return build_extract_error_envelope()

        else:
            # Unknown-tool branch (Phase 3 contract preserved; note: uses the QUERY
            # envelope because historically unknown-tool returned the cashew_query
            # error shape. This preserves backward compatibility with Phase 3's
            # test_handle_tool_call.py::test_unknown_tool_returns_error_envelope_and_logs_once
            # which asserts tool='cashew_query', error='unknown tool', query=None.
            # Rationale: unknown-tool routing predates the existence of multiple
            # tools; the envelope has served as a generic-error shape. Phase 4 does
            # not change this behavior — only ADDS a cashew_extract-specific error
            # envelope for the cashew_extract branch.)
            logger.warning("cashew unknown tool call: %r", name)
            return build_error_envelope(query=None, error_message="unknown tool")

    def system_prompt_block(self) -> str:
        """Return a ~10-line LLM-visible status string for the system prompt (UX-01).

        The returned string is included verbatim in Hermes's assembled system prompt
        so the LLM can reason about what memory is available.

        Format (~10 lines):
            [cashew] memory provider: available
            graph: <N> nodes, <M> edges
            recall depth: <recall_k>

        When unavailable or empty: clearly signals the LLM should not expect recall.
        Never raises. Returns a plain str, not a dict or JSON.
        """
        if self._config is None or self._hermes_home is None:
            return "[cashew] memory provider: not configured\n"

        try:
            recall_k = self._config.recall_k
        except AttributeError:
            recall_k = 5

        if self._db_path is None:
            return (
                f"[cashew] memory provider: available (db not initialized)\n"
                f"graph: uninitialized\n"
                f"recall depth: {recall_k}\n"
            )

        try:
            import sqlite3
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                "SELECT COUNT(*), (SELECT COUNT(*) FROM derivation_edges) FROM thought_nodes"
            )
            row = cursor.fetchone()
            conn.close()
            node_count = row[0] if row else 0
            edge_count = row[1] if row else 0
            if node_count == 0:
                graph_state = "empty"
            else:
                graph_state = f"{node_count} nodes, {edge_count} edges"
        except Exception:
            graph_state = "unknown"

        return (
            f"[cashew] memory provider: available\n"
            f"graph: {graph_state}\n"
            f"recall depth: {recall_k}\n"
        )

    # All other ABC methods are inherited as no-ops from the ABC defaults (when Hermes is present).


def register(ctx) -> None:
    """Hermes discovery entry point — filesystem loader calls this."""
    ctx.register_memory_provider(CashewMemoryProvider())
