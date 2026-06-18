# plugins/memory/cashew/__init__.py
# Source: Pattern mirrored from plugins/memory/hindsight/__init__.py (NousResearch/hermes-agent@main)
from __future__ import annotations

import logging
import os
import pathlib
import queue
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List

from .log_filter import add_scrub_filter
from .metrics import _METRICS
from .tracing import trace_operation

try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    MemoryProvider = object  # Hermes not installed — allows module to load for discovery/wheel-smoke; is_available() gates real usage

from .config import (
    CashewConfig,
    get_ai_domain,
    get_user_domain,
    is_feature_enabled,
    load_config,
    resolve_config_path,
    resolve_db_path,
    resolve_model_fn,
)
from .config import (
    get_config_schema as _config_get_config_schema,
)
from .config import (
    save_config as _config_save_config,
)

try:
    from core.context import ContextRetriever
except ImportError:
    # Cashew not installed — plugin still loads (for wheel-smoke + discovery paths).
    # initialize() will silent-degrade if ContextRetriever is unavailable at runtime.
    ContextRetriever = None

from .tools import (
    CASHEW_EXTRACT_SCHEMA,
    CASHEW_QUERY_SCHEMA,
    build_error_envelope,
    build_extract_error_envelope,
    build_extract_success_envelope,
    build_success_envelope,
)

logger = logging.getLogger(__name__)

# Install scrub filter on the cashew logger so all log output is sanitized.
add_scrub_filter(logger)

# Issue #18: sentence-transformers emits INFO-level progress bars and BertModel
# load reports directly to the terminal during embedding. That noise leaks into
# the Hermes UI and looks like broken output. Raise its logger to WARNING once
# at module load so every embed_text / end_session call downstream stays quiet.
# Leaves WARNING/ERROR from sentence_transformers untouched so real failures
# (e.g. model file missing) still surface.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


_SHUTDOWN = object()
"""Unique sentinel for graceful sync worker exit.

Compared with `is` (identity), never `==`. Never None — None collides with
legitimate test-code payloads and with Python 3.13's queue.Queue.shutdown()
signal path.
"""

# Probe for the Hermes cron module. In a full Hermes Agent environment the
# cron.jobs package is importable (the agent root is on sys.path). In CI and
# standalone test environments it is not — the sleep cycle cron job cannot be
# registered. This constant is checked at cron-registration time so the
# provider silently skips cron setup rather than logging WARNINGs.
_HAS_HERMES_CRON: bool = False
try:
    from cron.jobs import create_job, list_jobs, remove_job  # noqa: F401

    _HAS_HERMES_CRON = True
except ImportError:
    pass

# ── on_pre_compress prompt template ──────────────────────────────────────────
# Dedicated prompt for forest-level conversation-arc extraction.
# Not a reuse of end_session's prompt — asks about meta-patterns, not content.
PRE_COMPRESS_PROMPT_TEMPLATE = """You are analyzing a conversation about to be compressed.
Your job is to identify the forest-level patterns — signals visible across
multiple turns that no single turn would reveal.

For each signal, produce a JSON object with:
- "type": one of "insight" or "observation" (use insight for non-obvious connections)
- "domain": "{user_domain}" if about the human, "{ai_domain}" if about the system
- "content": standalone statement capturing the cross-turn pattern
- "tags": short descriptive labels (e.g. "communication_style", "topic_shift", "decision")
- "keep": true/false

Look for:
- **Topic arc**: what was this conversation *really* about? Topic shifts vs. deep dives.
- **Framing shifts**: did someone change their stance or framing as the conversation progressed?
- **Implicit decisions**: choices made without explicit deliberation
- **Unstated subjects**: topics conspicuously present in subtext but never named
- **Structural gaps**: what wasn't asked / what was assumed
- **Recurring patterns**: interaction patterns that recurred across multiple exchanges

BAD: "The user asked about X and the assistant explained Y" (turn-level summary)
BAD: "They discussed embeddings" (too vague — what *about* embeddings?)
OK: "Discussion shifted from architecture to cost 3 times — cost is the binding constraint"
GOOD: "User challenges assumptions by asking 'why' before accepting any solution — recurring pattern across topics"

Respond with ONLY a JSON array. No markdown, no explanation, no code fences.

Conversation about to be compressed:
{messages_text}
"""


# ── First-load bootstrap helpers ──────────────────────────────────────


def _ensure_config_file(hermes_home: pathlib.Path) -> None:
    """Write default cashew.json if none exists (first-load bootstrap).

    Uses generate_default_config from config.py which never overwrites
    an existing file. Safe to call on every initialize() — no-op after
    the first run.
    """
    from .config import generate_default_config

    generate_default_config(hermes_home)


def _ensure_auxiliary_memory(hermes_home: pathlib.Path) -> None:
    """Auto-populate auxiliary.memory in Hermes config.yaml if absent.

    When llm_aux_role="memory" is set (the new default) but the user has
    no auxiliary.memory section in their Hermes config.yaml, the plugin
    reads the main model config (provider, model, base_url) and creates an
    auxiliary.memory section with matching settings. This makes LLM-powered
    extraction work out of the box without manual config editing.

    Safe to call repeatedly — only writes when auxiliary.memory is absent
    and the main model config provides usable values. Never overwrites an
    existing auxiliary.memory section.
    """
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        return

    try:
        import yaml  # type: ignore[import-untyped]

        raw = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except Exception:
        logger.warning(
            "failed to read %s for auxiliary.memory auto-population",
            config_path,
            exc_info=True,
        )
        return

    # Don't touch existing auxiliary.memory config
    aux = data.get("auxiliary", {})
    if "memory" in aux:
        return

    model_section = data.get("model", {})
    provider = model_section.get("provider")
    default_model = model_section.get("default")
    base_url = model_section.get("base_url")

    if not provider or not default_model:
        logger.info(
            "cannot auto-populate auxiliary.memory: model.provider=%r model.default=%r",
            provider,
            default_model,
        )
        return

    # Build the auxiliary.memory section
    memory_config: dict[str, str | int] = {
        "provider": provider,
        "model": default_model,
    }
    if base_url:
        memory_config["base_url"] = base_url

    if "auxiliary" not in data:
        data["auxiliary"] = {}
    data["auxiliary"]["memory"] = memory_config

    try:
        config_path.write_text(
            yaml.safe_dump(
                data, default_flow_style=False, sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        logger.info(
            "auto-populated auxiliary.memory from main model config: "
            "provider=%s model=%s",
            provider,
            default_model,
        )
    except Exception:
        logger.warning(
            "failed to write auxiliary.memory to %s",
            config_path,
            exc_info=True,
        )


# ── Upstream embedding model patching ──────────────────────────────────

_UPSTREAM_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-large": 1024,
    "thenlper/gte-base": 768,
    "thenlper/gte-small": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}

# Try to patch upstream embedding model at import time, before any session
# initializes. The per-session call in initialize() is the primary path;
# this import-time attempt covers gateway sessions where the provider may
# already be initialized by the time a new session starts.
_IMPORT_TIME_EMBEDDING_MODEL = os.environ.get(
    "CASHEW_EMBEDDING_MODEL", "thenlper/gte-large"
)


def _patch_upstream_embedding(model_name: str) -> None:
    """Patch cashew-brain's module-level constants so the right model is used.

    PyPI cashew-brain v1.1.0 hardcodes ``DEFAULT_MODEL = "all-MiniLM-L6-v2"``
    and ``EMBEDDING_DIM = 384`` in ``core.embedding_service``. The upstream
    ``get_default_service()`` creates a **module-level singleton** with those
    values, so setting env vars or calling with different arguments has no
    effect — the singleton is already baked.

    This function patches the constants **and** the ``__defaults__`` tuple of
    ``EmbeddingService.__init__`` (Python evaluates default arguments at
    function definition time, so changing the module constant alone doesn't
    work) before the singleton is created, so all subsequent ``embed_nodes()``
    / ``end_session()`` calls produce embeddings at the correct dimension.

    Safe to call multiple times — resets the singleton on each call so a
    config change mid-lifecycle takes effect.
    """
    try:
        import core.embedding_service
    except ImportError:
        logger.warning("cashew-brain not installed; cannot patch embedding model")
        return

    dim = _UPSTREAM_KNOWN_DIMS.get(model_name, 1024)

    # Patch module-level constants
    core.embedding_service.DEFAULT_MODEL = model_name
    core.embedding_service.EMBEDDING_DIM = dim

    # Patch __defaults__ — Python evalutes default arguments at function
    # definition time, so changing DEFAULT_MODEL alone doesn't affect
    # EmbeddingService() calls that omit the model parameter.
    func = core.embedding_service.EmbeddingService.__init__
    defaults = list(func.__defaults__)
    defaults[0] = model_name
    func.__defaults__ = tuple(defaults)

    # Reset the singleton so next get_default_service() call creates one
    # with our patched constants and defaults.
    core.embedding_service.reset_default_service()

    # Patch hardcoded model label in core.embeddings.embed_nodes SQL INSERT.
    # PyPI v1.1.0 writes "all-MiniLM-L6-v2" as the model tag regardless of
    # which model was actually used. We replace the function with a wrapper
    # that swaps the label — more robust than patching co_consts.
    try:
        import core.embeddings

        _orig_embed_nodes = core.embeddings.embed_nodes

        def _patched_embed_nodes(db_path: str, batch_size: int = 100) -> dict:
            """Wrap embed_nodes, then fix model labels in the DB."""
            result = _orig_embed_nodes(db_path, batch_size)
            embedded = result.get("embedded", 0)
            if embedded > 0:
                try:
                    import sqlite3

                    conn = sqlite3.connect(db_path)
                    conn.execute("PRAGMA busy_timeout=5000")
                    conn.execute(
                        "UPDATE embeddings SET model=? WHERE model='all-MiniLM-L6-v2' AND updated_at>=datetime('now', '-1 minute')",
                        (model_name,),
                    )
                    conn.commit()
                    conn.close()
                except Exception:
                    logger.warning(
                        "could not fix model labels after embed", exc_info=True
                    )
            return result  # type: ignore[no-any-return]

        core.embeddings.embed_nodes = _patched_embed_nodes
        logger.info("patched embed_nodes model label: %s", model_name)
    except (ImportError, AttributeError, ValueError):
        logger.warning("could not patch core.embeddings model label", exc_info=True)

    logger.info(
        "patched upstream embedding: model=%s dim=%d",
        model_name,
        dim,
    )


# Apply the patch at import time — before any session initializes.
# This covers gateway scenarios where the provider may already be
# constructed by the time initialize() is called.
_patch_upstream_embedding(_IMPORT_TIME_EMBEDDING_MODEL)


def _remove_existing_sleep_job(hermes_home: pathlib.Path | None) -> None:
    """Remove any existing 'cashew-sleep-cycle' cron job to prevent duplicates.

    Hermes cron jobs persist across restarts in ``$HERMES_HOME/cron/jobs.json``.
    Without dedup, each provider initialize() would add another job, causing
    N sleep cycles per tick after N restarts.  This helper scans by name and
    removes any previous instance before registering a fresh one.
    """
    if hermes_home is None:
        return
    if not _HAS_HERMES_CRON:
        return
    try:
        from cron.jobs import list_jobs, remove_job

        for job in list_jobs():
            if job.get("name") == "cashew-sleep-cycle":
                remove_job(job["id"])
                logger.info("sleep: removed duplicate cron job %s", job["id"])
    except ImportError:
        logger.debug("sleep: cron module not available — skipping dedup")
    except Exception:
        logger.warning("sleep: failed to dedup cron jobs", exc_info=True)


# ── on_pre_compress prompt template ──────────────────────────────────
class CashewMemoryProvider(MemoryProvider):  # type: ignore[misc]
    """Cashew thought-graph memory provider for Hermes Agent."""

    def __init__(self) -> None:
        # ContextRetriever: constructed in initialize() once _db_path is
        # known, cleared in shutdown(). None in half-state (corrupt
        # config) so prefetch() + handle_tool_call() short-circuit uniformly.
        self._retriever: "ContextRetriever | None" = None
        # Lifecycle state (None until initialize() runs)
        self._hermes_home: pathlib.Path | None = None
        self._config: CashewConfig | None = None
        self._db_path: pathlib.Path | None = None
        self._sync_queue: queue.Queue | None = None
        self._session_id: str = ""
        self._sync_worker: "threading.Thread | None" = None
        # Non-daemon worker that drains _sync_queue. Started in
        # initialize() only on happy path. Joined in shutdown()
        # with bounded timeout.
        self._dropped_turn_count: int = 0
        # Monotonic counter of drop-oldest events on the sync queue.
        # Incremented inside sync_turn's overflow branch each time a queued
        # turn is evicted to make room for a new one.
        self._model_fn: Callable[[str], str] | None = None
        # Prefetch warm cache: cue → formatted context string.
        # Populated by queue_prefetch(), consumed by prefetch(), cleared in
        # shutdown(). Ephemeral per-turn state — never persisted.
        self._warm_cache: dict[str, str] = {}
        # Staging slot for background prefetch results. The queue_prefetch
        # background thread writes here; prefetch() atomically swaps it into
        # _warm_cache at the start of its call. This avoids concurrent access
        # between the daemon thread and the main agent loop.
        self._prefetch_pending: str | None = None
        # Last assistant response, buffered from sync_turn for use by
        # queue_prefetch's LLM cue extraction.
        self._last_assistant: str = ""
        # Cron job ID for the sleep cycle scheduler. Set in
        # _register_sleep_cron(), cleared in shutdown(). None when
        # sleep scheduling is disabled or registration failed.
        self._sleep_cron_job_id: str | None = None
        # Shutdown flag: set by shutdown() before the sentinel is posted, and
        # checked by _drain_once to abort early if the interpreter is shutting
        # down (avoids RuntimeError from sentence-transformers' atexit handler
        # racing with Python's shutdown sequence).
        self._shutdown_flag = threading.Event()

    @property
    def name(self) -> str:
        return "cashew"

    def is_available(self) -> bool:
        """Return True iff the provider can be initialized.

        Contract (strict):
        - When ``_hermes_home`` is known, check whether the config file exists there.
        - When ``_hermes_home`` is unknown, do NOT probe the real Hermes home.
          In that state the provider should answer based only on whether the
          plugin dependencies are importable.
        """
        if self._hermes_home is not None:
            return resolve_config_path(self._hermes_home).exists()

        # When the home is unknown, fall back to dependency availability only.
        return ContextRetriever is not None

    def get_config_schema(self) -> list[dict[str, Any]]:
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
        bounded sync queue and starts the sync worker. Creates the
        ContextRetriever lazily.

        Cashew/load failures degrade silently:
        WARNING is logged with exc_info=True and self._config stays None;
        is_available() will still answer correctly because it probes the file
        directly, not self._config.
        """
        if "hermes_home" not in kwargs:
            raise KeyError(
                "CashewMemoryProvider.initialize requires hermes_home in kwargs; "
                "Hermes Agent passes it as a keyword. Got: "
                + repr(sorted(kwargs.keys()))
            )
        self._session_id = session_id
        self._hermes_home = pathlib.Path(kwargs["hermes_home"])
        # Queue is created here so config-driven sizing/timeout values are wired in.
        # The non-daemon worker thread drains it (see CLAUDE.md ### Threading Rule).
        self._sync_queue = queue.Queue(maxsize=16)
        # Reset shutdown flag — a prior shutdown() may have set it.
        self._shutdown_flag.clear()
        with trace_operation("cashew.initialize") as span:
            span.set_attribute("session_id", session_id)
            try:
                self._config = load_config(self._hermes_home)
                # Propagate embedding model to upstream cashew-brain.
                # PyPI v1.1.0 hardcodes DEFAULT_MODEL = "all-MiniLM-L6-v2" and
                # EMBEDDING_DIM = 384; the embedded get_default_service() singleton
                # is created with those values. We patch the module-level constants
                # before any end_session() / embed_nodes() call so the upstream
                # creates 1024-dim embeddings matching our config.
                _patch_upstream_embedding(self._config.embedding_model)
                # First-load bootstrap: generate default cashew.json and
                # auto-populate auxiliary.memory if absent. Safe to call
                # on every initialize() — no-op after the first run.
                _ensure_config_file(self._hermes_home)
                if self._config.llm_aux_role:
                    _ensure_auxiliary_memory(self._hermes_home)
                self._db_path = resolve_db_path(
                    self._hermes_home, self._config.cashew_db_path
                )
                # ContextRetriever.__init__ is lazy — no SQLite open, no embedding load yet.
                # Guard against the defensive-import fallback (ContextRetriever = None).
                if ContextRetriever is None:
                    raise RuntimeError(
                        "core.context.ContextRetriever unavailable at import time; "
                        "cashew-brain dependency missing"
                    )
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                # Self-healing — clean up stale state from prior crashes.
                self._heal_stale_lock()
                self._ensure_db_schema(self._db_path)
                self._retriever = ContextRetriever(db_path=str(self._db_path))
                self._model_fn = self._build_model_fn()
                # Start the sync worker AFTER all worker-read state
                # is populated (db_path, session_id, sync_queue).
                self._start_sync_worker()
                # Register the sleep cycle cron job if configured.
                # Runs AFTER the sync worker so the provider is fully initialized
                # before any background work begins. Silently skips when the
                # Hermes cron module is not available (e.g. CI, standalone tests).
                if (
                    self._config.sleep_cycles
                    and self._config.sleep_schedule
                    and _HAS_HERMES_CRON
                ):
                    self._register_sleep_cron()
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

        MUST run AFTER self._db_path / self._session_id / self._sync_queue are set.
        daemon=False is load-bearing (see CLAUDE.md Threading Rule).
        """
        self._sync_worker = threading.Thread(
            target=self._worker_loop,
            name=f"cashew-sync-{self._session_id}",
            daemon=True,
        )
        self._sync_worker.start()

    # ── Sleep cycle cron scheduling ──────────────────────────────────────

    def _register_sleep_cron(self) -> None:
        """Install the cron script and register a no_agent cron job.

        Called from initialize() only on the happy path.  Safe to call
        multiple times — if a job is already registered for this provider
        instance, the call is a no-op.

        The cron job persists across session boundaries (survives shutdown)
        so the 12h schedule isn't reset on every session start.
        """
        if self._sleep_cron_job_id is not None:
            return  # already registered for this instance

        if self._hermes_home is None or self._config is None:
            return

        # Scan for an existing sleep cron job by name.
        # If one exists, adopt its ID and skip registration so the
        # original 12h timer isn't reset. Without this, every session
        # start would remove-and-re-register the job, resetting the
        # schedule and preventing it from ever firing.
        try:
            from cron.jobs import list_jobs

            for job in list_jobs():
                if job.get("name") == "cashew-sleep-cycle":
                    self._sleep_cron_job_id = job["id"]
                    logger.info(
                        "sleep: adopted existing cron job %s (preserving schedule)",
                        job["id"],
                    )
                    return  # keep the existing job running
        except ImportError:
            logger.debug("sleep: cron module not available — cannot adopt")
        except Exception:
            logger.warning("sleep: failed to adopt existing cron job", exc_info=True)

        try:
            # Read the cron script source from disk and install it.
            script_source = (
                pathlib.Path(__file__).parent / "sleep_cron_script.py"
            ).read_text()

            # Install the script to $HERMES_HOME/scripts/
            script_dest = self._hermes_home / "scripts" / "cashew-sleep-cycle.py"
            script_dest.parent.mkdir(parents=True, exist_ok=True)
            if not script_dest.exists():
                script_dest.write_text(script_source)
                script_dest.chmod(0o755)
                logger.info("sleep: installed cron script to %s", script_dest)
            else:
                logger.debug("sleep: cron script already exists at %s", script_dest)

            # Register via the Hermes cron API.
            from cron.jobs import create_job

            job = create_job(
                prompt="hermes-cashew sleep cycle",
                schedule=self._config.sleep_schedule,
                name="cashew-sleep-cycle",
                script="cashew-sleep-cycle.py",
                no_agent=True,
                repeat=None,  # forever
            )
            self._sleep_cron_job_id = job["id"]
            logger.info(
                "sleep: registered cron job %s (schedule=%s)",
                job["id"],
                self._config.sleep_schedule,
            )
        except ImportError:
            logger.warning(
                "sleep: cannot register cron job — Hermes cron module not available "
                "(schedule=%s); sleep cycles will not run automatically",
                self._config.sleep_schedule,
            )
        except Exception:
            logger.warning(
                "sleep: failed to register cron job (schedule=%s)",
                self._config.sleep_schedule,
                exc_info=True,
            )
            self._sleep_cron_job_id = None

    def _remove_sleep_cron(self) -> None:
        """Deregister the sleep cycle cron job.

        Called from shutdown().  Safe to call even when no job was registered.
        """
        if self._sleep_cron_job_id is None:
            return
        try:
            from cron.jobs import remove_job

            remove_job(self._sleep_cron_job_id)
            logger.info("sleep: removed cron job %s", self._sleep_cron_job_id)
        except Exception:
            logger.warning(
                "sleep: failed to remove cron job %s",
                self._sleep_cron_job_id,
                exc_info=True,
            )
        finally:
            self._sleep_cron_job_id = None

    # LLM integration via auxiliary.memory convention
    # ------------------------------------------------------------------

    def _build_model_fn(self) -> Callable[[str], str] | None:
        """Construct an LLM callable from the configured auxiliary.memory role.

        Delegates to ``config.resolve_model_fn()`` which reads Hermes'
        ``config.yaml`` and resolves the API key from config or well-known
        env vars. Returns None when:
        - No llm_aux_role is configured (heuristic-only mode)
        - The auxiliary section or API key cannot be found (logs warning)
        """
        if not self._config or not self._config.llm_aux_role:
            return None
        if self._hermes_home is None:
            return None
        return resolve_model_fn(
            hermes_home=self._hermes_home,
            config=self._config,
        )

    def sync_turn(
        self, user_content: str, assistant_content: str, session_id: str = ""
    ) -> None:
        """Hot-path enqueue of a completed turn.

        Contract: returns in <10ms. Never raises. If the queue is full, drops the
        OLDEST queued turn, logs a WARNING, and enqueues the new one (drop-oldest
        policy). If somehow still full after the drop (rare worker-draining race),
        drops the NEW turn with a second WARNING.

        Half-state (_sync_queue is None) is a silent no-op.
        """
        # Buffer assistant content for queue_prefetch cue extraction.
        # Must happen BEFORE the half-state guard so the most recent turn's
        # assistant content is always available, even if the queue is not.
        if assistant_content:
            self._last_assistant = assistant_content
        if self._sync_queue is None:
            return  # not initialized or silent-degraded; no worker to feed
        turn = (user_content, assistant_content, session_id)
        try:
            self._sync_queue.put_nowait(turn)
        except queue.Full:
            # Drop-oldest policy.
            try:
                self._sync_queue.get_nowait()
                self._sync_queue.task_done()  # balance the drop (exactly once)
            except queue.Empty:
                pass  # worker drained between Full and get_nowait — rare race; no-op
            self._dropped_turn_count += 1
            _METRICS.record_sync_dropped()
            logger.warning(
                "cashew sync queue overflow (maxsize=%d); dropped oldest turn",
                self._sync_queue.maxsize,
            )
            try:
                self._sync_queue.put_nowait(turn)
            except queue.Full:
                logger.warning(
                    "cashew sync queue still full after drop-oldest; dropping new turn"
                )

    def _worker_loop(self) -> None:
        """Background drain loop. Entry point for self._sync_worker.

        Sentinel check BEFORE try (must not be reachable from the exception
        path). task_done() ALWAYS in finally. Per-iteration except catches all
        Cashew failures without poisoning the queue.

        When ``experimental_batch_sync`` feature flag is enabled, drains up to
        ``_BATCH_SIZE`` items per iteration instead of one-at-a-time, reducing
        per-turn overhead.

        Binds the queue reference to a local `q` at loop entry. If shutdown()
        times out waiting for this worker, it clears `self._sync_queue = None`
        and abandons the worker. Without this local bind, the abandoned worker's
        `finally: self._sync_queue.task_done()` would raise AttributeError on
        NoneType. Using `q` keeps task_done() bound to the queue the worker was
        actually draining, race-free.
        """
        q = (
            self._sync_queue
        )  # bind once; shutdown may clear self._sync_queue before we exit
        assert q is not None  # invariant: worker only starts when queue exists
        _BATCH_SIZE = 8
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                q.task_done()
                return
            items = [item]
            # Batch drain when feature flag is enabled
            if self._config is not None and is_feature_enabled(
                self._config, "experimental_batch_sync"
            ):
                for _ in range(_BATCH_SIZE - 1):
                    try:
                        extra = q.get_nowait()
                    except queue.Empty:
                        break
                    if extra is _SHUTDOWN:
                        q.task_done()
                        items.append(extra)
                        break
                    items.append(extra)
            for turn in items:
                if turn is _SHUTDOWN:
                    q.task_done()
                    return
                try:
                    with trace_operation("cashew.sync") as span:
                        span.set_attribute("user.length", len(turn[0]))
                        self._drain_once(turn)
                except Exception:
                    _METRICS.record_sync_failure()
                    logger.warning("cashew sync worker: turn failed", exc_info=True)
                finally:
                    q.task_done()
                    _METRICS.set_queue_depth(q.qsize())

    def _heal_stale_lock(self) -> None:
        """Remove stale sleep-cycle lock files left by prior crashes.

        The sleep cycle creates ``brain.db.sleep.lock`` while running. If the
        process is killed (SIGKILL, power loss, OOM) before the lock is
        released, the file persists and blocks future sleep cycles.

        Rule: any lock file older than 60 minutes is considered stale and
        removed silently.  Newer locks are left alone (the sleep cycle may
        still be running in another process).
        """
        import time

        stale_threshold = 3600  # 60 minutes in seconds
        if self._db_path is None:
            return
        lock_path = self._db_path.parent / "brain.db.sleep.lock"
        if lock_path.exists():
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_threshold:
                lock_path.unlink(missing_ok=True)
                logger.info(
                    "cashew self-heal: removed stale sleep lock (age=%.0fs)", age
                )
            elif age > 0:
                logger.debug(
                    "cashew self-heal: sleep lock is fresh (age=%.0fs) — leaving in place",
                    age,
                )

    def _ensure_db_schema(self, db_path: pathlib.Path) -> None:
        """Create or migrate Cashew schema tables.

        Delegates to cashew-brain's core.db.ensure_schema() which handles
        upstream table creation (thought_nodes, derivation_edges, embeddings,
        hotspots, metrics), column migrations, index creation, and schema
        version stamping (PRAGMA user_version = 3). Then applies hermes-specific
        extensions (vec_embeddings virtual table for sqlite-vec).
        """
        from core.db import ensure_schema

        ensure_schema(str(db_path))

        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            try:
                conn.enable_load_extension(True)
                try:
                    import sqlite_vec

                    sqlite_vec.load(conn)
                except (ImportError, AttributeError):
                    conn.load_extension("vec0")
            except Exception:
                pass  # sqlite-vec not available at platform level; graceful degradation active
            self._migrate_vec_embeddings(conn)
            self._create_vec_embeddings(conn)
            # Hermes provider metadata store (persistent counters, flags)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hermes_provider_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _migrate_vec_embeddings(self, conn: sqlite3.Connection) -> None:
        """Migrate vec_embeddings from old schema (no node_id, no distance_metric)
        to the canonical schema matching upstream cashew-brain v1.1.0.

        Old schema:  USING vec0(embedding float[384])
        New schema:  USING vec0(node_id TEXT primary key, embedding float[384] distance_metric=cosine)

        The old schema caused _retrieve_with_vec to return rowid values that never
        matched thought_nodes.id (SHA hashes), making vec search always return 0.
        """
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
            )
            if cursor.fetchone() is None:
                return
            has_node_id = False
            try:
                conn.execute("SELECT node_id FROM vec_embeddings LIMIT 1")
                has_node_id = True
            except Exception:
                pass
            if not has_node_id:
                logger.info(
                    "Migrating vec_embeddings from old schema (dropping and recreating)"
                )
                conn.execute("DROP TABLE vec_embeddings")
        except Exception:
            logger.info(
                "sqlite-vec extension failed to load; vec_embeddings migration skipped"
            )

    def _create_vec_embeddings(self, conn: sqlite3.Connection) -> None:
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
            )
            if cursor.fetchone() is not None:
                return
            conn.enable_load_extension(True)
            try:
                import sqlite_vec

                sqlite_vec.load(conn)
            except (ImportError, AttributeError):
                conn.load_extension("vec0")
            # Resolve the configured embedding model's dimension at runtime
            # instead of hardcoding float[384] — supports gte-large (1024),
            # gte-base (768), MiniLM (384), etc. Without this, vec_embeddings
            # silently refuses dual-writes from any model with a different dim.
            try:
                from core.embedding_service import resolve_embedding_dim

                dim = resolve_embedding_dim()
            except Exception:
                dim = 384  # fallback for test environments without models
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
                USING vec0(node_id TEXT primary key, embedding float[{dim}] distance_metric=cosine)
            """)
            logger.debug(f"vec_embeddings virtual table ready (dim={dim})")
        except Exception:
            logger.info(
                "sqlite-vec extension failed to load; semantic search will use fallback"
            )

    def _enrich_results(self, node_ids: list[str]) -> list[dict]:
        """Fetch full node dicts from DB for upstream retrieval results.

        Upstream RetrievalResult carries core fields (id, content, type, domain),
        but Hermes-specific formatting needs permanent flag, tags, referent_time
        etc. Batch-query the DB to get these.
        """
        if not node_ids:
            return []
        import sqlite3

        conn = sqlite3.connect(str(self._db_path))
        try:
            placeholders = ",".join("?" * len(node_ids))
            cursor = conn.execute(
                f"SELECT * FROM thought_nodes WHERE id IN ({placeholders})", node_ids
            )
            rows = cursor.fetchall()
            cols = [col[0] for col in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            conn.close()

    def _format_context(self, nodes: list[dict]) -> str:
        if not nodes:
            return ""
        lines = ["=== RELEVANT CONTEXT ==="]
        for node in nodes:
            domain = node.get("domain")
            node_type = node.get("node_type")
            content = node.get("content", "")
            permanent = node.get("permanent") == 1
            labels = []
            if domain:
                labels.append(f"domain: {domain}")
            if node_type:
                labels.append(f"type: {node_type}")
            if permanent:
                labels.append("permanent")
            if labels:
                prefix = f"[{' | '.join(labels)}]"
                lines.append(f"{prefix} {content}")
            else:
                lines.append(content)
        return "\n".join(lines)

    def _update_access_metrics(self, node_ids: list[str]) -> None:
        if not node_ids:
            return
        try:
            import sqlite3

            conn = sqlite3.connect(str(self._db_path))
            try:
                placeholders = ",".join("?" * len(node_ids))
                conn.execute(
                    f"""
                    UPDATE thought_nodes
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                    """,
                    node_ids,
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.warning("cashew access metrics update failed", exc_info=True)

    def _drain_once(self, turn: tuple[str, str, str]) -> None:
        """Persist one turn via Cashew's heuristic extractor (or LLM if configured).

        Lazy-imports core.session so the plugin module loads even when cashew-brain
        is not installed. When self._model_fn is set (via llm_aux_role config),
        upstream receives the LLM callable and can perform LLM extraction, think
        cycles, and sleep synthesis. When None, Cashew's built-in heuristic
        extractor is used (no LLM round-trip).

        Retries up to 3 times on SQLITE_BUSY with exponential backoff before
        dropping the turn."""
        from core.session import end_session  # lazy import

        user, assistant, session_id = turn

        # Short-circuit if shutdown is in progress — the interpreter's atexit
        # handlers may have already finalized the embedding model's thread pool,
        # and calling end_session would raise RuntimeError from sentence-transformers.
        if self._shutdown_flag.is_set():
            logger.debug("cashew sync: shutdown flag set, dropping turn")
            return

        import sqlite3

        max_retries = 3
        for attempt in range(max_retries):
            try:
                end_session(
                    db_path=str(self._db_path),
                    session_id=session_id or self._session_id,
                    conversation_text=f"User: {user}\nAssistant: {assistant}",
                    model_fn=self._model_fn,
                )
                _METRICS.record_sync_success()
                break  # success
            except RuntimeError as e:
                # Python interpreter shutdown: sentence-transformers' thread pool
                # was finalized by atexit handlers. There is no recovery — drop
                # the turn silently and let the worker loop terminate naturally.
                msg = str(e)
                if "can't register atexit after shutdown" in msg:
                    logger.info("cashew sync: interpreter shutting down, dropping turn")
                    self._shutdown_flag.set()
                    return
                raise
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    wait = 0.5 * (2**attempt)
                    logger.info(
                        "cashew sync: database locked, retrying in %.1fs (attempt %d/%d)",
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait)
                else:
                    raise  # give up — let the outer except in _worker_loop handle it
        # Run think cycle periodically if LLM is wired
        if (
            self._model_fn is not None
            and self._config
            and self._config.think_interval > 0
        ):
            counter = self._load_think_counter() + 1
            if counter >= self._config.think_interval:
                counter = 0
                try:
                    from core.session import think_cycle

                    result = think_cycle(
                        db_path=str(self._db_path),
                        model_fn=self._model_fn,
                    )
                    if result.new_nodes:
                        logger.info(
                            "think cycle produced %d insight(s) on cluster: %s",
                            len(result.new_nodes),
                            result.cluster_topic or "unknown",
                        )
                except Exception:
                    logger.warning("think cycle failed", exc_info=True)
            self._save_think_counter(counter)

    def _load_think_counter(self) -> int:
        """Read persistent think counter from DB. Resets to 0 on any error."""
        try:
            import sqlite3

            conn = sqlite3.connect(str(self._db_path))
            try:
                row = conn.execute(
                    "SELECT value FROM hermes_provider_meta WHERE key='think_counter'"
                ).fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception:
            return 0

    def _save_think_counter(self, value: int) -> None:
        """Write persistent think counter to DB."""
        try:
            import sqlite3

            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO hermes_provider_meta (key, value) VALUES ('think_counter', ?)",
                    (str(value),),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def on_pre_compress(self, messages: list) -> str:
        """Extract forest-level conversation-arc insights before compression.

        Uses a dedicated LLM prompt (different from end_session's per-turn
        prompt) to identify topic shifts, framing changes, implicit decisions,
        unstated subjects, and recurring patterns that only become visible
        across multiple turns.

        Persists insight/observation nodes to the Cashew graph and returns
        a short summary string for the compressor.

        Silent-degrades to "" when no LLM is wired, insufficient exchanges,
        or any failure (never raises).
        """
        if self._model_fn is None or self._db_path is None:
            return ""

        import json as _json

        exchanges = self._extract_exchanges(messages)
        # Need at least 3 user+assistant exchanges for arc detection
        if len(exchanges) < 6:
            return ""

        # Cap at 20 most recent messages to bound prompt cost
        messages_text = "\n\n".join(exchanges[-20:])

        try:
            user_domain = self._config.user_domain if self._config else "user"
            ai_domain = self._config.ai_domain if self._config else "ai"

            prompt = PRE_COMPRESS_PROMPT_TEMPLATE.format(
                user_domain=user_domain,
                ai_domain=ai_domain,
                messages_text=messages_text,
            )

            response = self._model_fn(prompt)
            if not response or not response.strip():
                return ""

            # Parse JSON — handle markdown code fences (same pattern as end_session)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                start = cleaned.find("[")
                if start == -1:
                    return ""
                end = cleaned.rfind("]")
                if end == -1:
                    return ""
                cleaned = cleaned[start : end + 1]

            items = _json.loads(cleaned)
            if not isinstance(items, list):
                return ""

            # Filter to items marked for retention
            items = [it for it in items if it.get("keep", True)]
            if not items:
                return ""

            # Persist to graph
            created = self._create_insight_nodes(items)
            if created == 0:
                return ""

            # Build summary string for compressor
            summaries = []
            for item in items[:3]:
                content = item.get("content", "")
                if content:
                    summaries.append(f"- {content[:200]}")
            if summaries:
                return "Cashew insight extraction:\n" + "\n".join(summaries)
            return ""

        except _json.JSONDecodeError:
            logger.warning(
                "on_pre_compress: failed to parse LLM response", exc_info=True
            )
            return ""
        except Exception:
            logger.warning("on_pre_compress failed", exc_info=True)
            return ""

    def _extract_exchanges(self, messages: list) -> list[str]:
        """Extract user/assistant text exchanges from OpenAI-format messages.

        Handles multimodal content (list-of-parts format) by extracting only
        text parts. Filters out system and tool messages. Returns a list of
        "role: content" strings in conversation order.
        """
        exchanges: list[str] = []
        for msg in messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal: extract text parts only
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if text:
                            parts.append(text)
                content = " ".join(parts)

            if isinstance(content, str) and content.strip():
                exchanges.append(f"{role}: {content.strip()}")
        return exchanges

    def _create_insight_nodes(self, items: list[dict]) -> int:
        """Create insight/observation nodes in the Cashew graph.

        Uses upstream _create_node / _set_node_tags for persistence, then
        calls embed_nodes to generate embeddings. Returns node count.

        Silent-degrades: logs warning on failure, never raises.
        """
        from core.embeddings import embed_nodes
        from core.session import _create_node, _set_node_tags

        db_path = str(self._db_path)
        count = 0

        for item in items:
            content = item.get("content", "")
            node_type = item.get("type", "insight")
            domain = item.get("domain", "user")
            tags = item.get("tags", [])

            if not content or not content.strip():
                continue

            try:
                node_id = _create_node(
                    db_path=db_path,
                    content=content.strip(),
                    node_type=node_type,
                    session_id="pre_compress",
                    domain=domain,
                )
                if tags and isinstance(tags, list):
                    _set_node_tags(db_path, node_id, tags)
                count += 1
            except Exception:
                logger.warning("on_pre_compress: failed to create node", exc_info=True)
                continue

        if count > 0:
            try:
                embed_nodes(db_path)
            except Exception:
                logger.warning("on_pre_compress: embedding failed", exc_info=True)

        return count

    def on_session_end(self, messages: list) -> None:
        """Session boundary notification.

        Does NOT drain the sync queue — the background worker is non-daemon and
        keeps running across session boundaries. Data-loss protection is handled
        by shutdown(), which posts a sentinel and bounded-joins the worker.

        Sleep cycle processing is handled by a Hermes cron job.
        See ``sleep_schedule`` in cashew.json.
        """
        if self._sync_queue is None:
            return  # not initialized or silent-degraded

    def shutdown(self) -> None:
        """Post sentinel, bounded-join worker, clear references.

        Order is load-bearing:
          1. If not initialized, return.
          2. Post _SHUTDOWN sentinel to the queue. put_nowait first; fallback to
             a 1s blocking put if the queue is full (worker is draining fast).
          3. Bounded-join the worker using sync_queue_timeout. WARNING on timeout.
          4. Clear _sync_queue, _sync_worker, _config, _db_path, _retriever.

        _hermes_home is intentionally NOT reset — is_available() must keep
        reflecting on-disk reality.
        """
        if self._sync_queue is None:
            return  # safe no-op: initialize() was never called
        _METRICS.emit()
        timeout = self._config.sync_queue_timeout if self._config is not None else 30.0
        # Signal shutdown BEFORE the sentinel so _drain_once can short-circuit
        # even if the sentinel is still queued behind pending turns.
        self._shutdown_flag.set()
        # Post sentinel. put_nowait first; if somehow full, try a brief blocking put.
        try:
            self._sync_queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            try:
                self._sync_queue.put(_SHUTDOWN, block=True, timeout=1.0)
            except queue.Full:
                logger.warning(
                    "cashew shutdown: could not post sentinel; worker may leak"
                )
        # Bounded join. Never raise.
        if self._sync_worker is not None:
            self._sync_worker.join(timeout=timeout)
            if self._sync_worker.is_alive():
                logger.warning(
                    "cashew sync worker did not exit within %ss; abandoning",
                    timeout,
                )
        # Sleep cycle cron job is intentionally NOT removed here.
        # It persists across session boundaries so the 12h schedule
        # isn't reset on every session start. The next initialize()
        # will adopt the existing job if one exists.
        self._sleep_cron_job_id = None  # clear instance tracking only
        # Clear state. _hermes_home persists (see is_available() contract).
        self._sync_queue = None
        self._sync_worker = None
        self._config = None
        self._db_path = None
        self._retriever = None
        self._warm_cache.clear()
        self._prefetch_pending = None
        self._last_assistant = ""
        logger.debug("cashew provider shutdown complete")

    def prefetch(
        self,
        query: str,
        domain: str | None = None,
        tag: str | None = None,
        exclude_tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return recalled-context string from Cashew (RECALL-01).

        Checks the warm cache (populated by queue_prefetch) first. On a cache
        hit, returns the cached context immediately without hitting storage.
        On a cache miss, delegates to upstream cashew-brain's
        retrieve_recursive_bfs for full three-tier retrieval.

        Delegates to upstream cashew-brain's retrieve_recursive_bfs which handles
        the full three-tier retrieval (sqlite-vec semantic search → graph BFS →
        keyword fallback) with hybrid scoring. Hermes-specific fields (permanent
        flag, tags) are enriched from the DB before formatting.

        Falls back to SQL LIKE keyword search when upstream retrieval fails
        (e.g. sqlite-vec or sentence-transformers unavailable in test environment).

        Contract:
        - Half-state guard: if `_config is None`, return `""` without logging.
        - Empty result is valid (returns `""` without logging).
        - Failure path: `except Exception` logs ONE WARNING and returns `""`.
        """
        if self._config is None:
            return ""
        # Atomically swap in any background-prefetched results
        pending = self._prefetch_pending
        if pending is not None:
            self._prefetch_pending = None
            # Store under the raw query so the matching logic below can find it
            self._warm_cache[query] = pending
        # Warm cache fast path: check if a cached cue matches the query.
        if self._warm_cache:
            query_lower = query.lower()
            for cue, ctx in self._warm_cache.items():
                if not cue:
                    continue
                cue_lower = cue.lower()
                if cue_lower in query_lower or query_lower in cue_lower:
                    logger.info("prefetch warm cache HIT: cue=%r query=%r", cue, query)
                    self._warm_cache.clear()
                    return ctx
                cue_words = set(w for w in cue_lower.split() if len(w) > 3)
                query_words = set(w for w in query_lower.split() if len(w) > 3)
                if len(cue_words & query_words) >= 2:
                    logger.info(
                        "prefetch warm cache HIT: cue=%r query=%r (word overlap)",
                        cue,
                        query,
                    )
                    self._warm_cache.clear()
                    return ctx
            # No match — clear stale cache and fall through to cold retrieval.
            logger.info(
                "prefetch warm cache MISS (%d cue(s) in cache) — falling through to cold retrieval",
                len(self._warm_cache),
            )
            self._warm_cache.clear()
        max_nodes = self._config.recall_k
        with trace_operation(
            "cashew.prefetch",
            {"query.length": len(query)},
        ) as _span:
            try:
                from core.retrieval import retrieve_recursive_bfs

                results = retrieve_recursive_bfs(
                    db_path=str(self._db_path),
                    query=query,
                    top_k=max_nodes,
                    domain=domain,
                    tags=[tag] if tag else None,
                    exclude_tags=exclude_tags,
                )
                if results:
                    node_ids = [r.node_id for r in results]
                    self._update_access_metrics(node_ids)
                    nodes = self._enrich_results(node_ids)
                    return self._format_context(nodes)
            except Exception:
                logger.debug(
                    "upstream retrieval failed, falling back to keyword", exc_info=True
                )
            try:
                nodes = self._keyword_search(
                    query, max_nodes, domain, tag, exclude_tags
                )
                if nodes:
                    self._update_access_metrics([n["id"] for n in nodes])
                    return self._format_context(nodes)
            except Exception:
                logger.warning(
                    "cashew recall failed for query=%r", query, exc_info=True
                )
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm cashew memory for the next turn (ABC optional hook).

        Dispatches a background thread so this returns immediately and never
        blocks the user's turn. The thread runs vector search (and optionally
        LLM cue extraction) and populates _warm_cache when done.

        If the background thread hasn't finished before the next prefetch(),
        prefetch falls through to cold storage transparently. Results are
        stored in a staging slot (_prefetch_pending) and swapped atomically
        into _warm_cache at the start of the next prefetch to avoid
        concurrent access between thread and main loop.

        Contract:
        - Half-state guard: if _config is None, return silently.
        - Never blocks — returns in <1ms.
        - Never raises into Hermes (caught in background thread).
        """
        if self._config is None:
            return
        self._prefetch_pending = None  # clear any stale results
        if not query:
            logger.debug("queue_prefetch: empty query, no warmup")
            return

        # Capture the state the thread needs — these are stable after initialize()
        db_path = str(self._db_path)
        top_k = self._config.prefetch_k
        use_llm = self._model_fn is not None and self._config.prefetch_cues > 0

        def _warmup_worker() -> None:
            """Background thread: retrieve + optionally refine with LLM."""
            try:
                # If LLM is available, extract cues for better retrieval
                if use_llm:
                    try:
                        cues = self._extract_prefetch_cues(query)
                        logger.info(
                            "queue_prefetch: extracted %d LLM cue(s)",
                            len(cues),
                        )
                    except Exception:
                        logger.debug(
                            "queue_prefetch: LLM cue extraction failed, using raw query"
                        )
                        cues = [query] if query else []
                else:
                    cues = [query] if query else []

                if not cues:
                    return

                from core.retrieval import retrieve_recursive_bfs

                seen_ids: set[str] = set()
                all_nodes: list[dict] = []
                for cue in cues:
                    results = retrieve_recursive_bfs(
                        db_path=db_path,
                        query=cue,
                        top_k=top_k,
                    )
                    if results:
                        node_ids = [r.node_id for r in results]
                        nodes = self._enrich_results(node_ids)
                        for n in nodes:
                            nid = n.get("id", "")
                            if nid not in seen_ids:
                                seen_ids.add(nid)
                                all_nodes.append(n)

                if all_nodes:
                    ctx = self._format_context(all_nodes)
                    # Stage results for the next prefetch to pick up
                    self._prefetch_pending = ctx
                    logger.info(
                        "queue_prefetch: cached %d result(s) from %d cue(s) for next turn",
                        len(all_nodes),
                        len(cues),
                    )
            except Exception:
                logger.debug(
                    "queue_prefetch background worker failed (non-fatal)", exc_info=True
                )

        t = threading.Thread(
            target=_warmup_worker,
            daemon=True,
            name=f"cashew-prefetch-{self._session_id}",
        )
        t.start()

    def _extract_prefetch_cues(self, query: str) -> list[str]:
        """Use the auxiliary LLM to extract concrete search cues from the turn.

        Transforms the conversational turn into 2-3 concrete noun phrases
        suitable for semantic search. Uses both the user message (query) and
        the buffered assistant response (_last_assistant) for context.

        Returns:
            List of search cue strings (may be empty).

        Raises:
            Exception: forwarded to caller for logging.
        """
        if not self._model_fn:
            return [query] if query else []
        assistant = self._last_assistant or ""
        n = max(1, self._config.prefetch_cues) if self._config else 3
        prompt = (
            "Extract up to {} concrete search queries from this conversation "
            "turn that would find relevant semantic memory for what is likely "
            "to come next. Each search query must be a short noun phrase (not "
            "a question).\n\n"
            "User: {}\n"
            "Assistant: {}\n\n"
            "Respond with one search query per line. No numbering, no prefixes."
        ).format(n, query, assistant)
        raw = self._model_fn(prompt)
        cues = [
            line.strip()
            for line in raw.strip().split("\n")
            if line.strip() and not line.strip().startswith(("```", "Here", "Sure"))
        ]
        return cues[:n] if cues else [query] if query else []

    def _keyword_search(
        self,
        query: str,
        max_nodes: int,
        domain: str | None = None,
        tag: str | None = None,
        exclude_tags: list[str] | None = None,
    ) -> list[dict]:
        import sqlite3

        conn = sqlite3.connect(str(self._db_path))
        try:
            where_clauses: list[str] = ["(decayed IS NULL OR decayed = 0)"]
            params: list = []
            words = [w for w in query.split() if w]
            if words:
                where_clauses.append(
                    "(" + " AND ".join(["content LIKE ?"] * len(words)) + ")"
                )
                params.extend(f"%{w}%" for w in words)
            if domain:
                where_clauses.append("domain = ?")
                params.append(domain)
            if tag:
                where_clauses.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if exclude_tags:
                for ex_tag in exclude_tags:
                    if ex_tag:
                        where_clauses.append("(tags IS NULL OR tags NOT LIKE ?)")
                        params.append(f"%{ex_tag}%")
            order_params: list = []
            if words:
                order_params.append(f"%{query}%")
            cursor = conn.execute(
                f"""
                SELECT * FROM thought_nodes
                WHERE {" AND ".join(where_clauses) if where_clauses else "1=1"}
                ORDER BY
                    {"(CASE WHEN content LIKE ? THEN 1 ELSE 0 END) DESC," if order_params else ""}
                    referent_time DESC NULLS LAST,
                    timestamp DESC
                LIMIT ?
                """,
                (*params, *order_params, max_nodes),
            )
            rows = cursor.fetchall()
            cols = [col[0] for col in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            conn.close()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return the list of LLM tool schemas this provider exposes.

        Two tools — cashew_query (recall) and cashew_extract (explicit sync).
        Schema structure follows OpenAI's parameters convention.

        The returned list is a fresh list literal each call, but the schema dicts
        themselves are module constants (not copies) — callers must not mutate.
        """
        return [CASHEW_QUERY_SCHEMA, CASHEW_EXTRACT_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """Route an LLM tool call to the Cashew backend.

        Two tools are handled:
          - cashew_query: recall from the thought graph.
          - cashew_extract: explicit, synchronous extraction of
            one turn. Bypasses the sync queue — returns only after Cashew
            completes.

        Silent-degrade paths:
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
            if self._config is None:
                return build_error_envelope(
                    query=args.get("query"),
                    error_message="cashew recall failed",
                )
            try:
                _t0 = time.perf_counter()
                query = args["query"]
                with trace_operation(
                    "cashew.query",
                    {"query.length": len(query)},
                ) as span:
                    max_nodes = args.get("max_nodes", self._config.recall_k)
                    domain = args.get("domain")
                    tag = args.get("tag")
                    exclude_tags = args.get("exclude_tags")
                    try:
                        from core.retrieval import retrieve_recursive_bfs

                        results = retrieve_recursive_bfs(
                            db_path=str(self._db_path),
                            query=query,
                            top_k=max_nodes,
                            domain=domain,
                            tags=[tag] if tag else None,
                            exclude_tags=exclude_tags,
                        )
                    except Exception:
                        results = None
                if results:
                    node_ids = [r.node_id for r in results]
                    nodes = self._enrich_results(node_ids)
                else:
                    nodes = self._keyword_search(
                        query, max_nodes, domain, tag, exclude_tags
                    )
                if nodes:
                    self._update_access_metrics([n["id"] for n in nodes])
                    context = self._format_context(nodes)
                    node_count = len(nodes)
                else:
                    context = ""
                    node_count = 0
                _elapsed_ms = (time.perf_counter() - _t0) * 1000
                _METRICS.record_query(cache_hit=False, elapsed_ms=_elapsed_ms)
                span.set_attribute("node_count", node_count)
                span.set_attribute("elapsed_ms", _elapsed_ms)
                return build_success_envelope(
                    query=query,
                    context=context,
                    node_count=node_count,
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
            # Half-state guard. No log — initialize() already warned when it
            # set _db_path / _config to None.
            if self._db_path is None or self._config is None:
                return build_extract_error_envelope()
            try:
                user = args["user_content"]  # KeyError caught below — tool-call failure
                assistant = args["assistant_content"]
                # Lazy import — keeps is_available free of core.session side effects.
                from core.session import end_session

                result = end_session(
                    db_path=str(self._db_path),
                    session_id=self._session_id,
                    conversation_text=f"User: {user}\nAssistant: {assistant}",
                    model_fn=self._model_fn,
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
            # Unknown-tool branch. Uses the QUERY envelope because historically
            # unknown-tool returned the cashew_query error shape. This preserves
            # backward compatibility with test_handle_tool_call.py which asserts
            # tool='cashew_query', error='unknown tool', query=None.
            logger.warning("cashew unknown tool call: %r", name)
            return build_error_envelope(query=None, error_message="unknown tool")

    def system_prompt_block(self) -> str:
        """Return a ~10-line LLM-visible status string for the system prompt.

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

        user_domain = get_user_domain(self._config)
        ai_domain = get_ai_domain(self._config)

        return (
            f"[cashew] memory provider: available\n"
            f"graph: {graph_state}\n"
            f"recall depth: {recall_k}\n"
            f"user domain: {user_domain}\n"
            f"ai domain: {ai_domain}\n"
        )

    # All other ABC methods are inherited as no-ops from the ABC defaults (when Hermes is present).


def register(ctx: Any) -> None:
    """Hermes discovery entry point — filesystem loader calls this.

    Two code paths:
    - Directory scanner (source="bundled"|"user" in plugins/memory/__init__.py):
      ctx is a _ProviderCollector with register_memory_provider.
    - Entry-point loader (source="entrypoint"): ctx is a PluginContext without
      register_memory_provider — memory providers are discovered by the
      directory scanner, not entry points.
    """
    register_fn = getattr(ctx, "register_memory_provider", None)
    if register_fn is not None:
        register_fn(CashewMemoryProvider())
