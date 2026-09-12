# plugins/memory/cashew/__init__.py
# Source: Pattern mirrored from plugins/memory/hindsight/__init__.py (NousResearch/hermes-agent@main)
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import pathlib
import queue
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, cast

from .admission import (
    OperationAdmissionError,
    admit_operation,
    current_admission,
)
from .embedding import (
    DEFAULT_EMBEDDING_DEVICE,
    normalize_embedding_device,
)
from .embedding_process import (
    EmbeddingFailure,
    EmbeddingSupervisor,
    EmbeddingUnavailable,
    NoInProcessEmbeddingBackend,
    ProcessEmbeddingBackend,
    embedding_caller_wait,
)
from .error_tracking import (
    SentryTelemetry,
    capture_exception,
    close_sentry_telemetry,
    start_sentry_telemetry,
)
from .health import OutcomeLedger, safe_error_class
from .locking import (
    MaintenanceLockAcquisitionError,
    SQLiteWALUnsupportedError,
    bootstrap_sqlite_journal,
    guard_sqlite_journal,
    lock_path_for_db,
    open_readonly_verified,
    sqlite_journal_report,
    sqlite_wal_reset_vulnerable,
    try_maintenance_lock,
    verify_readonly_profile,
)
from .log_filter import acquire_provider_scrub_filters, release_provider_scrub_filters
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
from .cron_reconcile import (
    CRON_JOB_NAME,
    CRON_SCRIPT_NAME,
    installation_marker,
    owns_job,
    profile_cron_lock,
    profile_identity,
    render_script,
    stage_script,
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


def _retrieve_with_embedding_wait(**kwargs: Any) -> Any:
    """Give interactive retrieval bounded patience without killing active work."""
    from core.retrieval import retrieve_recursive_bfs

    with embedding_caller_wait(1.5):
        return retrieve_recursive_bfs(**kwargs)


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


@dataclasses.dataclass(frozen=True)
class _PrefetchRequestIdentity:
    """The retrieval inputs a warm result must match before it can be reused."""

    generation: int
    session_id: str
    db_path: str
    config_fingerprint: str
    recall_limit: int
    domain: str | None
    tag: str | None
    exclude_tags: tuple[str, ...]
    embedding_epoch: int | None = None
    embedding_generation: str | int | None = None


@dataclasses.dataclass(frozen=True)
class _PrefetchResult:
    """An ephemeral warmup result with the request context that produced it."""

    identity: _PrefetchRequestIdentity
    cues: tuple[str, ...]
    nodes: tuple[dict[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class _PrefetchRequest:
    """One latest-request slot entry for the bounded prefetch worker."""

    identity: _PrefetchRequestIdentity
    query: str
    db_path: str
    top_k: int
    use_llm: bool


class _InitializationCancelledError(RuntimeError):
    """Private control flow for shutdown cancelling a slow initialize()."""


class _PreAdmissionRejectedError(RuntimeError):
    """A queued turn could not enter the provider generation."""


class _OpaqueUpstreamError(RuntimeError):
    """An upstream write failed after admission, with bounded outcome evidence."""

    def __init__(self, *, partial: bool) -> None:
        super().__init__("upstream persistence outcome is not replayable")
        self.partial = partial


class _GenerationBoundEmbeddingService:
    """Make one upstream service unavailable once its owner closes.

    Upstream's service returns cache hits and zero vectors without consulting a
    backend.  The outer generation gate prevents those convenience paths from
    letting a closed profile serve another profile's global singleton.
    """

    def __init__(
        self,
        service: Any,
        supervisor: EmbeddingSupervisor,
        *,
        cache_path: pathlib.Path | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self._service = service
        self._supervisor = supervisor
        self._cache_path = cache_path
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return cast(str, self._service.model)

    @property
    def dim(self) -> int:
        return cast(int, self._service.dim)

    def embed_np(self, texts: list[str]) -> Any:
        admission = current_admission()
        if admission is None and self._cache_path is not None:
            with admit_operation(
                cache_path=self._cache_path,
                model=self._model,
                embedding_dim=self._dimension,
                supervisor=self._supervisor,
                embedding_generation=getattr(self._supervisor, "generation", None),
                cache_exclusive=True,
                deadline=1.5,
            ):
                with self._supervisor.serve_generation():
                    return self._service.embed_np(texts)
        with self._supervisor.serve_generation():
            return self._service.embed_np(texts)

    def embed(self, text: Any) -> Any:
        """Keep upstream's public single-text route behind the same gate."""
        admission = current_admission()
        if admission is None and self._cache_path is not None:
            with admit_operation(
                cache_path=self._cache_path,
                model=self._model,
                embedding_dim=self._dimension,
                supervisor=self._supervisor,
                embedding_generation=getattr(self._supervisor, "generation", None),
                cache_exclusive=True,
                deadline=1.5,
            ):
                with self._supervisor.serve_generation():
                    return self._service.embed(text)
        with self._supervisor.serve_generation():
            return self._service.embed(text)


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

# Cashew-brain at dd57ef0 has no instance-scoped migration or embedding API.
# These are the only private compatibility seams retained by this adapter:
#
# * ``core.config.config.embedding_model`` selects the model used by pinned
#   migration helpers, which call ``resolve_embedding_dim()`` without args.
# * ``core.embedding_service._default_service`` is read by ``embed_nodes()``
#   without accepting a service/backend argument.
# * ``core.embedding_service._KNOWN_DIMS[model]`` prevents that no-argument
#   resolver from constructing an in-process LocalBackend for an *unknown*
#   model after the child has already verified its dimension.
#
# Retire these assignments once upstream accepts an explicit service or
# dimension in its embedding and migration entry points.  Do not add backend
# method patches, reset the singleton, or write DEFAULT_MODEL/EMBEDDING_DIM:
# those process-wide compatibility constants cannot safely represent profiles.
_UPSTREAM_COMPATIBILITY_SHIMS = (
    "core.config.config.embedding_model",
    "core.embedding_service._default_service",
    "core.embedding_service._KNOWN_DIMS[model]",
)


class _NoopEmbeddingCache:
    """Exact cache surface used when the affected SQLite runtime is unsafe."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = str(path)

    def get_many(self, model: str, texts: list[str]) -> list[None]:
        del model
        return [None for _ in texts]

    def put_many(self, model: str, pairs: list[tuple[str, Any]]) -> int:
        del model, pairs
        return 0

    def get(self, model: str, text: str) -> None:
        del model, text
        return None

    def put(self, model: str, text: str, vector: Any) -> None:
        del model, text, vector

    def size(self, model: str | None = None) -> int:
        del model
        return 0

    def invalidate_model(self, model: str) -> None:
        del model


class _GenerationBoundEmbeddingCache:
    """Reuse the admission-owned cache lease without opening another handle."""

    def __init__(
        self,
        cache: Any,
        *,
        path: pathlib.Path,
        model: str,
        embedding_dim: int,
        supervisor: Any,
        generation: str | int | None,
    ) -> None:
        self._cache = cache
        self.path = str(path)
        self._path = path.resolve(strict=False)
        self._model = model
        self._embedding_dim = embedding_dim
        self._supervisor = supervisor
        self._generation = generation

    def _check(self, model: str) -> None:
        admission = current_admission()
        if admission is None:
            raise OperationAdmissionError("embedding cache operation is not admitted")
        if isinstance(self._cache, _NoopEmbeddingCache):
            # Affected-runtime profiles deliberately carry no cache lease; the
            # facade remains a harmless exact no-op for upstream calls.
            if model != self._model:
                raise OperationAdmissionError(
                    "embedding cache admission model mismatch"
                )
            return
        if admission.cache_path != self._path:
            raise OperationAdmissionError("embedding cache admission path mismatch")
        if admission.model != model or admission.model != self._model:
            raise OperationAdmissionError("embedding cache admission model mismatch")
        if admission.embedding_dim not in (None, self._embedding_dim):
            raise OperationAdmissionError(
                "embedding cache admission dimension mismatch"
            )
        if admission.supervisor is not self._supervisor:
            raise OperationAdmissionError(
                "embedding cache admission supervisor mismatch"
            )
        if admission.embedding_generation != self._generation:
            raise OperationAdmissionError(
                "embedding cache admission generation mismatch"
            )
        if admission.cache_lease is None:
            raise OperationAdmissionError("embedding cache lease is not owned")
        # The child handshake is the source of truth for dimensions.  Keep the
        # per-model cache record in the same file scoped to this admission so a
        # stale facade cannot publish a vector before identity is durable.
        try:
            with sqlite3.connect(str(self._path)) as conn:
                row = conn.execute(
                    "SELECT embedding_dim FROM hermes_cashew_cache_meta WHERE model=?",
                    (model,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise OperationAdmissionError(
                "embedding cache identity metadata is unavailable"
            ) from exc
        if row is None or int(row[0]) != self._embedding_dim:
            raise OperationAdmissionError(
                "embedding cache model dimension metadata mismatch"
            )

    def get_many(self, model: str, texts: list[str]) -> Any:
        self._check(model)
        return self._cache.get_many(model, texts)

    def put_many(self, model: str, pairs: list[tuple[str, Any]]) -> Any:
        self._check(model)
        return self._cache.put_many(model, pairs)

    def get(self, model: str, text: str) -> Any:
        self._check(model)
        return self._cache.get(model, text)

    def put(self, model: str, text: str, vector: Any) -> Any:
        self._check(model)
        return self._cache.put(model, text, vector)

    def size(self, model: str | None = None) -> Any:
        selected = model or self._model
        self._check(selected)
        return self._cache.size(selected)

    def invalidate_model(self, model: str) -> Any:
        self._check(model)
        return self._cache.invalidate_model(model)


def _open_profile_embedding_cache(cache_path: pathlib.Path) -> Any:
    """Serialize upstream cache schema setup behind its independent lease."""
    from core.embedding_cache import EmbeddingCache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 5.0
    while True:
        with try_maintenance_lock(cache_path) as lock_fd:
            if lock_fd is not None:
                return EmbeddingCache(str(cache_path))
        if time.monotonic() >= deadline:
            raise EmbeddingUnavailable(EmbeddingFailure.BUSY)
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _bind_upstream_embedding(
    model_name: str,
    device: str = DEFAULT_EMBEDDING_DEVICE,
    *,
    cache_path: pathlib.Path | None = None,
    cache_disabled: bool = False,
) -> EmbeddingSupervisor | None:
    """Bind one active profile to the owned child embedding service.

    This deliberately publishes the three pinned upstream compatibility shims
    only after the supervisor owns the process boundary, completes its child
    handshake, and opens the profile-scoped cache.  A rejected profile therefore
    cannot alter the active profile's upstream globals.
    """
    try:
        import core.config
        import core.embedding_service
    except ImportError:
        logger.warning("cashew-brain not installed; cannot patch embedding model")
        return None

    if cache_path is None:
        raise ValueError("profile-scoped embedding cache path is required")

    expected_dim = _UPSTREAM_KNOWN_DIMS.get(model_name, 0)
    selected_device = normalize_embedding_device(device)

    supervisor = EmbeddingSupervisor(
        model=model_name,
        device=selected_device,
        dimension=expected_dim,
        cache_dir=cache_path.parent / "model-cache",
    )
    try:
        # Starting every model, including known ones, is intentional: the child
        # owns model construction and validates known-model dimensions before
        # any schema/migration write can occur.
        dim = supervisor.start()
        raw_cache = (
            _NoopEmbeddingCache(cache_path)
            if cache_disabled
            else _open_profile_embedding_cache(cache_path)
        )
        if not cache_disabled:
            # Persist the child handshake before exposing the service.  The
            # short cache-only admission protects direct bindings; initialize
            # repeats this under its continuous graph-to-cache admission before
            # runtime identity publication.
            with admit_operation(
                cache_path=cache_path,
                model=model_name,
                embedding_dim=dim,
                supervisor=supervisor,
                embedding_generation=getattr(supervisor, "generation", None),
                cache_exclusive=True,
                deadline=1.5,
            ):
                _persist_cache_identity(cache_path, model=model_name, embedding_dim=dim)
        cache = _GenerationBoundEmbeddingCache(
            raw_cache,
            path=cache_path,
            model=model_name,
            embedding_dim=dim,
            supervisor=supervisor,
            generation=getattr(supervisor, "generation", None),
        )
        raw_service = core.embedding_service.EmbeddingService(
            model=model_name,
            cache=cache,
            daemon=ProcessEmbeddingBackend(supervisor),
            local=NoInProcessEmbeddingBackend(dim),
        )
        service = _GenerationBoundEmbeddingService(
            raw_service,
            supervisor,
            cache_path=cache_path,
            model=model_name,
            dimension=dim,
        )
    except Exception:
        supervisor.close()
        raise
    # Publish the process-global upstream selection only after ownership,
    # dimension discovery, and cache setup all succeed. A rejected second
    # provider must not disturb the active owner's model or service.
    core.config.config.embedding_model = model_name
    if expected_dim == 0:
        core.embedding_service._KNOWN_DIMS[model_name] = dim
    core.embedding_service._default_service = service
    logger.info(
        "configured upstream embedding: model=%s dim=%d device=%s",
        model_name,
        dim,
        selected_device,
    )
    return supervisor


def _sqlite_profile_policy(
    db_path: pathlib.Path,
) -> tuple[bool, bool, str | None]:
    """Return ``(read_only, cache_disabled, reason)`` before any Cashew writes."""
    affected = sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info)
    if not affected:
        return False, False, None
    if not db_path.exists():
        return False, True, "wal_runtime_unsupported_fresh_cache"
    try:
        conn, mode = open_readonly_verified(db_path)
        try:
            if mode == "wal":
                verify_readonly_profile(conn)
                return True, True, "wal_runtime_unsupported"
        finally:
            conn.close()
    except Exception as exc:
        # A profile that cannot be inspected safely is not a readable keyword
        # fallback.  Keep it unavailable rather than serving an identity we
        # could not prove against the affected WAL runtime.
        raise SQLiteWALUnsupportedError(
            "read-only WAL profile verification failed"
        ) from exc
    return False, True, "wal_runtime_unsupported_fresh_cache"


def _bootstrap_sqlite_profile(
    db_path: pathlib.Path,
    cache_path: pathlib.Path,
    *,
    cache_disabled: bool,
    model: str | None = None,
    embedding_dim: int | None = None,
) -> None:
    """Apply journal policy under one exclusive graph then cache bootstrap."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with admit_operation(graph_path=db_path, exclusive=True, deadline=5.0):
        conn = sqlite3.connect(str(db_path))
        try:
            if sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info):
                guard_sqlite_journal(conn)
            else:
                bootstrap_sqlite_journal(conn)
        finally:
            conn.close()
    if not cache_disabled:
        with admit_operation(cache_path=cache_path, exclusive=True, deadline=5.0):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_conn = sqlite3.connect(str(cache_path))
            try:
                if sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info):
                    guard_sqlite_journal(cache_conn)
                else:
                    bootstrap_sqlite_journal(cache_conn)
                cache_conn.execute(
                    "CREATE TABLE IF NOT EXISTS hermes_cashew_cache_meta "
                    "(model TEXT PRIMARY KEY, embedding_dim INTEGER NOT NULL)"
                )
                if model and embedding_dim and embedding_dim > 0:
                    prior = cache_conn.execute(
                        "SELECT embedding_dim FROM hermes_cashew_cache_meta WHERE model=?",
                        (model,),
                    ).fetchone()
                    if prior is not None and int(prior[0]) != embedding_dim:
                        raise OperationAdmissionError(
                            "embedding cache model dimension metadata mismatch"
                        )
                    cache_conn.execute(
                        "INSERT OR IGNORE INTO hermes_cashew_cache_meta "
                        "(model, embedding_dim) VALUES (?, ?)",
                        (model, embedding_dim),
                    )
                    cache_conn.commit()
            finally:
                cache_conn.close()


def _persist_cache_identity(
    cache_path: pathlib.Path | None,
    *,
    model: str | None,
    embedding_dim: int | None,
) -> None:
    """Persist a child-discovered cache dimension before publication.

    The caller owns the continuous graph-to-cache exclusive admission.  This
    helper deliberately performs no lock acquisition of its own, so publishing
    an unknown-model handshake cannot race a cache put or another initializer.
    """
    if cache_path is None or not model or embedding_dim is None or embedding_dim <= 0:
        raise OperationAdmissionError("embedding cache identity is unavailable")
    conn = sqlite3.connect(str(cache_path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hermes_cashew_cache_meta "
            "(model TEXT PRIMARY KEY, embedding_dim INTEGER NOT NULL)"
        )
        prior = conn.execute(
            "SELECT embedding_dim FROM hermes_cashew_cache_meta WHERE model=?",
            (model,),
        ).fetchone()
        if prior is not None and int(prior[0]) != embedding_dim:
            raise OperationAdmissionError(
                "embedding cache model dimension metadata mismatch"
            )
        conn.execute(
            "INSERT OR IGNORE INTO hermes_cashew_cache_meta "
            "(model, embedding_dim) VALUES (?, ?)",
            (model, embedding_dim),
        )
        conn.commit()
    finally:
        conn.close()


# Import-time model construction is intentionally avoided. The provider binds
# the process service after resolving its profile-scoped cache path.


def _remove_existing_sleep_job(hermes_home: pathlib.Path | None) -> None:
    """Compatibility no-op for older callers.

    Job ownership is now established by a profile token in the scheduler prompt
    and by the generated script marker.  A name-only scan could delete another
    profile's job, so reconciliation deliberately performs no global cleanup.
    """
    del hermes_home


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
        self._embedding_cache_path: pathlib.Path | None = None
        self._cache_writes_disabled = False
        self._read_only_mode = False
        self._sqlite_policy_reason: str | None = None
        self._sqlite_bootstrap_failed = False
        self._sqlite_report: dict[str, str] | None = None
        self._runtime_epoch: int | None = None
        self._embedding_generation: str | int | None = None
        self._think_claim_state = "none"
        self._sync_queue: queue.Queue | None = None
        self._session_id: str = ""
        # Hermes permits user-memory writes only from the primary agent context.
        # Unknown future contexts fail closed.
        self._write_enabled: bool = True
        self._sync_worker: "threading.Thread | None" = None
        # Serializes initialization ownership decisions. The lock is held only
        # while claiming/releasing a generation; sync_turn never waits on it.
        self._lifecycle_lock = threading.Lock()
        self._initializing = False
        self._initialization_cancelled = False
        # Serializes producer admission with sentinel insertion so no turn can
        # race behind the shutdown sentinel and remain unprocessed.
        self._sync_state_lock = threading.Lock()
        # Daemon worker that drains _sync_queue. Started in initialize() only
        # on the happy path and given a bounded drain during shutdown().
        self._dropped_turn_count: int = 0
        # Monotonic counter of drop-oldest events on the sync queue.
        # Incremented inside sync_turn's overflow branch each time a queued
        # turn is evicted to make room for a new one.
        self._model_fn: Callable[[str], str] | None = None
        # Prefetch warm cache: cue → structured result with its request identity.
        # Populated by queue_prefetch(), consumed by prefetch(), cleared in
        # shutdown(). Ephemeral per-turn state — never persisted.
        self._warm_cache: dict[str, _PrefetchResult] = {}
        # Staging slot for background prefetch results. The queue_prefetch
        # background thread writes here; prefetch() atomically swaps it into
        # _warm_cache at the start of its call. This avoids concurrent access
        # between the daemon thread and the main agent loop.
        self._prefetch_pending: _PrefetchResult | None = None
        self._prefetch_generation: int = 0
        # One daemon worker owns background warmups. At most one request is
        # active and one newer request is retained in this pending slot.
        self._prefetch_threads: set[threading.Thread] = set()
        self._prefetch_worker: threading.Thread | None = None
        self._prefetch_condition = threading.Condition(self._sync_state_lock)
        self._prefetch_pending_request: _PrefetchRequest | None = None
        self._prefetch_active_identity: _PrefetchRequestIdentity | None = None
        self._prefetch_latest_identity: _PrefetchRequestIdentity | None = None
        # Last assistant response, buffered from sync_turn for use by
        # queue_prefetch's LLM cue extraction.
        self._last_assistant: str = ""
        # Cron job ID for the sleep cycle scheduler. Set in
        # _register_sleep_cron(), cleared in shutdown(). None when
        # sleep scheduling is disabled or registration failed.
        self._sleep_cron_job_id: str | None = None
        # Stop accepting new turns once shutdown begins while allowing turns
        # already ahead of the sentinel to drain normally.
        self._shutdown_started = threading.Event()
        self._shutdown_cleanup_pending: (
            tuple[threading.Thread | None, queue.Queue, tuple[threading.Thread, ...]]
            | None
        ) = None
        # Interpreter-finalization flag: set only after sentence-transformers
        # reports that Python's atexit sequence has begun. Unlike normal
        # shutdown, this is unrecoverable and remaining queued turns must stop.
        self._shutdown_flag = threading.Event()
        # Provider-local operational status.  The module-level metrics object is
        # aggregate telemetry only; this snapshot is the truthful per-generation
        # diagnostic contract.
        self._health_state = "unconfigured"
        self._health_reason: str | None = "not_initialized"
        self._health_fallback = "none"
        self._health_cron = "disabled"
        self._health_generation = 0
        self._health_last_error: dict[str, str | float] | None = None
        self._health_last_error_at: float | None = None
        self._outcomes = OutcomeLedger()
        self._vector_available: bool | None = None
        # Set only after the owned child has proved the configured embedding
        # identity and any required backup-backed repair has succeeded.  It is
        # deliberately independent of sqlite-vec availability: a ready brain
        # may still use upstream BFS when that optional extension is absent.
        self._embedding_identity_ready = False
        self._embedding_supervisor: EmbeddingSupervisor | None = None
        # Optional diagnostics are explicitly enabled and owned by this
        # provider generation. It is closed only after accepted workers drain.
        self._sentry_telemetry: SentryTelemetry | None = None
        self._log_scrub_acquired = False

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

    def health_status(self) -> Dict[str, Any]:
        """Return a bounded, read-only operational snapshot for diagnostics.

        This method intentionally performs no database/model/cron work and does
        not change provider state.  Hermes discovery must continue to use
        :meth:`is_available`, whose cheap config/dependency contract is stable.
        """
        with self._sync_state_lock:
            last_error = None
            if self._health_last_error is not None:
                last_error = dict(self._health_last_error)
                if self._health_last_error_at is not None:
                    last_error["age_s"] = round(
                        max(0.0, time.monotonic() - self._health_last_error_at), 1
                    )
            runtime = {
                "config_loaded": self._config is not None,
                "retriever_ready": self._retriever is not None,
                "write_enabled": self._write_enabled,
                "worker_running": bool(
                    self._sync_worker is not None and self._sync_worker.is_alive()
                ),
            }
            snapshot: dict[str, Any] = {
                "state": self._health_state,
                "reason_code": self._health_reason,
                "generation": self._health_generation,
                "runtime": runtime,
                "fallback": self._health_fallback,
                "cron": self._health_cron,
                "last_error": last_error,
            }
            snapshot.update(self._outcomes.snapshot())
            if self._sqlite_report is not None:
                snapshot["sqlite"] = dict(self._sqlite_report)
            if self._think_claim_state != "none":
                snapshot["think"] = {"claim_state": self._think_claim_state}
            return snapshot

    def _set_health_locked(
        self,
        state: str,
        reason: str | None = None,
        *,
        fallback: str | None = None,
        cron: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Publish health state while ``_sync_state_lock`` is held."""
        self._health_state = state
        self._health_reason = reason
        if fallback is not None:
            self._health_fallback = fallback
        if cron is not None:
            self._health_cron = cron
        if error is not None:
            self._health_last_error = {
                "code": reason or "provider_error",
                "class": safe_error_class(error),
            }
            self._health_last_error_at = time.monotonic()
        elif state in {"ready", "degraded", "stopped", "unconfigured"}:
            self._health_last_error = None
            self._health_last_error_at = None

    def _mark_health_if_current(
        self,
        ledger: OutcomeLedger,
        generation: int,
        state: str,
        reason: str,
        *,
        fallback: str | None = None,
    ) -> None:
        """Publish an operation finding only for its admitted generation."""
        with self._sync_state_lock:
            if (
                ledger is not self._outcomes
                or generation != self._health_generation
                or self._shutdown_started.is_set()
                or self._health_state in {"stopping", "stopped"}
            ):
                return
            self._set_health_locked(state, reason, fallback=fallback)

    def _outcome_current_locked(self, ledger: OutcomeLedger, generation: int) -> bool:
        """Return whether an operation may publish into the current runtime."""
        return (
            ledger is self._outcomes
            and generation == self._health_generation
            and not self._shutdown_started.is_set()
            and self._health_state not in {"stopping", "stopped"}
        )

    @contextlib.contextmanager
    def _operation_admission(self, *, exclusive: bool = False) -> Any:
        """Reuse a matching token or admit this provider operation once."""
        if self._db_path is None or self._config is None:
            raise OperationAdmissionError("provider is not initialized")
        current = current_admission()
        cache_path = None if self._cache_writes_disabled else self._embedding_cache_path
        if current is not None:
            expected_graph = pathlib.Path(self._db_path).resolve(strict=False)
            expected_cache = (
                pathlib.Path(cache_path).resolve(strict=False)
                if cache_path is not None
                else None
            )
            expected_dim = self._active_embedding_dimension()
            if (
                current.graph_path != expected_graph
                or current.cache_path != expected_cache
                or current.model != self._config.embedding_model
                or current.embedding_dim != expected_dim
                or current.vec_dim != expected_dim
                or current.supervisor is not self._embedding_supervisor
                or current.embedding_generation != self._embedding_generation
                or (
                    self._runtime_epoch is not None
                    and current.epoch != self._runtime_epoch
                )
                or (exclusive and not current.exclusive)
                or (cache_path is not None and current.cache_lease is None)
            ):
                raise OperationAdmissionError("operation admission identity mismatch")
            yield current
            return
        with admit_operation(
            graph_path=self._db_path,
            cache_path=cache_path,
            model=self._config.embedding_model,
            embedding_dim=self._active_embedding_dimension(),
            vec_dim=self._active_embedding_dimension(),
            epoch=self._runtime_epoch,
            supervisor=self._embedding_supervisor,
            embedding_generation=self._embedding_generation,
            exclusive=exclusive,
            cache_exclusive=exclusive,
            deadline=1.5,
        ) as admission:
            self._validate_runtime_identity(admission)
            yield admission

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

    def initialize(  # noqa: C901 - lifecycle publication and cleanup share one ownership boundary
        self, session_id: str, **kwargs: Any
    ) -> None:
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
        with self._lifecycle_lock:
            if self._initializing:
                logger.warning(
                    "cashew initialize deferred: another initialization is in progress"
                )
                return
            with self._sync_state_lock:
                if self._sync_queue is not None or self._shutdown_started.is_set():
                    logger.warning(
                        "cashew initialize deferred: previous worker generation still owns runtime state"
                    )
                    return
                self._initializing = True
                self._initialization_cancelled = False
                self._health_generation += 1
                self._outcomes = OutcomeLedger()
                self._health_fallback = "none"
                self._health_cron = "disabled"
                self._health_last_error = None
                self._health_last_error_at = None
                self._vector_available = None
                self._sqlite_bootstrap_failed = False
                self._sqlite_report = None
                self._embedding_generation = None
                self._embedding_identity_ready = False
                self._set_health_locked("initializing", "initializing")
        try:
            self._session_id = session_id
            self._hermes_home = pathlib.Path(kwargs["hermes_home"])
            self._write_enabled = kwargs.get("agent_context", "primary") == "primary"
            # Queue is created here so config-driven sizing/timeout values are wired in.
            # The daemon worker drains it and receives a bounded flush during
            # provider shutdown.
            self._sync_queue = queue.Queue(maxsize=16)
            # Reset lifecycle flags — a provider instance may be initialized again
            # after a prior shutdown.
            self._shutdown_started.clear()
            self._shutdown_flag.clear()
            acquire_provider_scrub_filters()
            self._log_scrub_acquired = True
            self._sentry_telemetry = start_sentry_telemetry()
            with trace_operation("cashew.initialize"):
                self._config = load_config(self._hermes_home)
                # First-load bootstrap creates only Cashew's own config. The
                # user explicitly opts into host-backed LLM work by adding an
                # auxiliary role to Hermes config.yaml.
                _ensure_config_file(self._hermes_home)
                self._db_path = resolve_db_path(
                    self._hermes_home, self._config.cashew_db_path
                )
                self._embedding_cache_path = self._db_path.parent / "embedding-cache.db"
                (
                    self._read_only_mode,
                    self._cache_writes_disabled,
                    self._sqlite_policy_reason,
                ) = _sqlite_profile_policy(self._db_path)
                if self._read_only_mode:
                    # Existing WAL on an affected runtime is never opened by
                    # Cashew's write-capable upstream APIs. Keyword recall
                    # remains available through the wrapper's read path.
                    self._write_enabled = False
                    self._embedding_identity_ready = False
                    self._vector_available = False
                    self._retriever = None
                    self._model_fn = self._build_model_fn()
                else:
                    try:
                        _bootstrap_sqlite_profile(
                            self._db_path,
                            self._embedding_cache_path,
                            cache_disabled=self._cache_writes_disabled,
                            model=self._config.embedding_model,
                            embedding_dim=_UPSTREAM_KNOWN_DIMS.get(
                                self._config.embedding_model
                            ),
                        )
                    except (MaintenanceLockAcquisitionError, OperationAdmissionError):
                        self._sqlite_bootstrap_failed = True
                        self._cache_writes_disabled = True
                        self._sqlite_policy_reason = "journal_bootstrap_unavailable"
                        logger.warning(
                            "sqlite journal bootstrap unavailable; using keyword-only recall",
                            exc_info=True,
                        )
                    self._embedding_supervisor = _bind_upstream_embedding(
                        self._config.embedding_model,
                        self._config.embedding_device,
                        cache_path=self._embedding_cache_path,
                        cache_disabled=self._cache_writes_disabled,
                    )
                    self._embedding_generation = getattr(
                        self._embedding_supervisor, "generation", None
                    )
                    if self._sqlite_bootstrap_failed:
                        logger.warning(
                            "embedding identity %s; unable to acquire %s; using keyword-only recall",
                            (
                                "could not be verified"
                                if self._embedding_migration_required(self._db_path)
                                else "inspection deferred"
                            ),
                            lock_path_for_db(self._db_path),
                        )
                # ContextRetriever.__init__ is lazy — no SQLite open, no embedding load yet.
                # Guard against the defensive-import fallback (ContextRetriever = None).
                if ContextRetriever is None:
                    raise RuntimeError(
                        "core.context.ContextRetriever unavailable at import time; "
                        "cashew-brain dependency missing"
                    )
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                # Keep the wrapper-owned sqlite-vec migration behind the same
                # backup boundary as upstream's destructive re-embedding.  In
                # particular, an old vec table must not be dropped before a
                # failed embedding repair has a chance to restore it.
                if self._read_only_mode or self._sqlite_bootstrap_failed:
                    identity_ready = False
                else:
                    # Keep one exclusive graph->cache admission from schema
                    # bootstrap through claim recovery, migration, vec
                    # finalization, and identity publication. This prevents a
                    # second process from observing a half-published profile.
                    with admit_operation(
                        graph_path=self._db_path,
                        cache_path=(
                            None
                            if self._cache_writes_disabled
                            else self._embedding_cache_path
                        ),
                        model=self._config.embedding_model,
                        embedding_dim=self._active_embedding_dimension(),
                        vec_dim=self._active_embedding_dimension(),
                        epoch=self._runtime_epoch,
                        supervisor=self._embedding_supervisor,
                        embedding_generation=self._embedding_generation,
                        exclusive=True,
                        cache_exclusive=True,
                        deadline=5.0,
                    ) as bootstrap_admission:
                        self._ensure_db_schema(self._db_path)
                        self._recover_think_claim(self._db_path)
                        identity_ready = self._repair_embedding_dimension_locked(
                            self._db_path
                        )
                        if identity_ready:
                            self._finalize_vec_schema(self._db_path)
                            self._validate_runtime_identity(
                                bootstrap_admission, allow_metadata_mismatch=True
                            )
                            if not self._cache_writes_disabled:
                                _persist_cache_identity(
                                    self._embedding_cache_path,
                                    model=self._config.embedding_model,
                                    embedding_dim=self._active_embedding_dimension(),
                                )
                            self._write_runtime_identity(self._db_path)
                try:
                    if self._read_only_mode:
                        conn, _mode = open_readonly_verified(self._db_path)
                        try:
                            self._sqlite_report = verify_readonly_profile(conn)
                            self._sqlite_report["journal_mode"] = _mode
                            self._sqlite_report["sqlite_version"] = (
                                sqlite3.sqlite_version
                            )
                        finally:
                            conn.close()
                    else:
                        self._sqlite_report = sqlite_journal_report(self._db_path)
                except Exception:
                    self._sqlite_report = None
                with self._sync_state_lock:
                    self._embedding_identity_ready = identity_ready
                    if not identity_ready:
                        self._warm_cache.clear()
                        self._prefetch_pending = None
                if not identity_ready and not self._read_only_mode:
                    # Do not expose an embedding service whose persisted
                    # identity could not be proved or restored.  Query stays
                    # available through keyword search, but all semantic
                    # mutation and embedding-backed retrieval are fail-closed
                    # until a later successful initialize.
                    self._vector_available = False
                    self._suspend_sleep_cron()
                    self._close_embedding_runtime(self._embedding_supervisor)
                if not self._read_only_mode:
                    self._retriever = ContextRetriever(db_path=str(self._db_path))
                    self._model_fn = self._build_model_fn()
                # Finish all synchronous setup before publishing the worker.
                # Shutdown can cancel a slow initialize while this work runs;
                # the final lifecycle-locked handoff below is the only place
                # where a worker may become visible.
                # Reject cancellation before cron registration when possible.
                # A shutdown racing after this check can still leave an
                # already-started external registration; the final publication
                # check below prevents that generation from starting a worker.
                with self._lifecycle_lock:
                    if self._initialization_cancelled:
                        raise _InitializationCancelledError()
                if (
                    self._write_enabled
                    and self._embedding_identity_ready
                    and _HAS_HERMES_CRON
                ):
                    self._register_sleep_cron()
                with self._lifecycle_lock:
                    if self._initialization_cancelled:
                        raise _InitializationCancelledError()
                    # Start only after the complete runtime snapshot is ready.
                    self._start_sync_worker()
                    # Publish health while the same lifecycle ownership lock is
                    # held as worker publication.  Shutdown cannot clear the
                    # config or queue between the start and this snapshot.
                    with self._sync_state_lock:
                        if self._sync_queue is None or self._shutdown_started.is_set():
                            raise _InitializationCancelledError()
                        cron_state = (
                            "registered"
                            if self._sleep_cron_job_id is not None
                            else self._health_cron
                        )
                        config = self._config
                        model_fn = self._model_fn
                        if not self._embedding_identity_ready:
                            self._set_health_locked(
                                "degraded",
                                "identity_unresolved",
                                fallback="keyword",
                                cron=cron_state,
                            )
                        elif self._vector_available is False:
                            self._set_health_locked(
                                "degraded",
                                "vector_unavailable",
                                fallback="keyword",
                                cron=cron_state,
                            )
                        elif cron_state in {"unavailable", "failed"}:
                            self._set_health_locked(
                                "degraded",
                                "cron_unavailable"
                                if cron_state == "unavailable"
                                else "cron_registration_failed",
                                cron=cron_state,
                            )
                        elif (
                            config is not None
                            and config.llm_aux_role
                            and model_fn is None
                        ):
                            self._set_health_locked(
                                "degraded", "model_unavailable", cron=cron_state
                            )
                        else:
                            self._set_health_locked("ready", None, cron=cron_state)
        except _InitializationCancelledError:
            model_fn = self._model_fn
            supervisor = self._embedding_supervisor
            telemetry = self._sentry_telemetry
            with self._sync_state_lock:
                self._config = None
                self._db_path = None
                self._retriever = None
                self._model_fn = None
                self._embedding_identity_ready = False
                self._sync_queue = None
                self._sentry_telemetry = None
                self._shutdown_started.set()
                self._set_health_locked("stopped", "initialization_cancelled")
            close_model = getattr(model_fn, "_cashew_close", None)
            if callable(close_model):
                close_model()
            close_sentry_telemetry(telemetry)
            self._close_embedding_runtime(supervisor)
            if self._log_scrub_acquired:
                release_provider_scrub_filters()
                self._log_scrub_acquired = False
        except Exception as _exc:
            capture_exception(
                _exc,
                operation="cashew.initialize",
                session_id=session_id,
                telemetry=self._sentry_telemetry,
            )
            config_path = (
                resolve_config_path(self._hermes_home)
                if self._hermes_home is not None
                else "<unknown>"
            )
            logger.warning(
                "cashew initialize failed at %s; provider will report unavailable until fixed",
                config_path,
                exc_info=True,
            )
            model_fn = self._model_fn
            supervisor = self._embedding_supervisor
            telemetry = self._sentry_telemetry
            reason = self._initialization_reason(_exc)
            with self._sync_state_lock:
                self._config = None
                self._db_path = None
                self._retriever = None
                self._model_fn = None
                self._embedding_identity_ready = False
                self._sync_worker = None
                self._sync_queue = None
                self._sentry_telemetry = None
                self._shutdown_started.set()
                self._set_health_locked("failed", reason, error=_exc)
            close_model = getattr(model_fn, "_cashew_close", None)
            if callable(close_model):
                close_model()
            close_sentry_telemetry(telemetry)
            self._close_embedding_runtime(supervisor)
            if self._log_scrub_acquired:
                release_provider_scrub_filters()
                self._log_scrub_acquired = False
        finally:
            with self._lifecycle_lock:
                self._initializing = False
                self._initialization_cancelled = False

    @staticmethod
    def _initialization_reason(error: BaseException) -> str:
        """Map init failures to a small diagnostic vocabulary."""
        if ContextRetriever is None:
            return "dependency_missing"
        if isinstance(error, (ValueError, TypeError, json.JSONDecodeError)):
            return "config_invalid"
        if isinstance(error, sqlite3.OperationalError):
            return "storage_error"
        return "initialization_failed"

    def _start_sync_worker(self) -> None:
        """Launch the daemon worker. Called from initialize() only on happy path.

        MUST run AFTER self._db_path / self._session_id / self._sync_queue are set.
        The shutdown sentinel and bounded join provide the normal drain path;
        daemon=True prevents a wedged dependency from blocking interpreter exit.
        """
        self._sync_worker = threading.Thread(
            target=self._worker_loop,
            name=f"cashew-sync-{self._session_id}",
            daemon=True,
        )
        self._sync_worker.start()

    # ── Sleep cycle cron scheduling ──────────────────────────────────────

    def _register_sleep_cron(self) -> None:
        """Reconcile this profile's persistent cron job and managed script."""
        if (
            self._hermes_home is None
            or self._config is None
            or not self._embedding_identity_ready
        ):
            return
        try:
            from cron.jobs import (
                create_job,
                list_jobs,
                parse_schedule,
                remove_job,
                update_job,
                use_cron_store,
            )

            home = self._hermes_home
            config = self._config
            desired_schedule = config.sleep_schedule
            enabled = config.sleep_cycles and bool(desired_schedule)
            parsed_schedule = parse_schedule(desired_schedule) if enabled else None
            profile_id = profile_identity(home)
            script_dest = home / "scripts" / CRON_SCRIPT_NAME
            # The public cron API is profile-contextual.  Keep the whole
            # read/reconcile/write transaction in this profile's store rather
            # than relying on the process-wide active Hermes home.
            with profile_cron_lock(home), use_cron_store(home):
                existing = [job for job in list_jobs() if isinstance(job, dict)]
                owned = [job for job in existing if owns_job(job, profile_id)]
                if not enabled:
                    for job in owned:
                        job_id = job.get("id")
                        if isinstance(job_id, str):
                            remove_job(job_id)
                    self._sleep_cron_job_id = None
                    return

                marker = installation_marker(
                    home, pathlib.Path(__file__).parent.resolve(), config
                )
                template = (
                    pathlib.Path(__file__).parent / "sleep_cron_script.py"
                ).read_text(encoding="utf-8")
                rendered = render_script(template, marker)
                if stage_script(script_dest, rendered):
                    logger.info("sleep: refreshed cron script at %s", script_dest)

                if len(owned) == 1:
                    job_id = owned[0].get("id")
                    if isinstance(job_id, str):
                        # Preserve the scheduler-owned identity and next-run
                        # state when only the desired cadence changed.
                        if owned[0].get("schedule") != parsed_schedule:
                            update_job(job_id, {"schedule": desired_schedule})
                        self._sleep_cron_job_id = job_id
                        with self._sync_state_lock:
                            self._health_cron = "registered"
                        return

                for job in owned:
                    job_id = job.get("id")
                    if isinstance(job_id, str):
                        remove_job(job_id)
                job = create_job(
                    prompt=f"hermes-cashew sleep cycle [{profile_id}]",
                    schedule=desired_schedule,
                    name=CRON_JOB_NAME,
                    script=CRON_SCRIPT_NAME,
                    no_agent=True,
                    repeat=None,  # forever
                )
            self._sleep_cron_job_id = job["id"]
            with self._sync_state_lock:
                self._health_cron = "registered"
            logger.info(
                "sleep: registered cron job %s (schedule=%s)",
                job["id"],
                desired_schedule,
            )
        except ImportError:
            with self._sync_state_lock:
                self._health_cron = "unavailable"
            logger.warning(
                "sleep: cannot register cron job — Hermes cron module not available "
                "(schedule=%s); sleep cycles will not run automatically",
                self._config.sleep_schedule,
            )
        except Exception:
            with self._sync_state_lock:
                self._health_cron = "failed"
            logger.warning(
                "sleep: failed to register cron job (schedule=%s)",
                self._config.sleep_schedule,
                exc_info=True,
            )
            self._sleep_cron_job_id = None

    def _suspend_sleep_cron(self) -> None:
        """Remove a stale sleep job when this profile's identity is unresolved."""
        if self._hermes_home is None:
            return
        try:
            from cron.jobs import list_jobs, remove_job, use_cron_store

            home = self._hermes_home
            profile_id = profile_identity(home)
            with profile_cron_lock(home), use_cron_store(home):
                for job in list_jobs():
                    if isinstance(job, dict) and owns_job(job, profile_id):
                        job_id = job.get("id")
                        if isinstance(job_id, str):
                            remove_job(job_id)
        except ImportError:
            pass
        except Exception:
            logger.warning("sleep: failed to suspend unresolved identity cron")
        finally:
            self._sleep_cron_job_id = None
            with self._sync_state_lock:
                self._health_cron = "disabled"

    # LLM integration via auxiliary.memory convention
    # ------------------------------------------------------------------

    def _build_model_fn(self) -> Callable[[str], str] | None:
        """Construct an LLM callable from the configured auxiliary.memory role.

        Delegates to ``config.resolve_model_fn()``, which verifies the active
        Hermes profile's explicit auxiliary role and resolves its public client.
        Returns None when:
        - No llm_aux_role is configured (heuristic-only mode)
        - The auxiliary role is absent, null, or malformed
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
        with self._sync_state_lock:
            config = self._config
            if (
                not self._write_enabled
                or not self._embedding_identity_ready
                or config is None
                or self._initializing
                or self._shutdown_started.is_set()
                or not config.auto_extraction
                or self._sync_queue is None
            ):
                if self._sync_queue is not None and (
                    self._initializing or self._shutdown_started.is_set()
                ):
                    self._outcomes.reject()
                return
            # Buffer assistant content for queue_prefetch cue extraction only
            # after the turn has been admitted.
            if assistant_content:
                self._last_assistant = assistant_content
            q = self._sync_queue
            effective_session = str(session_id or self._session_id)
            turn = (user_content, assistant_content, effective_session)
            try:
                q.put_nowait(turn)
                self._outcomes.admit()
            except queue.Full:
                # Drop-oldest policy.
                evicted = False
                try:
                    q.get_nowait()
                    q.task_done()  # balance the drop (exactly once)
                    evicted = True
                except queue.Empty:
                    pass  # worker drained between Full and get_nowait — rare race
                if evicted:
                    self._dropped_turn_count += 1
                    self._outcomes.drop_pending()
                    _METRICS.record_sync_dropped()
                logger.warning(
                    "cashew sync queue overflow (maxsize=%d); dropped oldest turn",
                    q.maxsize,
                )
                try:
                    q.put_nowait(turn)
                    self._outcomes.admit()
                except queue.Full:
                    self._outcomes.reject()
                    logger.warning(
                        "cashew sync queue still full after drop-oldest; "
                        "dropping new turn"
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
            try:
                item = q.get(timeout=0.05)
            except queue.Empty:
                if self._shutdown_started.is_set():
                    return
                continue
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
                        items.append(extra)
                        break
                    items.append(extra)
            for turn in items:
                if turn is _SHUTDOWN:
                    q.task_done()
                    return
                with self._sync_state_lock:
                    ledger = self._outcomes
                    generation = self._health_generation
                    ledger.start()
                try:
                    with trace_operation("cashew.sync") as span:
                        span.set_attribute("input_length", len(turn[0]))
                        completed = self._drain_once(turn)
                    self._finish_worker_turn(ledger, generation, completed)
                except _PreAdmissionRejectedError:
                    with self._sync_state_lock:
                        if (
                            ledger is self._outcomes
                            and generation == self._health_generation
                        ):
                            ledger.reject_in_flight()
                    logger.info("cashew sync: turn rejected before provider admission")
                except Exception as _exc:
                    self._fail_worker_turn(ledger, generation, _exc)
                    _METRICS.record_sync_failure()
                    capture_exception(
                        _exc,
                        operation="cashew.sync",
                        session_id=self._session_id,
                        extra={"turn_user_len": len(turn[0])},
                        telemetry=self._sentry_telemetry,
                    )
                    logger.warning("cashew sync worker: turn failed", exc_info=True)
                finally:
                    q.task_done()
                    _METRICS.set_queue_depth(q.qsize())

    def _finish_worker_turn(
        self, ledger: OutcomeLedger, generation: int, completed: bool
    ) -> None:
        """Publish a worker completion only to its admitted generation."""
        with self._sync_state_lock:
            if ledger is not self._outcomes or generation != self._health_generation:
                return
            if completed is False:
                ledger.drop_in_flight()
            else:
                ledger.complete()

    def _fail_worker_turn(
        self, ledger: OutcomeLedger, generation: int, error: BaseException
    ) -> None:
        """Publish a worker failure unless shutdown has taken ownership."""
        with self._sync_state_lock:
            if ledger is not self._outcomes or generation != self._health_generation:
                return
            # core.session owns a multi-statement persistence operation.  A
            # failure after admission is never replayed: a visible node prefix
            # is partial, while an unchanged/unknown prefix is uncertain.
            ledger.fail(
                partial=isinstance(error, _OpaqueUpstreamError) and error.partial,
                uncertain=isinstance(error, _OpaqueUpstreamError) and not error.partial,
            )
            if not self._shutdown_started.is_set() and self._health_state not in {
                "stopping",
                "stopped",
            }:
                self._set_health_locked("degraded", "backend_error", error=error)

    def _ensure_db_schema(self, db_path: pathlib.Path) -> None:
        """Create or migrate Cashew schema tables.

        Delegates to cashew-brain's core.db.ensure_schema() which handles
        upstream table creation (thought_nodes, derivation_edges, embeddings,
        hotspots, metrics), column migrations, index creation, and schema
        version stamping (PRAGMA user_version = 3). Then applies hermes-specific
        extensions (provider metadata).  The sqlite-vec schema is finalized
        only after the backup-backed embedding repair has succeeded.
        """
        from core.db import ensure_schema

        with admit_operation(
            graph_path=db_path,
            cache_path=None
            if self._cache_writes_disabled
            else self._embedding_cache_path,
            model=self._config.embedding_model if self._config else None,
            embedding_dim=self._active_embedding_dimension(),
            vec_dim=self._active_embedding_dimension(),
            epoch=self._runtime_epoch,
            supervisor=self._embedding_supervisor,
            embedding_generation=self._embedding_generation,
            exclusive=True,
            cache_exclusive=True,
            deadline=5.0,
        ):
            ensure_schema(str(db_path))

        with self._operation_admission(exclusive=True):
            conn = sqlite3.connect(str(db_path))
            try:
                # Hermes provider metadata store (persistent counters, flags).
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hermes_provider_meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _runtime_metadata(db_path: pathlib.Path) -> dict[str, str]:
        """Read provider identity, tolerating pre-extension databases."""
        conn = sqlite3.connect(str(db_path))
        try:
            try:
                return dict(
                    conn.execute(
                        "SELECT key, value FROM hermes_provider_meta"
                    ).fetchall()
                )
            except sqlite3.OperationalError:
                return {}
        finally:
            conn.close()

    def _connect_graph(self, db_path: pathlib.Path | str) -> sqlite3.Connection:
        """Open the graph read-only when affected WAL policy requires it."""
        path = pathlib.Path(db_path).resolve(strict=False)
        if (
            self._read_only_mode
            and self._db_path is not None
            and path == pathlib.Path(self._db_path).resolve(strict=False)
        ):
            conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
            conn.execute("PRAGMA query_only=ON")
            if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
                conn.close()
                raise SQLiteWALUnsupportedError("query-only verification failed")
            return conn
        return sqlite3.connect(str(path))

    def _write_runtime_identity(self, db_path: pathlib.Path) -> None:
        """Publish identity atomically, preserving an unchanged maintenance epoch."""
        if self._config is None:
            return
        dim = self._active_embedding_dimension()
        if dim is None:
            raise OperationAdmissionError("embedding dimension is unavailable")
        vec_dim = self._embedding_dimensions(db_path)[1]
        values = {
            "embedding_model": self._config.embedding_model,
            "embedding_dim": str(dim),
            "vec_dim": str(vec_dim if vec_dim is not None else dim),
        }
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            previous = dict(
                conn.execute(
                    "SELECT key, value FROM hermes_provider_meta WHERE key IN "
                    "('embedding_model','embedding_dim','vec_dim','maintenance_epoch')"
                ).fetchall()
            )
            try:
                old_epoch = int(previous.get("maintenance_epoch", "0"))
            except ValueError:
                old_epoch = 0
            # A partial or malformed previous record is not a stable identity.
            # Only an exact complete match may retain the maintenance epoch.
            identity_changed = any(
                previous.get(key) != value for key, value in values.items()
            )
            # The first publication and a maintenance identity transition fence
            # prior owners. A same-identity concurrent provider keeps its epoch.
            epoch = old_epoch + 1 if not previous or identity_changed else old_epoch
            values["maintenance_epoch"] = str(epoch)
            conn.executemany(
                "INSERT OR REPLACE INTO hermes_provider_meta (key, value) VALUES (?, ?)",
                values.items(),
            )
            conn.commit()
            self._runtime_epoch = epoch
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _recover_think_claim(self, db_path: pathlib.Path) -> None:
        """Mark a previous process's unfinished claim uncertain on startup."""
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM hermes_provider_meta WHERE key='think_claim_state'"
            ).fetchone()
            if row is not None and row[0] == "in_flight":
                conn.execute(
                    "UPDATE hermes_provider_meta SET value='uncertain' "
                    "WHERE key='think_claim_state'"
                )
            conn.commit()
        except sqlite3.OperationalError:
            conn.rollback()
        finally:
            conn.close()

    def _validate_runtime_identity(
        self, admission: Any | None = None, *, allow_metadata_mismatch: bool = False
    ) -> None:
        """Reject stale graph/cache work immediately before opaque calls."""
        token = admission or current_admission()
        if token is None or self._db_path is None or self._config is None:
            raise OperationAdmissionError("operation has no active identity")
        if token.graph_path != pathlib.Path(self._db_path).resolve(strict=False):
            raise OperationAdmissionError("graph admission path mismatch")
        if token.model != self._config.embedding_model:
            raise OperationAdmissionError("graph admission model mismatch")
        expected_dim = token.embedding_dim
        if self._embedding_supervisor is not None:
            expected_dim = self._active_embedding_dimension()
            if expected_dim is None or token.embedding_dim not in (None, expected_dim):
                raise OperationAdmissionError("graph admission dimension mismatch")
            if token.supervisor is not self._embedding_supervisor:
                raise OperationAdmissionError(
                    "embedding supervisor generation mismatch"
                )
        if self._runtime_epoch is not None and token.epoch not in (
            None,
            self._runtime_epoch,
        ):
            raise OperationAdmissionError("maintenance epoch mismatch")
        metadata = self._runtime_metadata(self._db_path)
        if not allow_metadata_mismatch and (
            token.epoch is not None or self._runtime_epoch is not None
        ):
            try:
                persisted_epoch = int(metadata.get("maintenance_epoch", ""))
            except (TypeError, ValueError):
                raise OperationAdmissionError(
                    "persisted maintenance epoch is invalid"
                ) from None
            if token.epoch != persisted_epoch or self._runtime_epoch != persisted_epoch:
                raise OperationAdmissionError("persisted maintenance epoch is stale")
        if not allow_metadata_mismatch and metadata.get("embedding_model") not in (
            None,
            self._config.embedding_model,
        ):
            raise OperationAdmissionError("persisted embedding model is stale")
        if not allow_metadata_mismatch and metadata.get("embedding_dim") not in (
            None,
            str(expected_dim),
        ):
            raise OperationAdmissionError("persisted embedding dimension is stale")
        if not allow_metadata_mismatch and metadata.get("vec_dim") not in (
            None,
            str(expected_dim),
        ):
            raise OperationAdmissionError("persisted vector dimension is stale")
        if token.vec_dim not in (None, expected_dim):
            raise OperationAdmissionError("admitted vector dimension mismatch")

    def _finalize_vec_schema(self, db_path: pathlib.Path) -> None:
        """Apply the reversible-wrapper vec schema work after repair succeeds."""
        conn = sqlite3.connect(str(db_path))
        try:
            self._vector_available = True
            try:
                conn.enable_load_extension(True)
                try:
                    import sqlite_vec

                    sqlite_vec.load(conn)
                except (ImportError, AttributeError):
                    conn.load_extension("vec0")
            except Exception:
                self._vector_available = False
                pass  # sqlite-vec not available at platform level; graceful degradation active
            self._migrate_vec_embeddings(conn)
            self._create_vec_embeddings(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _embedding_dimensions(db_path: pathlib.Path) -> tuple[set[int], int | None]:
        """Return stored vector dimensions and the sqlite-vec table dimension."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            rows = conn.execute(
                "SELECT DISTINCT LENGTH(vector) / 4 FROM embeddings "
                "WHERE vector IS NOT NULL"
            ).fetchall()
            stored_dims = {int(row[0]) for row in rows if row[0] is not None}
            vec_row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='vec_embeddings'"
            ).fetchone()
        finally:
            conn.close()
        vec_match = (
            re.search(r"float\[(\d+)\]", vec_row[0], re.IGNORECASE)
            if vec_row and vec_row[0]
            else None
        )
        return stored_dims, int(vec_match.group(1)) if vec_match else None

    @staticmethod
    def _restore_embedding_backup(
        db_path: pathlib.Path, backup_path: pathlib.Path
    ) -> None:
        """Restore a pre-migration SQLite backup without copying live WAL files."""
        source = sqlite3.connect(str(backup_path))
        target = sqlite3.connect(str(db_path))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()

    def _repair_embedding_dimension(self, db_path: pathlib.Path) -> bool:
        """Back up and re-embed a brain whose stored identity is inconsistent.

        Migration runs during initialization, before the provider starts its
        worker or exposes a retriever. cashew-brain owns the destructive
        re-embedding operation; this adapter adds detection, a mandatory
        profile-scoped backup, postcondition validation, and rollback. It uses
        the same advisory lock as the synchronous sleep cycle so participating
        maintenance operations do not overlap. Broader writer coordination is
        deliberately outside this helper's scope.
        """
        lock_path = lock_path_for_db(db_path)
        try:
            with admit_operation(
                graph_path=db_path,
                cache_path=(
                    None if self._cache_writes_disabled else self._embedding_cache_path
                ),
                model=self._config.embedding_model if self._config else None,
                embedding_dim=self._active_embedding_dimension(),
                epoch=self._runtime_epoch,
                supervisor=self._embedding_supervisor,
                embedding_generation=self._embedding_generation,
                exclusive=True,
                cache_exclusive=True,
                deadline=0.1,
            ):
                return self._repair_embedding_dimension_locked(db_path)
        except (MaintenanceLockAcquisitionError, OperationAdmissionError) as exc:
            if isinstance(exc, OperationAdmissionError):
                logger.warning(
                    "embedding migration deferred; another Cashew process holds %s",
                    lock_path,
                )
            required = self._embedding_migration_required(db_path)
            logger.warning(
                "embedding identity %s; unable to acquire %s; using keyword-only recall",
                "could not be verified" if required else "inspection deferred",
                lock_path,
                exc_info=True,
            )
            return False

    def _active_embedding_dimension(self) -> int | None:
        """Return only a dimension verified by this provider's child worker."""
        supervisor = self._embedding_supervisor
        if supervisor is None or supervisor.dimension <= 0:
            return None
        return supervisor.dimension

    @staticmethod
    def _active_embedding_ids(db_path: pathlib.Path) -> set[str]:
        """Return the exact upstream rows a migration is expected to re-embed."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT id FROM thought_nodes "
                "WHERE (decayed IS NULL OR decayed = 0) "
                "AND content IS NOT NULL AND TRIM(content) != ''"
            ).fetchall()
            return {str(row[0]) for row in rows}
        finally:
            conn.close()

    @staticmethod
    def _active_embedding_models(db_path: pathlib.Path) -> set[str]:
        """Return model identities for the live rows a repair must replace."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT DISTINCT e.model FROM embeddings e "
                "JOIN thought_nodes n ON n.id = e.node_id "
                "WHERE (n.decayed IS NULL OR n.decayed = 0) "
                "AND n.content IS NOT NULL AND TRIM(n.content) != ''"
            ).fetchall()
            return {str(row[0]) for row in rows if row[0] is not None}
        finally:
            conn.close()

    @staticmethod
    def _migration_postconditions(
        db_path: pathlib.Path,
        *,
        expected_ids: set[str],
        expected_model: str,
        expected_dimension: int,
        vec_dimension: int | None,
    ) -> None:
        """Validate every active embedding before retaining destructive work."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT e.node_id, e.model, LENGTH(e.vector) / 4 "
                "FROM embeddings e JOIN thought_nodes n ON n.id = e.node_id "
                "WHERE (n.decayed IS NULL OR n.decayed = 0) "
                "AND n.content IS NOT NULL AND TRIM(n.content) != ''"
            ).fetchall()
            actual_ids = {str(row[0]) for row in rows}
            if actual_ids != expected_ids:
                raise RuntimeError("migration postcondition active node IDs mismatch")
            if any(
                row[1] != expected_model or int(row[2]) != expected_dimension
                for row in rows
            ):
                raise RuntimeError(
                    "migration postcondition model or dimension mismatch"
                )
            if vec_dimension is not None:
                extension_enabled = False
                try:
                    conn.enable_load_extension(True)
                    extension_enabled = True
                    import sqlite_vec

                    sqlite_vec.load(conn)
                except Exception:
                    # sqlite-vec is optional at this boundary. A platform can
                    # retain its schema while losing the native extension;
                    # exact ordinary embedding identity still authorizes the
                    # ready BFS path, while finalization reports vec degraded.
                    logger.info(
                        "sqlite-vec unavailable during migration postcondition validation; "
                        "skipping vec row check"
                    )
                else:
                    vec_ids = {
                        str(row[0])
                        for row in conn.execute(
                            "SELECT node_id FROM vec_embeddings"
                        ).fetchall()
                    }
                    if vec_ids != expected_ids:
                        raise RuntimeError(
                            "migration postcondition vec node IDs mismatch"
                        )
                finally:
                    if extension_enabled:
                        conn.enable_load_extension(False)
        finally:
            conn.close()

    def _embedding_migration_required(self, db_path: pathlib.Path) -> bool:
        """Fail closed unless a read-only inspection proves migration is unnecessary."""
        if self._config is None:
            return True
        try:
            expected_dim = self._active_embedding_dimension()
            if expected_dim is None:
                return True
            stored_dims, vec_dim = self._embedding_dimensions(db_path)
            stored_models = self._active_embedding_models(db_path)
            metadata = self._runtime_metadata(db_path)
        except Exception:
            logger.warning(
                "could not inspect embedding dimensions after lock acquisition failure",
                exc_info=True,
            )
            return True

        stored_mismatch = bool(stored_dims) and stored_dims != {expected_dim}
        vec_mismatch = vec_dim is not None and vec_dim != expected_dim
        model_mismatch = bool(stored_models) and stored_models != {
            self._config.embedding_model
        }
        persisted_model_mismatch = metadata.get("embedding_model") not in (
            None,
            self._config.embedding_model,
        )
        persisted_dim_mismatch = metadata.get("embedding_dim") not in (
            None,
            str(expected_dim),
        )
        return (
            stored_mismatch
            or vec_mismatch
            or model_mismatch
            or persisted_model_mismatch
            or persisted_dim_mismatch
        )

    def _repair_embedding_dimension_locked(self, db_path: pathlib.Path) -> bool:
        """Repair embedding identity while the cross-process Cashew lock is held."""
        if self._config is None:
            return False
        try:
            expected_dim = self._active_embedding_dimension()
            if expected_dim is None:
                logger.warning(
                    "embedding dimension unavailable from owned child; migration skipped"
                )
                return False
            stored_dims, vec_dim = self._embedding_dimensions(db_path)
            stored_models = self._active_embedding_models(db_path)
            metadata = self._runtime_metadata(db_path)
        except Exception:
            logger.warning(
                "could not inspect embedding dimensions; migration skipped",
                exc_info=True,
            )
            return False

        stored_mismatch = bool(stored_dims) and stored_dims != {expected_dim}
        vec_mismatch = vec_dim is not None and vec_dim != expected_dim
        model_mismatch = bool(stored_models) and stored_models != {
            self._config.embedding_model
        }
        persisted_model_mismatch = metadata.get("embedding_model") not in (
            None,
            self._config.embedding_model,
        )
        persisted_dim_mismatch = metadata.get("embedding_dim") not in (
            None,
            str(expected_dim),
        )
        if not (
            stored_mismatch
            or vec_mismatch
            or model_mismatch
            or persisted_model_mismatch
            or persisted_dim_mismatch
        ):
            return True

        from core.backup import create_backup

        backup_dir = db_path.parent / "backups"
        backup = create_backup(str(db_path), str(backup_dir))
        if backup is None:
            logger.warning(
                "embedding identity mismatch detected (stored=%s models=%s vec=%s expected=%s/%s), "
                "but backup failed; migration skipped",
                sorted(stored_dims),
                sorted(stored_models),
                vec_dim,
                self._config.embedding_model,
                expected_dim,
            )
            return False

        backup_path = pathlib.Path(backup)
        expected_ids = self._active_embedding_ids(db_path)
        expected_count = len(expected_ids)
        try:
            from scripts.migrate_embeddings import migrate_embeddings

            summary = migrate_embeddings(str(db_path), confirm=True, quiet=True)
            embedded_count = summary.get("nodes_embedded")
            if (
                not isinstance(embedded_count, int)
                or isinstance(embedded_count, bool)
                or embedded_count != expected_count
            ):
                raise RuntimeError(
                    "migration embedded "
                    f"{embedded_count!r} nodes; expected {expected_count}"
                )
            stored_after, vec_after = self._embedding_dimensions(db_path)
            if stored_after and stored_after != {expected_dim}:
                raise RuntimeError(
                    f"stored embeddings remain at dimensions {sorted(stored_after)}"
                )
            if vec_after is not None and vec_after != expected_dim:
                raise RuntimeError(f"vec_embeddings remains at dimension {vec_after}")
            self._migration_postconditions(
                db_path,
                expected_ids=expected_ids,
                expected_model=self._config.embedding_model,
                expected_dimension=expected_dim,
                vec_dimension=vec_after,
            )
        except Exception:
            logger.warning(
                "embedding migration failed; restoring pre-migration backup %s",
                backup_path,
                exc_info=True,
            )
            try:
                self._restore_embedding_backup(db_path, backup_path)
            except Exception:
                logger.warning(
                    "embedding migration rollback failed for %s",
                    db_path,
                    exc_info=True,
                )
            return False

        logger.info(
            "embedding migration complete: stored=%s models=%s vec=%s expected=%s/%s "
            "nodes_embedded=%s backup=%s",
            sorted(stored_dims),
            sorted(stored_models),
            vec_dim,
            self._config.embedding_model,
            expected_dim,
            summary.get("nodes_embedded", 0),
            backup_path,
        )
        return True

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
            # sqlite-vec schema is semantic state.  It may only be created with
            # the active child-worker dimension; never load a parent LocalBackend
            # or guess a fallback dimension here.
            dim = self._active_embedding_dimension()
            if dim is None:
                logger.warning(
                    "embedding dimension unavailable from owned child; vec schema deferred"
                )
                return
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
                USING vec0(node_id TEXT primary key, embedding float[{dim}] distance_metric=cosine)
            """)
            logger.debug(f"vec_embeddings virtual table ready (dim={dim})")
        except Exception:
            logger.info(
                "sqlite-vec extension failed to load; semantic search will use fallback"
            )

    def _enrich_results(
        self,
        node_ids: list[str],
        *,
        db_path: str | pathlib.Path | None = None,
    ) -> list[dict]:
        """Fetch full node dicts from DB for upstream retrieval results.

        Upstream RetrievalResult carries core fields (id, content, type, domain),
        but Hermes-specific formatting needs permanent flag, tags, referent_time
        etc. Batch-query the DB to get these.
        """
        if not node_ids:
            return []
        target_db = db_path if db_path is not None else self._db_path
        if target_db is None:
            return []
        conn = self._connect_graph(target_db)
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

    def _update_access_metrics(
        self, node_ids: list[str], db_path: pathlib.Path | str | None = None
    ) -> None:
        if (
            not self._write_enabled
            or not self._embedding_identity_ready
            or not node_ids
        ):
            return
        target_db = db_path if db_path is not None else self._db_path
        if target_db is None:
            return
        try:
            with self._operation_admission():
                conn = self._connect_graph(target_db)
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
        except OperationAdmissionError:
            logger.info(
                "cashew access metrics deferred: operation admission unavailable"
            )
        except Exception:
            logger.warning("cashew access metrics update failed", exc_info=True)

    def _drain_once(self, turn: tuple[str, str, str]) -> bool:
        """Persist one turn via Cashew's heuristic extractor (or LLM if configured).

        Lazy-imports core.session so the plugin module loads even when cashew-brain
        is not installed. When self._model_fn is set (via llm_aux_role config),
        upstream receives the LLM callable and can perform LLM extraction, think
        cycles, and sleep synthesis. When None, Cashew's built-in heuristic
        extractor is used (no LLM round-trip).

        Lease contention is handled before admission. Once upstream begins, an
        exception is terminal for this turn; replaying it could duplicate the
        stages that Cashew already committed."""
        from core.session import end_session  # lazy import

        user, assistant, session_id = turn

        # Short-circuit only after Python interpreter finalization has been
        # observed. Normal provider shutdown must drain accepted turns.
        if self._shutdown_flag.is_set() or not self._embedding_identity_ready:
            logger.debug("cashew sync: interpreter shutdown flag set, dropping turn")
            return False

        try:
            with self._operation_admission() as admission:
                self._validate_runtime_identity(admission)
                before_nodes: int | None
                try:
                    conn = sqlite3.connect(str(self._db_path))
                    try:
                        before_nodes = int(
                            conn.execute(
                                "SELECT COUNT(*) FROM thought_nodes"
                            ).fetchone()[0]
                        )
                    finally:
                        conn.close()
                except Exception:
                    before_nodes = None
                try:
                    extraction_result = end_session(
                        db_path=str(self._db_path),
                        session_id=session_id or self._session_id,
                        conversation_text=f"User: {user}\nAssistant: {assistant}",
                        model_fn=self._model_fn,
                    )
                    # The pinned upstream contract returns an ExtractionResult.
                    # A swallowed internal failure or wrapper that returns no
                    # result gives us no evidence about committed progress and
                    # must remain uncertain rather than being replayed.
                    if extraction_result is None or not all(
                        hasattr(extraction_result, field)
                        for field in ("new_nodes", "new_edges", "updated_nodes")
                    ):
                        raise _OpaqueUpstreamError(partial=False)
                except Exception as exc:
                    # Interpreter finalization is a terminal local condition,
                    # not an opaque upstream persistence outcome.  Preserve
                    # the outer drop path that stops the worker cleanly.
                    if isinstance(
                        exc, RuntimeError
                    ) and "can't register atexit after shutdown" in str(exc):
                        raise
                    try:
                        conn = sqlite3.connect(str(self._db_path))
                        try:
                            after_nodes = int(
                                conn.execute(
                                    "SELECT COUNT(*) FROM thought_nodes"
                                ).fetchone()[0]
                            )
                        finally:
                            conn.close()
                    except Exception:
                        after_nodes = None
                    raise _OpaqueUpstreamError(
                        partial=(
                            before_nodes is not None
                            and after_nodes is not None
                            and after_nodes > before_nodes
                        )
                    ) from exc
            _METRICS.record_sync_success()
        except RuntimeError as e:
            # Python interpreter shutdown: sentence-transformers' thread pool
            # was finalized by atexit handlers. There is no recovery — drop
            # the turn silently and let the worker loop terminate naturally.
            msg = str(e)
            if "can't register atexit after shutdown" in msg:
                logger.info("cashew sync: interpreter shutting down, dropping turn")
                self._shutdown_flag.set()
                return False
            raise
        except OperationAdmissionError:
            logger.info("cashew sync: operation rejected before upstream admission")
            raise _PreAdmissionRejectedError() from None
        self._run_think_cycle_if_due()
        return True

    def _run_think_cycle_if_due(self) -> None:
        """Atomically claim one think cycle and retain its identity lease."""
        if (
            self._model_fn is None
            or not self._embedding_identity_ready
            or self._config is None
            or not self._config.think_cycles
            or self._config.think_interval <= 0
            or self._db_path is None
        ):
            return
        from core.session import think_cycle

        claim_epoch: str | None = None
        try:
            with self._operation_admission(exclusive=True) as admission:
                self._validate_runtime_identity(admission)
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS hermes_provider_meta "
                        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    values = dict(
                        conn.execute(
                            "SELECT key, value FROM hermes_provider_meta WHERE key IN "
                            "('think_counter','think_claim_epoch','think_claim_state')"
                        ).fetchall()
                    )
                    state = values.get("think_claim_state", "none")
                    if state in {"in_flight", "failed", "partial", "uncertain"}:
                        conn.rollback()
                        return
                    raw_counter = values.get("think_counter", "0")
                    try:
                        counter = int(raw_counter) + 1
                    except (TypeError, ValueError):
                        # Corrupt metadata must not escape through sync_turn or
                        # accidentally trigger an opaque think call. Reset it
                        # deterministically and wait for the next admitted turn.
                        logger.warning(
                            "think cycle counter metadata invalid; resetting counter"
                        )
                        conn.execute(
                            "INSERT OR REPLACE INTO hermes_provider_meta (key,value) "
                            "VALUES ('think_counter', '0')"
                        )
                        conn.commit()
                        return
                    if counter < self._config.think_interval:
                        conn.execute(
                            "INSERT OR REPLACE INTO hermes_provider_meta (key,value) "
                            "VALUES ('think_counter', ?)",
                            (str(counter),),
                        )
                        conn.commit()
                        return
                    claim_epoch = f"{self._health_generation}:{time.monotonic_ns()}"
                    with self._sync_state_lock:
                        self._think_claim_state = "in_flight"
                    conn.executemany(
                        "INSERT OR REPLACE INTO hermes_provider_meta (key,value) VALUES (?,?)",
                        [
                            ("think_counter", str(counter)),
                            ("think_claim_epoch", claim_epoch),
                            ("think_claim_state", "in_flight"),
                        ],
                    )
                    conn.commit()
                finally:
                    conn.close()

                try:
                    result = think_cycle(
                        db_path=str(self._db_path),
                        model_fn=self._model_fn,
                    )
                    progress = bool(
                        getattr(result, "new_nodes", None)
                        or getattr(result, "new_edges", None)
                    )
                    terminal = "completed" if progress else "failed"
                except Exception:
                    logger.warning("think cycle failed", exc_info=True)
                    terminal = "uncertain"
                    result = None
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT value FROM hermes_provider_meta WHERE key='think_claim_epoch'"
                    ).fetchone()
                    state_row = conn.execute(
                        "SELECT value FROM hermes_provider_meta WHERE key='think_claim_state'"
                    ).fetchone()
                    if (
                        claim_epoch is not None
                        and row is not None
                        and row[0] == claim_epoch
                        and state_row is not None
                        and state_row[0] == "in_flight"
                    ):
                        if terminal == "completed":
                            conn.execute(
                                "UPDATE hermes_provider_meta SET value=? WHERE key='think_counter'",
                                (str(max(0, counter - self._config.think_interval)),),
                            )
                        conn.execute(
                            "UPDATE hermes_provider_meta SET value=? WHERE key='think_claim_state'",
                            (terminal,),
                        )
                        with self._sync_state_lock:
                            self._think_claim_state = terminal
                    conn.commit()
                finally:
                    conn.close()
                if result is not None and getattr(result, "new_nodes", None):
                    logger.info(
                        "think cycle produced %d insight(s) on cluster: %s",
                        len(result.new_nodes),
                        getattr(result, "cluster_topic", None) or "unknown",
                    )
        except (OperationAdmissionError, sqlite3.Error):
            logger.info("think cycle deferred: admission or transaction unavailable")

    def _load_think_counter(self) -> int:
        """Read persistent think counter from DB. Resets to 0 on any error."""
        try:
            assert self._db_path is not None
            conn = self._connect_graph(self._db_path)
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
        if not self._embedding_identity_ready:
            return
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
        if (
            not self._write_enabled
            or not self._embedding_identity_ready
            or self._model_fn is None
            or self._db_path is None
            or self._initializing
            or self._shutdown_started.is_set()
        ):
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
        """Persist pre-compression insights under the graph/cache admission."""
        try:
            with self._operation_admission() as admission:
                self._validate_runtime_identity(admission)
                return self._create_insight_nodes_unlocked(items)
        except OperationAdmissionError:
            logger.info("on_pre_compress: operation rejected before admission")
            return 0

    def _create_insight_nodes_unlocked(self, items: list[dict]) -> int:
        """Create insight/observation nodes in the Cashew graph.

        Uses upstream _create_node / _set_node_tags for persistence, then
        calls embed_nodes to generate embeddings. Returns node count.

        Silent-degrades: logs warning on failure, never raises.
        """
        if not self._embedding_identity_ready:
            return 0

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

        Does NOT drain the sync queue — the background worker keeps running across
        session boundaries. Data-loss protection is handled by shutdown(), which
        stops producers, posts a sentinel, and bounded-joins the worker.

        Sleep cycle processing is handled by a Hermes cron job.
        See ``sleep_schedule`` in cashew.json.
        """
        if self._sync_queue is None:
            return  # not initialized or silent-degraded

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Rebind session identity and discard ephemeral context from the old session."""
        del parent_session_id, reset, rewound, kwargs
        with self._sync_state_lock:
            if self._initializing or self._shutdown_started.is_set():
                return
            self._session_id = str(new_session_id)
            self._prefetch_generation += 1
            self._warm_cache.clear()
            self._prefetch_pending = None
            self._prefetch_pending_request = None
            self._prefetch_latest_identity = None
            self._last_assistant = ""
            self._prefetch_condition.notify_all()

    def shutdown(self) -> None:  # noqa: C901 - teardown state machine is explicit
        """Stop producers, bounded-join background work, clear references.

        Order is load-bearing:
          1. If not initialized, return.
          2. Post _SHUTDOWN sentinel to the queue. put_nowait first; fallback to
             a 1s blocking put if the queue is full (worker is draining fast).
          3. Bounded-join the sync and prefetch workers using
             sync_queue_timeout. WARNING on timeout.
          4. Clear runtime state after the worker exits. If the bounded join
             times out, a daemon cleanup watcher retains that state until the
             worker actually finishes.

        _hermes_home is intentionally NOT reset — is_available() must keep
        reflecting on-disk reality.
        """
        # Serialize the ownership decision only. The bounded joins happen
        # outside both locks so sync_turn remains a non-blocking hot path.
        retry_cleanup: (
            tuple[threading.Thread | None, queue.Queue, tuple[threading.Thread, ...]]
            | None
        ) = None
        with self._lifecycle_lock:
            if self._initializing:
                if self._sync_worker is None:
                    # Do not tear down fields while initialize() is still
                    # constructing them. The initializer will observe this
                    # flag at its publication handoff and discard its partial
                    # state.
                    self._initialization_cancelled = True
                    self._shutdown_started.set()
                    return
                # A worker has already been published. Continue with normal
                # shutdown; initialize() retains ownership until its trace
                # context exits and its unconditional finalizer runs.
            with self._sync_state_lock:
                if self._sync_queue is None:
                    return  # safe no-op
                if self._shutdown_started.is_set():
                    retry_cleanup = self._shutdown_cleanup_pending
                    if retry_cleanup is None:
                        return  # shutdown already in progress
                else:
                    retry_cleanup = None
                if retry_cleanup is not None:
                    # Retry outside both provider locks. The prior shutdown
                    # already closed producer admission and the model callable.
                    pass
                else:
                    timeout = (
                        self._config.sync_queue_timeout
                        if self._config is not None
                        else 30.0
                    )
                    deadline = time.monotonic() + max(0.0, timeout)
                    self._shutdown_started.set()
                    self._set_health_locked("stopping", "shutdown_requested")
                    self._prefetch_generation += 1
                    self._prefetch_pending = None
                    model_fn = self._model_fn
                    self._prefetch_pending_request = None
                    self._prefetch_latest_identity = None
                    q = self._sync_queue
                    assert q is not None
                    prefetch_threads = tuple(self._prefetch_threads)
                    self._prefetch_condition.notify_all()
        if retry_cleanup is not None:
            self._schedule_shutdown_cleanup(*retry_cleanup)
            return
        _METRICS.emit()
        # Items already in the queue remain ahead of the sentinel and receive a
        # bounded opportunity to persist before the worker exits.
        try:
            q.put_nowait(_SHUTDOWN)
        except queue.Full:
            try:
                q.put(
                    _SHUTDOWN,
                    block=True,
                    # Keep the historical one-second sentinel wait cap while
                    # still charging it against the single shutdown deadline.
                    timeout=min(1.0, max(0.0, deadline - time.monotonic())),
                )
            except queue.Full:
                logger.warning(
                    "cashew shutdown: could not post sentinel; worker may leak"
                )
        # The signal and all joins share the deadline captured before
        # shutdown admission. A wedged/full queue must not get a fresh join
        # budget after the sentinel fallback has consumed the timeout.
        # Never raise.
        worker = self._sync_worker
        if worker is not None:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                logger.warning(
                    "cashew sync worker did not exit within %ss; abandoning",
                    timeout,
                )
        for prefetch_thread in prefetch_threads:
            prefetch_thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive_prefetch = tuple(t for t in prefetch_threads if t.is_alive())
        if alive_prefetch:
            logger.warning(
                "cashew prefetch worker(s) did not exit within %ss; retaining state",
                timeout,
            )
        alive_workers = tuple(
            t
            for t in ((worker,) if worker is not None else ()) + alive_prefetch
            if t.is_alive()
        )
        # Accepted turns retain their LLM callable until the worker drains or
        # its existing shutdown deadline expires. Closing earlier would turn
        # already-admitted LLM extraction into heuristic extraction.
        close_model = getattr(model_fn, "_cashew_close", None)
        if callable(close_model):
            close_model()
        if alive_workers:
            with self._sync_state_lock:
                self._set_health_locked("stopping", "worker_timeout")
                self._shutdown_cleanup_pending = (worker, q, alive_workers)
            self._schedule_shutdown_cleanup(worker, q, alive_workers)
            return
        self._clear_runtime_state(
            worker,
            q,
            embedding_close_timeout=max(0.0, deadline - time.monotonic()),
        )

    def _clear_state_after_workers_exit(
        self,
        sync_worker: threading.Thread | None,
        sync_queue: queue.Queue,
        workers: tuple[threading.Thread, ...],
    ) -> None:
        """Keep worker dependencies alive until every timed-out worker exits."""
        for worker in workers:
            worker.join()
        self._clear_runtime_state(sync_worker, sync_queue)

    def _schedule_shutdown_cleanup(
        self,
        sync_worker: threading.Thread | None,
        sync_queue: queue.Queue,
        workers: tuple[threading.Thread, ...],
    ) -> None:
        """Start or retry deferred cleanup without raising from shutdown()."""
        if not any(worker.is_alive() for worker in workers):
            self._clear_runtime_state(sync_worker, sync_queue)
            return
        try:
            cleanup = threading.Thread(
                target=self._clear_state_after_workers_exit,
                args=(sync_worker, sync_queue, workers),
                daemon=True,
                name=f"cashew-shutdown-{self._session_id}",
            )
            cleanup.start()
        except Exception:
            logger.warning(
                "cashew shutdown cleanup could not start; retry shutdown after worker exit"
            )
            return
        with self._sync_state_lock:
            if self._sync_queue is sync_queue:
                self._shutdown_cleanup_pending = None

    def _finish_embedding_runtime_close(
        self,
        supervisor: EmbeddingSupervisor | None,
        *,
        mark_stopped: bool,
    ) -> None:
        """Release lifecycle admission only after embedding ownership is gone."""
        with self._sync_state_lock:
            if supervisor is not None and self._embedding_supervisor is not supervisor:
                return
            self._embedding_supervisor = None
            if self._sync_queue is not None:
                return
            self._shutdown_started.clear()
            if mark_stopped:
                self._set_health_locked("stopped", "shutdown_complete")
        logger.debug("cashew provider shutdown complete")

    def _close_embedding_runtime(
        self,
        supervisor: EmbeddingSupervisor | None,
        *,
        timeout: float | None = None,
        mark_stopped: bool = False,
    ) -> None:
        """Close one embedding owner while retaining the shutdown admission gate."""
        if supervisor is None:
            self._finish_embedding_runtime_close(None, mark_stopped=mark_stopped)
            return
        supervisor._when_closed(
            lambda: self._finish_embedding_runtime_close(
                supervisor, mark_stopped=mark_stopped
            )
        )
        supervisor.close(timeout=timeout)

    def _clear_runtime_state(
        self,
        worker: threading.Thread | None,
        sync_queue: queue.Queue,
        *,
        embedding_close_timeout: float | None = None,
    ) -> None:
        """Clear provider state if it still belongs to the exiting worker."""
        with self._sync_state_lock:
            if self._sync_queue is not sync_queue:
                return
            if worker is not None and self._sync_worker is not worker:
                return
            # Sleep cycle cron job is intentionally NOT removed here.
            # It persists across session boundaries so the 12h schedule
            # isn't reset on every session start. The next initialize()
            # will adopt the existing job if one exists.
            self._sleep_cron_job_id = None  # clear instance tracking only
            # Clear state. _hermes_home persists (see is_available() contract).
            self._sync_queue = None
            self._shutdown_cleanup_pending = None
            self._sync_worker = None
            self._config = None
            self._db_path = None
            self._retriever = None
            self._model_fn = None
            self._embedding_identity_ready = False
            supervisor = self._embedding_supervisor
            telemetry = self._sentry_telemetry
            self._sentry_telemetry = None
            self._warm_cache.clear()
            self._prefetch_generation += 1
            self._prefetch_pending = None
            self._prefetch_threads.clear()
            self._prefetch_worker = None
            self._prefetch_active_identity = None
            self._prefetch_pending_request = None
            self._prefetch_latest_identity = None
            self._last_assistant = ""
            self._prefetch_condition.notify_all()
            self._set_health_locked("stopping", "embedding_shutdown")
        close_sentry_telemetry(telemetry)
        self._close_embedding_runtime(
            supervisor,
            timeout=embedding_close_timeout,
            mark_stopped=True,
        )
        if self._log_scrub_acquired:
            release_provider_scrub_filters()
            self._log_scrub_acquired = False

    def prefetch(  # noqa: C901 - retrieval fallback branches preserve the adapter contract
        self,
        query: str,
        domain: str | None = None,
        tag: str | None = None,
        exclude_tags: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return recalled-context string from Cashew (RECALL-01)."""
        with self._sync_state_lock:
            config = self._config
            db_path = self._db_path
            if (
                config is None
                or db_path is None
                or self._initializing
                or self._shutdown_started.is_set()
            ):
                return ""
            requested_session = str(kwargs.get("session_id") or self._session_id)
            ledger = self._outcomes
            generation = self._health_generation
            identity = self._prefetch_request_identity(
                session_id=requested_session,
                generation=self._prefetch_generation,
                domain=domain,
                tag=tag,
                exclude_tags=exclude_tags,
            )
            max_nodes = config.recall_k
            identity_ready = self._embedding_identity_ready
            # Consume the pending result and snapshot the warm cache while
            # the same runtime identity is admitted. Releasing the lock
            # between these steps could let a reinitialized profile publish a
            # same-session cache that belongs to a newer generation.
            pending = self._consume_prefetch_pending_locked(identity)
            self._warm_cache.update(pending)
            warm_cache = tuple(self._warm_cache.items())
            # Consume the snapshot before doing any potentially slow retrieval.
            # Shutdown may clear the live cache while this call continues.
            self._warm_cache.clear()
        # A blank query is never allowed to match every cached cue.
        if query.strip() and warm_cache:
            query_lower = query.lower()
            for cue, warm_result in warm_cache:
                if not cue or warm_result.identity != identity:
                    continue
                cue_lower = cue.lower()
                if cue_lower in query_lower or query_lower in cue_lower:
                    logger.info(
                        "prefetch warm cache HIT: cue_len=%d query_len=%d",
                        len(cue),
                        len(query),
                    )
                    return self._format_context(
                        copy.deepcopy(list(warm_result.nodes[: identity.recall_limit]))
                    )
                cue_words = set(w for w in cue_lower.split() if len(w) > 3)
                query_words = set(w for w in query_lower.split() if len(w) > 3)
                if len(cue_words & query_words) >= 2:
                    logger.info(
                        "prefetch warm cache HIT: cue_len=%d query_len=%d (word overlap)",
                        len(cue),
                        len(query),
                    )
                    return self._format_context(
                        copy.deepcopy(list(warm_result.nodes[: identity.recall_limit]))
                    )
            logger.info(
                "prefetch warm cache MISS (%d cue(s) in cache) — falling through to cold retrieval",
                len(warm_cache),
            )
        if not identity_ready:
            try:
                return self._format_context(
                    self._keyword_search(
                        query,
                        max_nodes,
                        domain,
                        tag,
                        exclude_tags,
                        db_path=db_path,
                    )
                )
            except Exception:
                logger.warning(
                    "cashew keyword-only recall failed (query_len=%d)",
                    len(query),
                    exc_info=True,
                )
                return ""
        with trace_operation(
            "cashew.prefetch",
            {"input_length": len(query)},
        ) as _span:
            vector_failed = False
            keyword_failed = False
            try:
                with self._operation_admission() as admission:
                    try:
                        results = _retrieve_with_embedding_wait(
                            db_path=str(db_path),
                            query=query,
                            top_k=max_nodes,
                            domain=domain,
                            tags=[tag] if tag else None,
                            exclude_tags=exclude_tags,
                        )
                    except Exception:
                        vector_failed = True
                        results = None
                        logger.debug(
                            "upstream retrieval failed, falling back to keyword",
                            exc_info=True,
                        )
                    if results:
                        node_ids = [r.node_id for r in results]
                        self._validate_runtime_identity(admission)
                        # Keep the original graph/cache token through
                        # enrichment, metrics, and publication.
                        nodes = self._enrich_results(node_ids, db_path=str(db_path))
                        self._update_access_metrics(node_ids, db_path=db_path)
                        self._validate_runtime_identity(admission)
                        return self._format_context(nodes)
                    try:
                        nodes = self._keyword_search(
                            query,
                            max_nodes,
                            domain,
                            tag,
                            exclude_tags,
                            db_path=db_path,
                        )
                    except Exception:
                        keyword_failed = True
                        raise
                    if nodes:
                        if vector_failed:
                            self._mark_health_if_current(
                                ledger,
                                generation,
                                "degraded",
                                "vector_unavailable",
                                fallback="keyword",
                            )
                        self._update_access_metrics(
                            [n["id"] for n in nodes], db_path=db_path
                        )
                        self._validate_runtime_identity(admission)
                        return self._format_context(nodes)
            except Exception:
                if not keyword_failed:
                    vector_failed = True
                logger.warning(
                    "cashew recall failed (query_len=%d)", len(query), exc_info=True
                )
            if keyword_failed:
                self._mark_health_if_current(
                    ledger, generation, "degraded", "backend_error"
                )
            elif vector_failed:
                self._mark_health_if_current(
                    ledger,
                    generation,
                    "degraded",
                    "vector_unavailable",
                    fallback="keyword",
                )
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm cashew memory for the next turn (ABC optional hook).

        Enqueues the latest request for the single daemon worker. One request
        may be active while one newer request waits in the coalescing slot;
        newer requests replace that slot without starting more workers.

        Contract:
        - Half-state guard: if _config is None, return silently.
        - Never blocks — returns in <1ms.
        - Never raises into Hermes (caught in background thread).
        """
        with self._prefetch_condition:
            config = self._config
            db_path = self._db_path
            if (
                config is None
                or db_path is None
                or self._initializing
                or self._shutdown_started.is_set()
            ):
                return
            effective_session = str(session_id or self._session_id)
            self._prefetch_generation += 1
            generation = self._prefetch_generation
            self._prefetch_pending = None
            identity = self._prefetch_request_identity(
                session_id=effective_session,
                generation=generation,
                domain=None,
                tag=None,
                exclude_tags=None,
            )
            if not query:
                logger.debug("queue_prefetch: empty query, no warmup")
                self._prefetch_latest_identity = None
                return
            request = _PrefetchRequest(
                identity=identity,
                query=query,
                db_path=str(db_path),
                top_k=config.prefetch_k,
                use_llm=(
                    self._embedding_identity_ready
                    and self._model_fn is not None
                    and config.prefetch_cues > 0
                ),
            )
            if self._prefetch_pending_request is not None:
                _METRICS.record_prefetch_coalesced()
            self._prefetch_pending_request = request
            self._prefetch_latest_identity = identity
            worker = self._prefetch_worker
            if worker is None or not worker.is_alive():
                worker = threading.Thread(
                    target=self._prefetch_worker_loop,
                    daemon=True,
                    name=f"cashew-prefetch-{self._session_id}",
                )
                self._prefetch_worker = worker
                self._prefetch_threads.add(worker)
                try:
                    worker.start()
                except Exception:
                    self._prefetch_threads.discard(worker)
                    self._prefetch_worker = None
                    self._prefetch_pending_request = None
                    _METRICS.record_prefetch_failed()
                    logger.debug(
                        "queue_prefetch: failed to start warmup worker", exc_info=True
                    )
                    return
            self._prefetch_condition.notify()

    def _prefetch_worker_loop(self) -> None:
        """Run at most one active request and one pending latest request."""
        current_thread = threading.current_thread()
        try:
            while True:
                with self._prefetch_condition:
                    while (
                        self._prefetch_pending_request is None
                        and not self._shutdown_started.is_set()
                    ):
                        # Keep the single provider-owned daemon alive until
                        # shutdown. Retiring after an idle timeout would leave
                        # a producer able to publish a pending request between
                        # the timeout check and worker finalization.
                        self._prefetch_condition.wait()
                    if (
                        self._shutdown_started.is_set()
                        and self._prefetch_pending_request is None
                    ):
                        return
                    request = self._prefetch_pending_request
                    self._prefetch_pending_request = None
                    assert request is not None
                    self._prefetch_active_identity = request.identity
                self._run_prefetch_request(request)
                with self._prefetch_condition:
                    self._prefetch_active_identity = None
                    self._prefetch_condition.notify_all()
        finally:
            with self._prefetch_condition:
                self._prefetch_threads.discard(current_thread)
                if self._prefetch_worker is current_thread:
                    self._prefetch_worker = None
                self._prefetch_active_identity = None
                self._prefetch_condition.notify_all()

    def _run_prefetch_request(self, request: _PrefetchRequest) -> None:
        """Execute one request, checking identity before expensive stages."""
        identity = request.identity
        if not self._prefetch_request_is_current(identity):
            _METRICS.record_prefetch_cancelled()
            return
        try:
            if request.use_llm:
                try:
                    cues = self._extract_prefetch_cues(request.query)
                    logger.info("queue_prefetch: extracted %d LLM cue(s)", len(cues))
                except Exception:
                    logger.debug(
                        "queue_prefetch: LLM cue extraction failed, using raw query"
                    )
                    cues = [request.query]
            else:
                cues = [request.query]
            cues = [cue for cue in cues if cue and cue.strip()]
            if not cues:
                return
            if not self._prefetch_request_is_current(identity):
                _METRICS.record_prefetch_cancelled()
                return

            seen_ids: set[str] = set()
            all_nodes: list[dict] = []
            for cue in cues:
                if not self._prefetch_request_is_current(identity):
                    _METRICS.record_prefetch_cancelled()
                    return
                nodes = self._prefetch_nodes_for_cue(cue, request)
                for node in nodes:
                    node_id = node.get("id", "")
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        all_nodes.append(node)
            if not self._prefetch_request_is_current(identity):
                _METRICS.record_prefetch_cancelled()
                return
            if all_nodes:
                self._stage_prefetch_result(identity, cues, all_nodes)
                logger.info(
                    "queue_prefetch: cached %d result(s) from %d cue(s)",
                    len(all_nodes),
                    len(cues),
                )
        except Exception:
            if self._prefetch_request_is_current(identity):
                _METRICS.record_prefetch_failed()
                logger.debug(
                    "queue_prefetch background worker failed (non-fatal)",
                    exc_info=True,
                )
            else:
                _METRICS.record_prefetch_cancelled()

    def _prefetch_nodes_for_cue(
        self, cue: str, request: _PrefetchRequest
    ) -> list[dict]:
        """Retrieve one cue without embedding when identity is unresolved."""
        with self._sync_state_lock:
            identity_ready = self._embedding_identity_ready
        if identity_ready:
            with self._operation_admission() as admission:
                results = _retrieve_with_embedding_wait(
                    db_path=request.db_path,
                    query=cue,
                    top_k=request.top_k,
                )
                if results:
                    self._validate_runtime_identity(admission)
                    nodes = self._enrich_results(
                        [result.node_id for result in results], db_path=request.db_path
                    )
                    self._validate_runtime_identity(admission)
                    return nodes
        return self._keyword_search(cue, request.top_k, db_path=request.db_path)

    def _prefetch_request_is_current(self, identity: _PrefetchRequestIdentity) -> bool:
        """Check request and complete runtime identity without slow work."""
        with self._sync_state_lock:
            if (
                self._shutdown_started.is_set()
                or self._initializing
                or self._prefetch_latest_identity != identity
                or self._config is None
                or self._db_path is None
            ):
                return False
            current_identity = self._prefetch_request_identity(
                session_id=identity.session_id,
                generation=self._prefetch_generation,
                domain=identity.domain,
                tag=identity.tag,
                exclude_tags=list(identity.exclude_tags),
            )
            return current_identity == identity

    def _stage_prefetch_result(
        self,
        identity: _PrefetchRequestIdentity,
        cues: list[str],
        nodes: list[dict],
    ) -> None:
        """Publish a warmup result only if its request is still current."""
        with self._sync_state_lock:
            current_identity = self._prefetch_request_identity(
                session_id=identity.session_id,
                generation=self._prefetch_generation,
                domain=identity.domain,
                tag=identity.tag,
                exclude_tags=list(identity.exclude_tags),
            )
            if identity != current_identity:
                return
            self._prefetch_pending = _PrefetchResult(
                identity=identity,
                cues=tuple(cue for cue in cues if cue.strip()),
                nodes=tuple(copy.deepcopy(nodes)),
            )

    def _consume_prefetch_pending(
        self, identity: _PrefetchRequestIdentity
    ) -> dict[str, _PrefetchResult]:
        """Atomically consume a current result, retaining its source cues."""
        with self._sync_state_lock:
            return self._consume_prefetch_pending_locked(identity)

    def _consume_prefetch_pending_locked(
        self, identity: _PrefetchRequestIdentity
    ) -> dict[str, _PrefetchResult]:
        """Consume a pending result while ``_sync_state_lock`` is held."""
        pending = self._prefetch_pending
        self._prefetch_pending = None
        if pending is None or pending.identity != identity:
            return {}
        return {cue: pending for cue in pending.cues}

    def _prefetch_request_identity(
        self,
        *,
        session_id: str,
        generation: int,
        domain: str | None,
        tag: str | None,
        exclude_tags: list[str] | None,
    ) -> _PrefetchRequestIdentity:
        """Snapshot every retrieval input that makes a warm result reusable."""
        config = self._config
        assert config is not None
        config_values = (
            dataclasses.asdict(config)
            if dataclasses.is_dataclass(config)
            else repr(config)
        )
        return _PrefetchRequestIdentity(
            generation=generation,
            session_id=session_id,
            db_path=str(self._db_path),
            config_fingerprint=json.dumps(
                config_values, sort_keys=True, separators=(",", ":"), default=repr
            ),
            recall_limit=config.recall_k,
            domain=domain or None,
            tag=tag or None,
            exclude_tags=tuple(
                sorted({value for value in exclude_tags or [] if value})
            ),
            embedding_epoch=self._runtime_epoch,
            embedding_generation=(
                getattr(self._embedding_supervisor, "generation", None)
            ),
        )

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
        db_path: pathlib.Path | str | None = None,
    ) -> list[dict]:
        target_db = db_path if db_path is not None else self._db_path
        if target_db is None:
            return []
        conn = self._connect_graph(target_db)
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

    def handle_tool_call(  # noqa: C901 - compatibility dispatcher keeps envelope branches together
        self, name: str, args: Dict[str, Any], **kwargs: Any
    ) -> str:
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

        ``kwargs`` is accepted for forward compatibility with Hermes' real
        ``MemoryManager`` dispatch, which supplies session and turn metadata
        to provider tool handlers.  Cashew tool behavior is unchanged; the
        current handlers do not need those optional values.

        Returns:
            JSON string — NEVER None, NEVER raises into Hermes.
        """
        if name == "cashew_query":
            if (
                self._config is None
                or self._initializing
                or self._shutdown_started.is_set()
            ):
                return build_error_envelope(
                    query=args.get("query"),
                    error_message="cashew recall failed",
                )
            with self._sync_state_lock:
                ledger = self._outcomes
                generation = self._health_generation
                identity_ready = self._embedding_identity_ready
            try:
                _t0 = time.perf_counter()
                query = args["query"]
                with trace_operation(
                    "cashew.query",
                    {"input_length": len(query)},
                ) as span:
                    max_nodes = args.get("max_nodes", self._config.recall_k)
                    domain = args.get("domain")
                    tag = args.get("tag")
                    exclude_tags = args.get("exclude_tags")
                    vector_failed = False
                    if identity_ready:
                        # One immutable admission owns every graph-facing
                        # phase.  In particular, enrichment and access metrics
                        # must not reacquire against a newer provider identity
                        # after BFS has produced its node IDs.
                        with self._operation_admission():
                            try:
                                results = _retrieve_with_embedding_wait(
                                    db_path=str(self._db_path),
                                    query=query,
                                    top_k=max_nodes,
                                    domain=domain,
                                    tags=[tag] if tag else None,
                                    exclude_tags=exclude_tags,
                                )
                            except Exception:
                                vector_failed = True
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
                    else:
                        results = None
                        nodes = self._keyword_search(
                            query, max_nodes, domain, tag, exclude_tags
                        )
                    if nodes:
                        context = self._format_context(nodes)
                        node_count = len(nodes)
                    else:
                        context = ""
                        node_count = 0
                    with self._sync_state_lock:
                        if self._outcome_current_locked(ledger, generation):
                            ledger.record_tool(
                                "cashew_query", success=True, empty=node_count == 0
                            )
                            if (
                                vector_failed
                                and identity_ready
                                and not self._shutdown_started.is_set()
                            ):
                                self._set_health_locked(
                                    "degraded",
                                    "vector_unavailable",
                                    fallback="keyword",
                                )
                    _elapsed_ms = (time.perf_counter() - _t0) * 1000
                    _METRICS.record_query(cache_hit=False, elapsed_ms=_elapsed_ms)
                    span.set_attribute("result_count", node_count)
                    span.set_attribute("duration_bucket_ms", _elapsed_ms)
                return build_success_envelope(
                    query=query,
                    context=context,
                    node_count=node_count,
                )
            except Exception as _exc:
                with self._sync_state_lock:
                    if self._outcome_current_locked(ledger, generation):
                        ledger.record_tool("cashew_query", success=False)
                        self._set_health_locked(
                            "degraded",
                            "backend_error",
                            error=_exc,
                            fallback="keyword",
                        )
                capture_exception(
                    _exc,
                    operation="cashew.query",
                    session_id=self._session_id,
                    extra={
                        "query_len": len(args.get("query", "")),
                        "max_nodes": args.get("max_nodes", "default"),
                    },
                    telemetry=self._sentry_telemetry,
                )
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
            if (
                not self._write_enabled
                or not self._embedding_identity_ready
                or self._db_path is None
                or self._config is None
                or self._initializing
                or self._shutdown_started.is_set()
            ):
                return build_extract_error_envelope()
            with self._sync_state_lock:
                ledger = self._outcomes
                generation = self._health_generation
            try:
                user = args["user_content"]  # KeyError caught below — tool-call failure
                assistant = args["assistant_content"]
                # Lazy import — keeps is_available free of core.session side effects.
                from core.session import end_session

                with self._operation_admission() as admission:
                    self._validate_runtime_identity(admission)
                    result = end_session(
                        db_path=str(self._db_path),
                        session_id=self._session_id,
                        conversation_text=f"User: {user}\nAssistant: {assistant}",
                        model_fn=self._model_fn,
                    )
                with self._sync_state_lock:
                    if self._outcome_current_locked(ledger, generation):
                        ledger.record_tool(
                            "cashew_extract",
                            success=True,
                            empty=not result.new_nodes and not result.new_edges,
                        )
                return build_extract_success_envelope(
                    new_nodes=len(result.new_nodes),
                    new_edges=len(result.new_edges),
                )
            except Exception as _exc:
                with self._sync_state_lock:
                    if self._outcome_current_locked(ledger, generation):
                        ledger.record_tool("cashew_extract", success=False)
                        self._set_health_locked("degraded", "backend_error", error=_exc)
                capture_exception(
                    _exc,
                    operation="cashew.extract",
                    session_id=self._session_id,
                    extra={"user_len": len(args.get("user_content", ""))},
                    telemetry=self._sentry_telemetry,
                )
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
        if (
            self._config is None
            or self._hermes_home is None
            or self._initializing
            or self._shutdown_started.is_set()
        ):
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
            conn = self._connect_graph(self._db_path)
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
