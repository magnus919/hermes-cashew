"""Pure config + path-resolution helpers for the Cashew memory provider.

This module has zero coupling to the Hermes ABC, the CashewMemoryProvider class,
or the Cashew runtime. It owns:
  - the expanded config schema (~31 flat JSON keys aligned with upstream Cashew)
  - the on-disk JSON layout under hermes_home (CONF-02, CONF-03)
  - the rule that every path derives from hermes_home (CONF-04)

CashewMemoryProvider in plugins/memory/cashew/__init__.py delegates to these
helpers; tests in tests/test_config_roundtrip.py exercise them directly.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FILENAME: str = "cashew.json"
"""The flat-layout JSON file save_config writes under hermes_home."""

DEFAULTS: dict[str, Any] = {
    # Core paths and models (4 existing)
    "cashew_db_path": "cashew/brain.db",
    "embedding_model": "all-MiniLM-L6-v2",
    "recall_k": 5,
    "sync_queue_timeout": 30.0,
    # Domains (6)
    "user_domain": "user",
    "ai_domain": "ai",
    "default_domain": "general",
    "auto_classify": True,
    "domain_classifications": ["personal", "work", "projects", "learning", "system"],
    "domain_separation_enabled": True,
    # Performance / retrieval (11)
    "token_budget": 2000,
    "walk_depth": 2,
    "similarity_threshold": 0.3,
    "access_weight": 0.2,
    "temporal_weight": 0.1,
    "clustering_eps": 0.35,
    "clustering_min_samples": 3,
    "novelty_threshold": 0.82,
    "confidence_threshold": 0.7,
    "max_think_iterations": 3,
    "think_cycle_nodes": 5,
    # GC (5)
    "gc_mode": "soft",
    "gc_threshold": 0.05,
    "gc_grace_days": 7,
    "gc_protect_types": ["seed", "core_memory"],
    "gc_think_cycle_penalty": 1.5,
    # Features (5)
    "auto_extraction": True,
    "think_cycles": True,
    "sleep_cycles": True,
    "decay_pruning": True,
    "pattern_detection": True,
}


@dataclasses.dataclass(frozen=True)
class CashewConfig:
    """Typed view over the Cashew config dict (~31 keys aligned with upstream)."""

    cashew_db_path: str = DEFAULTS["cashew_db_path"]
    embedding_model: str = DEFAULTS["embedding_model"]
    recall_k: int = DEFAULTS["recall_k"]
    sync_queue_timeout: float = DEFAULTS["sync_queue_timeout"]
    # Domains
    user_domain: str = DEFAULTS["user_domain"]
    ai_domain: str = DEFAULTS["ai_domain"]
    default_domain: str = DEFAULTS["default_domain"]
    auto_classify: bool = DEFAULTS["auto_classify"]
    domain_classifications: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULTS["domain_classifications"])
    )
    domain_separation_enabled: bool = DEFAULTS["domain_separation_enabled"]
    # Performance / retrieval
    token_budget: int = DEFAULTS["token_budget"]
    walk_depth: int = DEFAULTS["walk_depth"]
    similarity_threshold: float = DEFAULTS["similarity_threshold"]
    access_weight: float = DEFAULTS["access_weight"]
    temporal_weight: float = DEFAULTS["temporal_weight"]
    clustering_eps: float = DEFAULTS["clustering_eps"]
    clustering_min_samples: int = DEFAULTS["clustering_min_samples"]
    novelty_threshold: float = DEFAULTS["novelty_threshold"]
    confidence_threshold: float = DEFAULTS["confidence_threshold"]
    max_think_iterations: int = DEFAULTS["max_think_iterations"]
    think_cycle_nodes: int = DEFAULTS["think_cycle_nodes"]
    # GC
    gc_mode: str = DEFAULTS["gc_mode"]
    gc_threshold: float = DEFAULTS["gc_threshold"]
    gc_grace_days: int = DEFAULTS["gc_grace_days"]
    gc_protect_types: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULTS["gc_protect_types"])
    )
    gc_think_cycle_penalty: float = DEFAULTS["gc_think_cycle_penalty"]
    # Features
    auto_extraction: bool = DEFAULTS["auto_extraction"]
    think_cycles: bool = DEFAULTS["think_cycles"]
    sleep_cycles: bool = DEFAULTS["sleep_cycles"]
    decay_pruning: bool = DEFAULTS["decay_pruning"]
    pattern_detection: bool = DEFAULTS["pattern_detection"]


def _env_var_name(key: str) -> str:
    """Derive the CASHEW_* environment variable name for a config key.

    Rule: strip 'cashew_' prefix if present, uppercase, prepend 'CASHEW_'.
    Examples: 'cashew_db_path' → 'CASHEW_DB_PATH', 'user_domain' → 'CASHEW_USER_DOMAIN'.
    """
    suffix = key.removeprefix("cashew_")
    return f"CASHEW_{suffix.upper()}"


def get_user_domain(config: CashewConfig) -> str:
    """Return the configured user domain label (replaces hardcoded 'user')."""
    return config.user_domain


def get_ai_domain(config: CashewConfig) -> str:
    """Return the configured AI domain label (replaces hardcoded 'ai')."""
    return config.ai_domain


ENV_VAR_MAP: dict[str, str] = {key: _env_var_name(key) for key in DEFAULTS}
"""Mapping from config key to its CASHEW_* environment variable name."""


def get_config_schema() -> list[dict[str, Any]]:
    """Return the list-of-field-descriptors Hermes uses to drive `hermes memory setup`.

    Each element is a dict with keys: `key`, `description`, `default`, `env_var`.
    All fields have defaults — no required-from-user fields.
    """
    schema = [
        {
            "key": "cashew_db_path",
            "description": (
                "Path to the Cashew SQLite brain DB, relative to hermes_home. "
                "Absolute paths are rejected to preserve profile isolation."
            ),
            "default": DEFAULTS["cashew_db_path"],
            "env_var": _env_var_name("cashew_db_path"),
        },
        {
            "key": "embedding_model",
            "description": "Sentence-transformers model identifier Cashew loads on first use.",
            "default": DEFAULTS["embedding_model"],
            "env_var": _env_var_name("embedding_model"),
        },
        {
            "key": "recall_k",
            "description": "How many context fragments prefetch() requests from Cashew per turn.",
            "default": DEFAULTS["recall_k"],
            "env_var": _env_var_name("recall_k"),
        },
        {
            "key": "sync_queue_timeout",
            "description": (
                "Bounded join timeout (seconds) shutdown() applies when draining the sync queue."
            ),
            "default": DEFAULTS["sync_queue_timeout"],
            "env_var": _env_var_name("sync_queue_timeout"),
        },
        {
            "key": "user_domain",
            "description": "Domain label for user-created nodes (replaces hardcoded 'user').",
            "default": DEFAULTS["user_domain"],
            "env_var": _env_var_name("user_domain"),
        },
        {
            "key": "ai_domain",
            "description": "Domain label for AI-generated nodes (replaces hardcoded 'ai').",
            "default": DEFAULTS["ai_domain"],
            "env_var": _env_var_name("ai_domain"),
        },
        {
            "key": "default_domain",
            "description": "Fallback domain when none is specified during extraction.",
            "default": DEFAULTS["default_domain"],
            "env_var": _env_var_name("default_domain"),
        },
        {
            "key": "auto_classify",
            "description": "Automatically assign domain classifications during node extraction.",
            "default": DEFAULTS["auto_classify"],
            "env_var": _env_var_name("auto_classify"),
        },
        {
            "key": "domain_classifications",
            "description": "List of domain tags available for auto-classification.",
            "default": DEFAULTS["domain_classifications"],
            "env_var": _env_var_name("domain_classifications"),
        },
        {
            "key": "domain_separation_enabled",
            "description": "Keep user and AI nodes in separate domains by default.",
            "default": DEFAULTS["domain_separation_enabled"],
            "env_var": _env_var_name("domain_separation_enabled"),
        },
        {
            "key": "token_budget",
            "description": "Max tokens to inject into context generation per retrieval.",
            "default": DEFAULTS["token_budget"],
            "env_var": _env_var_name("token_budget"),
        },
        {
            "key": "walk_depth",
            "description": "Graph BFS walk depth for context expansion from seed nodes.",
            "default": DEFAULTS["walk_depth"],
            "env_var": _env_var_name("walk_depth"),
        },
        {
            "key": "similarity_threshold",
            "description": "Minimum cosine similarity (0-1) for vec search results.",
            "default": DEFAULTS["similarity_threshold"],
            "env_var": _env_var_name("similarity_threshold"),
        },
        {
            "key": "access_weight",
            "description": "Weight of access_count in hybrid scoring (0-1).",
            "default": DEFAULTS["access_weight"],
            "env_var": _env_var_name("access_weight"),
        },
        {
            "key": "temporal_weight",
            "description": "Weight of recency in hybrid scoring (0-1).",
            "default": DEFAULTS["temporal_weight"],
            "env_var": _env_var_name("temporal_weight"),
        },
        {
            "key": "clustering_eps",
            "description": "DBSCAN epsilon for hierarchical clustering during sleep.",
            "default": DEFAULTS["clustering_eps"],
            "env_var": _env_var_name("clustering_eps"),
        },
        {
            "key": "clustering_min_samples",
            "description": "DBSCAN min_samples for hierarchical clustering.",
            "default": DEFAULTS["clustering_min_samples"],
            "env_var": _env_var_name("clustering_min_samples"),
        },
        {
            "key": "novelty_threshold",
            "description": "Score threshold (0-1) below which nodes are considered redundant.",
            "default": DEFAULTS["novelty_threshold"],
            "env_var": _env_var_name("novelty_threshold"),
        },
        {
            "key": "confidence_threshold",
            "description": "Minimum confidence (0-1) for extracted nodes to be persisted.",
            "default": DEFAULTS["confidence_threshold"],
            "env_var": _env_var_name("confidence_threshold"),
        },
        {
            "key": "max_think_iterations",
            "description": "Max autonomous think-cycle iterations per run.",
            "default": DEFAULTS["max_think_iterations"],
            "env_var": _env_var_name("max_think_iterations"),
        },
        {
            "key": "think_cycle_nodes",
            "description": "Number of seed nodes to use in a think cycle.",
            "default": DEFAULTS["think_cycle_nodes"],
            "env_var": _env_var_name("think_cycle_nodes"),
        },
        {
            "key": "gc_mode",
            "description": "Garbage collection mode: soft, hard, or off.",
            "default": DEFAULTS["gc_mode"],
            "env_var": _env_var_name("gc_mode"),
        },
        {
            "key": "gc_threshold",
            "description": "Relevance score below which nodes are eligible for GC.",
            "default": DEFAULTS["gc_threshold"],
            "env_var": _env_var_name("gc_threshold"),
        },
        {
            "key": "gc_grace_days",
            "description": "Days since last_accessed before a low-score node can be GC'd.",
            "default": DEFAULTS["gc_grace_days"],
            "env_var": _env_var_name("gc_grace_days"),
        },
        {
            "key": "gc_protect_types",
            "description": "Node types immune from garbage collection.",
            "default": DEFAULTS["gc_protect_types"],
            "env_var": _env_var_name("gc_protect_types"),
        },
        {
            "key": "gc_think_cycle_penalty",
            "description": "Multiplier on gc_threshold for think-cycle nodes.",
            "default": DEFAULTS["gc_think_cycle_penalty"],
            "env_var": _env_var_name("gc_think_cycle_penalty"),
        },
        {
            "key": "auto_extraction",
            "description": "Enable automatic knowledge extraction from conversation turns.",
            "default": DEFAULTS["auto_extraction"],
            "env_var": _env_var_name("auto_extraction"),
        },
        {
            "key": "think_cycles",
            "description": "Enable autonomous think cycles for cross-domain connection discovery.",
            "default": DEFAULTS["think_cycles"],
            "env_var": _env_var_name("think_cycles"),
        },
        {
            "key": "sleep_cycles",
            "description": "Enable sleep cycles for deep graph consolidation.",
            "default": DEFAULTS["sleep_cycles"],
            "env_var": _env_var_name("sleep_cycles"),
        },
        {
            "key": "decay_pruning",
            "description": "Enable organic decay of low-value nodes over time.",
            "default": DEFAULTS["decay_pruning"],
            "env_var": _env_var_name("decay_pruning"),
        },
        {
            "key": "pattern_detection",
            "description": "Enable automatic pattern detection across node clusters.",
            "default": DEFAULTS["pattern_detection"],
            "env_var": _env_var_name("pattern_detection"),
        },
    ]
    return schema


def resolve_config_path(hermes_home: str | os.PathLike[str]) -> pathlib.Path:
    """Return the path Cashew's flat-layout config file lives at: $HERMES_HOME/cashew.json."""
    return pathlib.Path(hermes_home) / CONFIG_FILENAME


def resolve_db_path(hermes_home: str | os.PathLike[str], db_path_value: str) -> pathlib.Path:
    """Resolve the Cashew DB path under hermes_home.

    `db_path_value` is the user-configured string from `cashew_db_path`. Absolute
    paths are REJECTED with `ValueError` to preserve profile isolation
    (CONF-04). The default `cashew/brain.db` resolves to
    `$HERMES_HOME/cashew/brain.db`.
    """
    if pathlib.PurePath(db_path_value).is_absolute():
        raise ValueError(
            f"cashew_db_path must be relative to hermes_home; got absolute path {db_path_value!r}. "
            "Configure a relative path (e.g. 'cashew/brain.db') instead."
        )
    return pathlib.Path(hermes_home) / db_path_value


def load_config(hermes_home: str | os.PathLike[str]) -> CashewConfig:
    """Read $HERMES_HOME/cashew.json and return a fully-populated CashewConfig.

    Missing keys are filled from DEFAULTS. If the file does not exist, returns
    `CashewConfig(**DEFAULTS)` — callers that need to distinguish "no file" from
    "default values" should use `resolve_config_path(...).exists()` directly.

    CASHEW_* environment variables override corresponding config keys with type coercion:
    - bool: env_val.lower() in ("1", "true", "yes", "on")
    - int: int(env_val)
    - float: float(env_val)
    - list: split on comma, strip whitespace, drop empties
    - str: pass through as-is

    Invalid env var values are logged and skipped (do NOT crash load_config).
    """
    path = resolve_config_path(hermes_home)
    if not path.exists():
        merged: dict[str, Any] = dict(DEFAULTS)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a JSON object, got {type(raw).__name__}")
        merged = {**DEFAULTS, **raw}

    for key, default_val in DEFAULTS.items():
        env_name = _env_var_name(key)
        env_val = os.environ.get(env_name)
        if env_val is not None:
            try:
                if isinstance(default_val, bool):
                    merged[key] = env_val.lower() in ("1", "true", "yes", "on")
                elif isinstance(default_val, int):
                    merged[key] = int(env_val)
                elif isinstance(default_val, float):
                    merged[key] = float(env_val)
                elif isinstance(default_val, list):
                    merged[key] = [item.strip() for item in env_val.split(",") if item.strip()]
                else:
                    merged[key] = env_val
            except ValueError:
                logger.warning("Invalid value for %s (%s), skipping: %r", key, env_name, env_val)

    known = {f.name for f in dataclasses.fields(CashewConfig)}
    filtered = {k: v for k, v in merged.items() if k in known}
    return CashewConfig(**filtered)


def save_config(values: dict[str, Any], hermes_home: str | os.PathLike[str]) -> pathlib.Path:
    """Persist the provider config to $HERMES_HOME/cashew.json (CONF-02).

    Preserves unknown keys from existing files. Unknown keys in `values` are dropped.
    Returns the path written.

    Writes are UTF-8, 2-space indent, sorted keys, trailing newline — stable
    diff-friendly format. Parent directory is created with `parents=True,
    exist_ok=True`.
    """
    path = resolve_config_path(hermes_home)
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except (json.JSONDecodeError, ValueError):
            existing = {}
    known = {f.name for f in dataclasses.fields(CashewConfig)}
    merged: dict[str, Any] = dict(existing)
    for k, v in DEFAULTS.items():
        if k not in merged:
            merged[k] = v
    for k, v in values.items():
        if k in known:
            merged[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.debug("wrote cashew config to %s", path)
    return path
