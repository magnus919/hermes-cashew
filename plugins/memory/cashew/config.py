"""Pure config + path-resolution helpers for the Cashew memory provider.

|This module has zero coupling to the Hermes ABC, the CashewMemoryProvider class,
|or the Cashew runtime. It owns:
|  - the 17-field runtime-backed config schema
  - the on-disk JSON layout under hermes_home (CONF-02, CONF-03)
  - the rule that every path derives from hermes_home (CONF-04)

CashewMemoryProvider in plugins/memory/cashew/__init__.py delegates to these
helpers; tests in tests/test_config_roundtrip.py exercise them directly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import os
import pathlib
import stat
import tempfile
import threading
from collections.abc import Iterator
from typing import Any, Callable

logger = logging.getLogger(__name__)

_CONFIG_SAVE_LOCK = threading.RLock()
"""Serialize in-process config writes before taking the profile lock."""

CONFIG_FILENAME: str = "cashew.json"
"""The flat-layout JSON file save_config writes under hermes_home."""

DEFAULTS: dict[str, Any] = {
    # Core paths, models, and device selection
    "cashew_db_path": "cashew/brain.db",
    "embedding_model": "thenlper/gte-large",
    "embedding_device": "cpu",
    "recall_k": 5,
    "sync_queue_timeout": 30.0,
    # Domain labels
    "user_domain": "user",
    "ai_domain": "ai",
    # Runtime behavior switches
    "auto_extraction": True,
    "think_cycles": True,
    "sleep_cycles": True,
    # LLM integration — "memory" activates LLM-powered extraction by default
    # via auxiliary.memory in Hermes config.yaml. On first load, the plugin
    # auto-populates auxiliary.memory from the main model config if absent.
    "llm_aux_role": "memory",
    "think_interval": 10,
    # Prefetch warmup
    "prefetch_k": 3,
    "prefetch_cues": 3,
    # Sleep cycle cron scheduling (2)
    "sleep_schedule": "every 12h",
    "sleep_max_nodes": 2000,
    # Feature flags for experimental behavior gating.
    # Agents and users can toggle these to opt into experimental features
    # without affecting stable code paths. All default to false.
    "_features": {
        "experimental_batch_sync": False,
        "experimental_parallel_retrieval": False,
    },
}

# Removed from the runtime surface after a full deprecation cycle. This list is
# used only to prune obsolete values from existing cashew.json files on save.
REMOVED_LEGACY_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "default_domain",
        "auto_classify",
        "domain_classifications",
        "domain_separation_enabled",
        "token_budget",
        "walk_depth",
        "similarity_threshold",
        "access_weight",
        "temporal_weight",
        "clustering_eps",
        "clustering_min_samples",
        "novelty_threshold",
        "max_think_iterations",
        "think_cycle_nodes",
        "gc_mode",
        "gc_threshold",
        "gc_grace_days",
        "gc_protect_types",
        "gc_think_cycle_penalty",
        "decay_pruning",
        "pattern_detection",
    }
)


@dataclasses.dataclass(frozen=True)
class CashewConfig:
    """Typed view over the 17 behavior-backed Cashew adapter settings."""

    cashew_db_path: str = DEFAULTS["cashew_db_path"]
    embedding_model: str = DEFAULTS["embedding_model"]
    embedding_device: str = DEFAULTS["embedding_device"]
    recall_k: int = DEFAULTS["recall_k"]
    sync_queue_timeout: float = DEFAULTS["sync_queue_timeout"]
    # Domain labels
    user_domain: str = DEFAULTS["user_domain"]
    ai_domain: str = DEFAULTS["ai_domain"]
    # Runtime behavior switches
    auto_extraction: bool = DEFAULTS["auto_extraction"]
    think_cycles: bool = DEFAULTS["think_cycles"]
    sleep_cycles: bool = DEFAULTS["sleep_cycles"]
    # LLM integration
    llm_aux_role: str | None = DEFAULTS["llm_aux_role"]
    think_interval: int = DEFAULTS["think_interval"]
    # Prefetch warmup
    prefetch_k: int = DEFAULTS["prefetch_k"]
    prefetch_cues: int = DEFAULTS["prefetch_cues"]
    # Sleep cycle cron scheduling
    sleep_schedule: str = DEFAULTS["sleep_schedule"]
    sleep_max_nodes: int = DEFAULTS["sleep_max_nodes"]
    # Feature flags
    _features: dict[str, bool] = dataclasses.field(
        default_factory=lambda: dict(DEFAULTS["_features"])
    )


def is_feature_enabled(config: CashewConfig, flag: str) -> bool:
    """Check whether an experimental feature flag is enabled.

    Feature flags are stored in the ``_features`` dict of the config.
    Unknown flags default to False. This allows agents to ship new
    behavior behind a toggle without affecting the stable code path.

    Example:
        if is_feature_enabled(config, "experimental_batch_sync"):
            _batch_drain(queue, config)

    Flags are configured in cashew.json:
        {"_features": {"experimental_batch_sync": true}}
    """
    return config._features.get(flag, False)


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


_COUNT_RANGES: dict[str, tuple[int, int]] = {
    "recall_k": (1, 20),
    "think_interval": (0, 10_000),
    "prefetch_k": (1, 20),
    "prefetch_cues": (0, 20),
    "sleep_max_nodes": (1, 2_000),
}
_BOOL_FIELDS = {"auto_extraction", "think_cycles", "sleep_cycles"}
_NONEMPTY_STRING_FIELDS = {
    "embedding_model",
    "embedding_device",
    "user_domain",
    "ai_domain",
}


def _invalid_value(key: str, expectation: str) -> ValueError:
    """Build concise config errors without including untrusted values."""
    return ValueError(f"{key} must be {expectation}")


def _validate_nonempty_string(key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid_value(key, "a non-empty string")
    return value


def _validate_count(key: str, value: Any) -> int:
    lower, upper = _COUNT_RANGES[key]
    if type(value) is not int or not lower <= value <= upper:
        raise _invalid_value(key, f"an integer from {lower} to {upper}")
    return value


def _validate_timeout(value: Any) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 300
    ):
        raise _invalid_value("sync_queue_timeout", "a finite number from 0 to 300")
    return float(value)


def _validate_features(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise _invalid_value("_features", "a JSON object of boolean feature flags")
    if any(not isinstance(name, str) or not name for name in value):
        raise _invalid_value("_features", "an object with non-empty string flag names")
    if any(type(enabled) is not bool for enabled in value.values()):
        raise _invalid_value("_features", "an object of boolean feature flags")
    return dict(value)


def _validate_field(key: str, value: Any, hermes_home: pathlib.Path) -> Any:
    """Validate and normalize one persisted runtime setting."""
    if key == "cashew_db_path":
        value = _validate_nonempty_string(key, value)
        resolve_db_path(hermes_home, value)
        return value

    if key in _NONEMPTY_STRING_FIELDS:
        return _validate_nonempty_string(key, value)

    if key == "llm_aux_role":
        if value is not None and not isinstance(value, str):
            raise _invalid_value(key, "a string, empty string, or null")
        return value

    if key == "sleep_schedule":
        if not isinstance(value, str):
            raise _invalid_value(key, "a string")
        return value

    if key in _BOOL_FIELDS:
        if type(value) is not bool:
            raise _invalid_value(key, "a boolean")
        return value

    if key in _COUNT_RANGES:
        return _validate_count(key, value)

    if key == "sync_queue_timeout":
        return _validate_timeout(value)

    if key == "_features":
        return _validate_features(value)

    raise AssertionError(f"Unhandled Cashew config field: {key}")


def _validate_config_values(
    values: dict[str, Any], hermes_home: pathlib.Path
) -> dict[str, Any]:
    """Validate all known fields and return a normalized dataclass payload."""
    return {key: _validate_field(key, values[key], hermes_home) for key in DEFAULTS}


def _coerce_environment_value(key: str, value: str) -> Any:
    """Parse one CASHEW_* override before normal field validation."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError("invalid boolean")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if key == "_features":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("invalid feature object")
        return parsed
    return value


@contextlib.contextmanager
def _exclusive_config_lock(path: pathlib.Path) -> Iterator[None]:
    """Hold a stable profile-local advisory lock for a complete read/replace."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _CONFIG_SAVE_LOCK:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _read_existing_config(path: pathlib.Path) -> dict[str, Any]:
    """Read an existing config without turning corruption into data loss."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Cannot save Cashew config because {path} is not valid JSON; "
            "correct the file before saving."
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Cannot save Cashew config because {path} must contain a JSON object."
        )
    return loaded


def _fsync_directory(directory: pathlib.Path) -> None:
    """Persist an atomic replacement's directory entry before reporting success."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_config(path: pathlib.Path, contents: str) -> None:
    """Write config through a unique same-directory temporary file and replace."""
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = pathlib.Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


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
            "key": "embedding_device",
            "description": (
                "Sentence-transformers device. Defaults to 'cpu' for process "
                "stability; use 'auto', 'mps', 'cuda', or a device index only "
                "after validating that backend in your environment."
            ),
            "default": DEFAULTS["embedding_device"],
            "env_var": _env_var_name("embedding_device"),
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
            "key": "llm_aux_role",
            "description": (
                "Hermes auxiliary role for LLM-powered operations "
                "(think cycles, sleep synthesis, LLM extraction). "
                "Defaults to 'memory', which reads from auxiliary.memory "
                "in Hermes config.yaml. The plugin auto-populates "
                "auxiliary.memory from the main model config on first load "
                "if absent. Set to null or empty string to disable LLM "
                "extraction and use heuristic-only mode."
            ),
            "default": DEFAULTS["llm_aux_role"],
            "env_var": _env_var_name("llm_aux_role"),
        },
        {
            "key": "think_interval",
            "description": (
                "Number of sync turns between think cycles. When an LLM is "
                "configured (via llm_aux_role), the plugin runs upstream "
                "think_cycle() every N turns to discover cross-domain "
                "connections and generate insight nodes. Set to 0 to disable."
            ),
            "default": DEFAULTS["think_interval"],
            "env_var": _env_var_name("think_interval"),
        },
        {
            "key": "prefetch_k",
            "description": (
                "Number of results to retrieve per cue during background "
                "prefetch warmup. Higher values warm more context but cost "
                "more vector searches."
            ),
            "default": DEFAULTS["prefetch_k"],
            "env_var": _env_var_name("prefetch_k"),
        },
        {
            "key": "prefetch_cues",
            "description": (
                "Number of LLM-extracted search cues for prefetch warmup "
                "(0 to disable LLM cues, using raw query only). Only applies "
                "when llm_aux_role is configured."
            ),
            "default": DEFAULTS["prefetch_cues"],
            "env_var": _env_var_name("prefetch_cues"),
        },
        {
            "key": "sleep_schedule",
            "description": (
                "Cron schedule expression or interval string for the sleep cycle. "
                "Set to empty string to disable cron-based scheduling entirely. "
                "Examples: 'every 30m', '0 */2 * * *', 'every 1h'. "
                "The sleep cycle runs as a no_agent cron script — no LLM overhead per tick."
            ),
            "default": DEFAULTS["sleep_schedule"],
            "env_var": _env_var_name("sleep_schedule"),
        },
        {
            "key": "sleep_max_nodes",
            "description": (
                "Maximum number of nodes to cross-link in a single sleep cycle. "
                "Higher values converge faster but take longer per tick. "
                "Previously hardcoded at 2000."
            ),
            "default": DEFAULTS["sleep_max_nodes"],
            "env_var": _env_var_name("sleep_max_nodes"),
        },
        {
            "key": "_features",
            "description": (
                "Experimental feature flags as a boolean key-value object. "
                "Flags are off by default; enable individually to opt into "
                "new behavior. See README Feature Flags section for available keys."
            ),
            "default": DEFAULTS["_features"],
            "env_var": "",
        },
    ]
    return schema


_PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "opencode-go": "OPENCODE_GO_API_KEY",
    "opencode-zen": "OPENCODE_ZEN_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "github": "COPILOT_GITHUB_TOKEN",
    "huggingface": "HF_TOKEN",
    "qwen": "QWEN_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "minimax-cn": "MINIMAX_CN_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}

# Well-known inference base URLs for OpenAI-compatible providers.
# Mirrors Hermes core's PROVIDER_REGISTRY inference_base_url values.
# Used as fallback when auxiliary.<role>.base_url is missing or empty.
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "opencode-zen": "https://opencode.ai/zen/v1",
    "opencode-go": "https://opencode.ai/zen/go/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "minimax": "https://api.minimaxi.chat/v1",
    "minimax-cn": "https://api.minimax.chat/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}
# NOTE: "anthropic", "google", and "huggingface" are intentionally omitted.
# They do not expose OpenAI-compatible /chat/completions endpoints natively;
# users targeting them must set an explicit base_url in config.yaml.


def _read_cashew_config(hermes_home: pathlib.Path) -> CashewConfig | None:
    """Read cashew.json and return a CashewConfig, or None if absent/unparseable."""
    cashew_path = resolve_config_path(hermes_home)
    if not cashew_path.exists():
        logger.debug(
            "cashew.json not found at %s; cannot resolve model_fn", cashew_path
        )
        return None
    try:
        return load_config(hermes_home)
    except Exception:
        logger.warning(
            "Failed to load cashew.json from %s; cannot resolve model_fn",
            cashew_path,
            exc_info=True,
        )
        return None


def _auxiliary_role_config(raw: Any, role: str) -> dict[str, Any] | None:
    """Extract one auxiliary role mapping without trusting YAML shapes."""
    if not isinstance(raw, dict):
        logger.warning(
            "llm_aux_role=%r: config.yaml must contain an object; "
            "falling back to heuristic extraction",
            role,
        )
        return None
    auxiliary = raw.get("auxiliary", {})
    if not isinstance(auxiliary, dict):
        logger.warning(
            "llm_aux_role=%r: auxiliary config must be an object; "
            "falling back to heuristic extraction",
            role,
        )
        return None
    aux_config = auxiliary.get(role, {})
    if not isinstance(aux_config, dict):
        logger.warning(
            "llm_aux_role=%r: auxiliary role config must be an object; "
            "falling back to heuristic extraction",
            role,
        )
        return None
    return aux_config


def resolve_model_fn(
    hermes_home: pathlib.Path,
    config: CashewConfig | None = None,
) -> Callable[[str], str] | None:
    """Resolve an OpenAI-compatible LLM callable from Hermes auxiliary config.

    Reads ``llm_aux_role`` from ``cashew.json`` (or the provided
    *config*), then ``auxiliary.<role>`` from ``config.yaml``, resolves
    the API key from config or well-known env vars, and returns an
    ``httpx``-based callable suitable for dream generation and text
    extraction.

    Returns ``None`` when:
    - No ``llm_aux_role`` is configured
    - ``config.yaml`` is absent or unparseable
    - The auxiliary section or API key cannot be found (logs warning)

    Designed for use by both ``CashewMemoryProvider`` (which passes its
    already-loaded ``CashewConfig``) and the sleep cron script (which
    passes ``None`` and lets the function load ``cashew.json`` itself).
    """
    # Resolve llm_aux_role from config
    if config is None:
        config = _read_cashew_config(hermes_home)
    if config is None:
        return None

    role = config.llm_aux_role
    if not role:
        logger.debug("llm_aux_role not set; heuristic-only mode — no model_fn")
        return None

    # Read auxiliary config from config.yaml
    config_yaml_path = hermes_home / "config.yaml"
    if not config_yaml_path.exists():
        logger.info(
            "llm_aux_role=%r but %s not found; falling back to heuristic extraction",
            role,
            config_yaml_path,
        )
        return None

    try:
        import yaml  # type: ignore[import-untyped, unused-ignore]

        raw = yaml.safe_load(config_yaml_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning(
            "llm_aux_role=%r: failed to parse %s; falling back to heuristic extraction",
            role,
            config_yaml_path,
            exc_info=True,
        )
        return None

    aux_config = _auxiliary_role_config(raw, role)
    if aux_config is None:
        return None

    model = aux_config.get("model")
    if not isinstance(model, str) or not model:
        logger.warning(
            "llm_aux_role=%r: no model in auxiliary.%s config; "
            "falling back to heuristic extraction",
            role,
            role,
        )
        return None

    provider = aux_config.get("provider", "openai")
    if not isinstance(provider, str) or not provider:
        logger.warning(
            "llm_aux_role=%r: provider must be a non-empty string; "
            "falling back to heuristic extraction",
            role,
        )
        return None
    configured_base_url = aux_config.get("base_url")
    if configured_base_url is not None and not isinstance(configured_base_url, str):
        logger.warning(
            "llm_aux_role=%r: base_url must be a string; "
            "falling back to heuristic extraction",
            role,
        )
        return None
    base_url = (
        configured_base_url
        or _PROVIDER_BASE_URLS.get(provider, "https://api.openai.com/v1")
    ).rstrip("/")

    # Resolve API key: explicit config > env var by provider convention
    api_key = aux_config.get("api_key")
    if not api_key:
        env_var = _PROVIDER_ENV_MAP.get(provider)
        if env_var:
            api_key = os.environ.get(env_var)

    if not isinstance(api_key, str) or not api_key:
        logger.warning(
            "llm_aux_role=%r: no API key for provider=%r; "
            "falling back to heuristic extraction",
            role,
            provider,
        )
        return None

    logger.info(
        "llm_aux_role=%r: using %s %s via %s",
        role,
        provider,
        model,
        base_url,
    )

    def _model_fn(prompt: str) -> str:
        """OpenAI-compatible chat completion callable."""
        try:
            import httpx

            resp = httpx.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
            resp.raise_for_status()
            data: Any = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception:
            logger.warning(
                "LLM call failed for role=%r (model=%s)",
                role,
                model,
                exc_info=True,
            )
            return ""

    return _model_fn


def resolve_config_path(hermes_home: str | os.PathLike[str]) -> pathlib.Path:
    """Return the path Cashew's flat-layout config file lives at: $HERMES_HOME/cashew.json."""
    return pathlib.Path(hermes_home) / CONFIG_FILENAME


def resolve_db_path(
    hermes_home: str | os.PathLike[str], db_path_value: str
) -> pathlib.Path:
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

    home = pathlib.Path(hermes_home).resolve()
    candidate = (home / db_path_value).resolve()
    try:
        candidate.relative_to(home)
    except ValueError as exc:
        raise ValueError(
            "cashew_db_path must stay within hermes_home; "
            f"got path {db_path_value!r} which resolves outside {str(home)!r}."
        ) from exc
    return candidate


def load_config(hermes_home: str | os.PathLike[str]) -> CashewConfig:
    """Read $HERMES_HOME/cashew.json and return a fully-populated CashewConfig.

    Missing keys are filled from DEFAULTS. If the file does not exist, returns
    `CashewConfig(**DEFAULTS)` — callers that need to distinguish "no file" from
    "default values" should use `resolve_config_path(...).exists()` directly.

    CASHEW_* environment variables override corresponding config keys with
    validated type coercion. Invalid overrides are logged without their values
    and leave the JSON/default value in place.
    """
    home = pathlib.Path(hermes_home)
    path = resolve_config_path(home)
    if not path.exists():
        merged: dict[str, Any] = dict(DEFAULTS)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"{path} must contain a JSON object, got {type(raw).__name__}"
            )
        merged = {**DEFAULTS, **raw}

    for key in DEFAULTS:
        env_name = ENV_VAR_MAP[key]
        env_val = os.environ.get(env_name)
        if env_val is not None:
            try:
                candidate = _coerce_environment_value(key, env_val)
                merged[key] = _validate_field(key, candidate, home)
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "Ignoring invalid value for environment variable %s", env_name
                )

    known_values = {key: merged[key] for key in DEFAULTS}
    return CashewConfig(**_validate_config_values(known_values, home))


def generate_default_config(hermes_home: str | os.PathLike[str]) -> pathlib.Path:
    """Write a default cashew.json if none exists.

    Creates the config file at $HERMES_HOME/cashew.json with all DEFAULTS,
    so subsequent is_available() checks find it. Existing files are never
    overwritten — this is a one-time bootstrap on first load.

    Returns the path written, or the existing path if file already existed.
    """
    path = resolve_config_path(hermes_home)
    if path.exists():
        return path
    return save_config(dict(DEFAULTS), hermes_home)


def save_config(
    values: dict[str, Any], hermes_home: str | os.PathLike[str]
) -> pathlib.Path:
    """Persist the provider config to $HERMES_HOME/cashew.json (CONF-02).

    Preserves unknown keys from existing files, except for the explicitly
    removed legacy settings. Unknown keys in `values` are dropped.
    Returns the path written.

    Writes are UTF-8, 2-space indent, sorted keys, trailing newline — stable
    diff-friendly format. A profile-local stable lock serializes writers; each
    writer re-reads under that lock, so non-overlapping updates are retained and
    the last completed write wins for the same key. The replacement is atomic,
    and malformed existing JSON is left untouched for correction.
    """
    home = pathlib.Path(hermes_home)
    path = resolve_config_path(home)
    known = set(DEFAULTS)
    with _exclusive_config_lock(path):
        existing = {
            key: value
            for key, value in _read_existing_config(path).items()
            if key not in REMOVED_LEGACY_CONFIG_KEYS
        }
        merged: dict[str, Any] = {**DEFAULTS, **existing}
        for key, value in values.items():
            if key in known:
                merged[key] = value
        validated = _validate_config_values(
            {key: merged[key] for key in DEFAULTS}, home
        )
        merged.update(validated)
        contents = json.dumps(merged, indent=2, sort_keys=True) + "\n"
        _atomic_write_config(path, contents)
    logger.debug("wrote cashew config to %s", path)
    return path
