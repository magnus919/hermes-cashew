"""Pure config + path-resolution helpers for the Cashew memory provider.

This module has zero coupling to the Hermes ABC, the CashewMemoryProvider class,
or the Cashew runtime. It owns:
  - the four-key config schema (CONF-01)
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
    "cashew_db_path": "cashew/brain.db",
    "embedding_model": "all-MiniLM-L6-v2",
    "recall_k": 5,
    "sync_queue_timeout": 30.0,
}
"""Phase 2 defaults. Every key MUST match a CashewConfig field name.
Defaults match PHASE_DESIGN_NOTES.md decision-point verdicts; revise via
PROJECT.md Key Decisions before changing."""


@dataclasses.dataclass(frozen=True)
class CashewConfig:
    """Typed view over the four-key Cashew config dict."""
    cashew_db_path: str = DEFAULTS["cashew_db_path"]
    embedding_model: str = DEFAULTS["embedding_model"]
    recall_k: int = DEFAULTS["recall_k"]
    sync_queue_timeout: float = DEFAULTS["sync_queue_timeout"]


def get_config_schema() -> dict[str, Any]:
    """Return the JSON-Schema-shaped dict Hermes uses to drive `hermes memory setup`.

    Shape mirrors what bundled providers (Honcho, Hindsight) emit. Phase 2 has
    NO required-from-user fields — every key has a default — so `required: []`.
    """
    return {
        "type": "object",
        "properties": {
            "cashew_db_path": {
                "type": "string",
                "description": (
                    "Path to the Cashew SQLite brain DB, relative to hermes_home. "
                    "Absolute paths are rejected to preserve profile isolation."
                ),
                "default": DEFAULTS["cashew_db_path"],
            },
            "embedding_model": {
                "type": "string",
                "description": (
                    "Sentence-transformers model identifier Cashew loads on first use. "
                    "The default matches Cashew's documented default."
                ),
                "default": DEFAULTS["embedding_model"],
            },
            "recall_k": {
                "type": "integer",
                "minimum": 1,
                "description": "How many context fragments prefetch() requests from Cashew per turn.",
                "default": DEFAULTS["recall_k"],
            },
            "sync_queue_timeout": {
                "type": "number",
                "minimum": 0,
                "description": (
                    "Bounded join timeout (seconds) shutdown() applies when draining the sync queue. "
                    "Worker thread is added in Phase 4; the value is wired through now."
                ),
                "default": DEFAULTS["sync_queue_timeout"],
            },
        },
        "required": [],
        "additionalProperties": False,
    }


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

    Cashew failures (corrupt JSON, permission errors) are NOT swallowed here —
    the caller (CashewMemoryProvider.initialize in Plan 02-02) decides whether
    to silent-degrade or surface; this helper is honest about I/O.
    """
    path = resolve_config_path(hermes_home)
    if not path.exists():
        return CashewConfig(**DEFAULTS)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(raw).__name__}")
    merged: dict[str, Any] = {**DEFAULTS, **raw}
    # Drop any unknown keys so future schema additions don't break dataclass construction.
    known = {f.name for f in dataclasses.fields(CashewConfig)}
    filtered = {k: v for k, v in merged.items() if k in known}
    return CashewConfig(**filtered)


def save_config(values: dict[str, Any], hermes_home: str | os.PathLike[str]) -> pathlib.Path:
    """Persist the provider config to $HERMES_HOME/cashew.json (CONF-02).

    `values` is merged over DEFAULTS so partial dicts from `hermes memory setup`
    still produce a complete file. Keys outside the schema are dropped. Returns
    the path written.

    Writes are UTF-8, 2-space indent, sorted keys, trailing newline — stable
    diff-friendly format. Parent directory is created with `parents=True,
    exist_ok=True`.
    """
    path = resolve_config_path(hermes_home)
    known = {f.name for f in dataclasses.fields(CashewConfig)}
    merged: dict[str, Any] = {**DEFAULTS, **{k: v for k, v in values.items() if k in known}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.debug("wrote cashew config to %s", path)
    return path
