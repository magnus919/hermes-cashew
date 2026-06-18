"""Optional Sentry error tracking for the cashew memory provider.

Initializes Sentry when ``SENTRY_DSN`` is present in the environment.
Without it, all functions are graceful no-ops with zero overhead.

Captures plugin-level exceptions with session context (session_id,
operation name, config snapshot) so operators can trace errors back
to specific sessions and code paths.

Install with: ``pip install hermes-cashew[tracing]``

Setup::

    export SENTRY_DSN="https://..."
    # Optional:
    export SENTRY_ENVIRONMENT="production"
    export SENTRY_RELEASE="hermes-cashew@0.10.2"
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    import sentry_sdk
    from sentry_sdk import set_context, set_tag

    _HAS_SENTRY = True
except ImportError:
    _HAS_SENTRY = False


def _init_sentry() -> None:
    """One-time Sentry initialization. Safe to call multiple times (idempotent)."""
    if not _HAS_SENTRY:
        return
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return
    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
            release=os.environ.get("SENTRY_RELEASE"),
            traces_sample_rate=0.0,  # tracing handled by OTel, not Sentry
            send_default_pii=False,
        )
        logger.info(
            "sentry: initialized (environment=%s)",
            os.environ.get("SENTRY_ENVIRONMENT", "development"),
        )
    except Exception:
        logger.debug("sentry: init failed", exc_info=True)


def capture_exception(
    error: Exception,
    operation: str = "unknown",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Capture a handled exception with operational context.

    Args:
        error: The caught exception.
        operation: Label for the failing operation (e.g., ``"cashew.sync"``).
        session_id: Hermes session ID for correlating errors to user sessions.
        extra: Optional dict of additional context (config keys, DB path, etc.).
    """
    if not _HAS_SENTRY:
        return
    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("operation", operation)
            if session_id:
                scope.set_tag("session_id", session_id)
            if extra:
                for key, value in extra.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_exception(error)
    except Exception:
        logger.debug("sentry: capture_exception failed", exc_info=True)


def set_plugin_context(
    session_id: str = "",
    config: dict[str, Any] | None = None,
) -> None:
    """Set global context for all subsequent error captures.

    Call this after initialize() succeeds so every captured error
    carries the session and configuration context.

    Args:
        session_id: Current Hermes session ID.
        config: Snapshot of relevant config keys (sanitized — no secrets).
    """
    if not _HAS_SENTRY:
        return
    try:
        if session_id:
            set_tag("session_id", session_id)
        if config:
            set_context("cashew_config", config)
    except Exception:
        logger.debug("sentry: set_plugin_context failed", exc_info=True)


# Auto-init at import time if SENTRY_DSN is set.
_init_sentry()
