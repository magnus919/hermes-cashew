"""Opt-in, Cashew-owned Sentry diagnostics with bounded payloads."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)
_SENTRY_DSN_ENV = "HERMES_CASHEW_SENTRY_DSN"
_SAFE_OPERATIONS = frozenset(
    {
        "cashew.initialize",
        "cashew.sync",
        "cashew.query",
        "cashew.extract",
        "cashew.prefetch",
        "cashew.sleep",
        "cashew.verify",
    }
)
_SAFE_STATES = frozenset(
    {"initializing", "ready", "degraded", "stopping", "stopped", "failed"}
)
_SAFE_FALLBACKS = frozenset({"none", "keyword", "bfs", "cpu", "disabled"})
_SAFE_ERROR_CLASSES = frozenset(
    {
        "Exception",
        "ImportError",
        "KeyError",
        "OSError",
        "OperationalError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)


def safe_error_class(error: BaseException) -> str:
    """Return a stable exception category without exposing exception text."""
    name = type(error).__name__
    return name if name in _SAFE_ERROR_CLASSES else "Exception"


def _operation(value: object) -> str:
    return (
        value
        if isinstance(value, str) and value in _SAFE_OPERATIONS
        else "cashew.unknown"
    )


def _bounded_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(0, min(value, 1_000_000_000))


def _duration_bucket(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    milliseconds = max(0.0, min(float(value), 86_400_000.0))
    for ceiling in (1, 10, 100, 1_000, 10_000, 60_000, 600_000, 3_600_000, 86_400_000):
        if milliseconds <= ceiling:
            return ceiling
    return 86_400_000


def _safe_metadata(
    *,
    generation: object = None,
    state: object = None,
    fallback: object = None,
    duration_ms: object = None,
    count: object = None,
    queue_depth: object = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, int | str]:
    """Extract only values approved for external diagnostics."""
    if extra:
        generation = extra.get("generation", generation)
        state = extra.get("state", state)
        fallback = extra.get("fallback", fallback)
        duration_ms = extra.get("duration_ms", duration_ms)
        count = extra.get("count", count)
        queue_depth = extra.get("queue_depth", queue_depth)
    result: dict[str, int | str] = {}
    for key, value in (
        ("generation", generation),
        ("count", count),
        ("queue_depth", queue_depth),
    ):
        bounded = _bounded_int(value)
        if bounded is not None:
            result[key] = bounded
    if isinstance(state, str) and state in _SAFE_STATES:
        result["state"] = state
    if isinstance(fallback, str) and fallback in _SAFE_FALLBACKS:
        result["fallback"] = fallback
    bucket = _duration_bucket(duration_ms)
    if bucket is not None:
        result["duration_bucket_ms"] = bucket
    return result


@dataclass
class SentryTelemetry:
    """One provider-owned Sentry client, safe to close exactly once."""

    client: Any
    _lock: threading.Lock
    _closed: bool = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client = self.client
        try:
            client.close(timeout=0)
        except Exception:
            logger.debug("sentry: Cashew client close failed")

    def capture(self, event: dict[str, Any], scope: Any) -> None:
        with self._lock:
            if self._closed:
                return
            client = self.client
        try:
            client.capture_event(event, scope=scope)
        except Exception:
            logger.debug("sentry: Cashew capture failed")


def start_sentry_telemetry() -> SentryTelemetry | None:
    """Create an isolated client only after explicit Cashew opt-in."""
    dsn = os.environ.get(_SENTRY_DSN_ENV, "").strip()
    if not dsn:
        return None
    try:
        import sentry_sdk

        return SentryTelemetry(
            client=sentry_sdk.Client(
                dsn=dsn,
                send_default_pii=False,
                default_integrations=False,
                max_breadcrumbs=0,
                traces_sample_rate=0.0,
            ),
            _lock=threading.Lock(),
        )
    except Exception:
        # An optional diagnostics backend must never change provider behavior.
        logger.debug("sentry: Cashew client unavailable")
        return None


def close_sentry_telemetry(telemetry: SentryTelemetry | None) -> None:
    """Release one provider's isolated client without touching host SDK state."""
    if telemetry is not None:
        telemetry.close()


def _event(error: BaseException, operation: object, **metadata: Any) -> dict[str, Any]:
    return {
        "level": "error",
        "platform": "python",
        "message": "cashew operation failed",
        "tags": {"operation": _operation(operation)},
        "contexts": {"cashew": _safe_metadata(**metadata)},
        "breadcrumbs": {"values": []},
        "exception": {
            "values": [
                {
                    "type": safe_error_class(error),
                    "value": "redacted",
                }
            ]
        },
    }


def capture_exception(
    error: Exception,
    operation: str = "unknown",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
    *,
    telemetry: SentryTelemetry | None = None,
    **metadata: Any,
) -> None:
    """Capture a bounded event through the calling provider's client only."""
    del session_id
    if telemetry is None:
        return
    try:
        import sentry_sdk

        telemetry.capture(
            _event(error, operation, extra=extra, **metadata), sentry_sdk.Scope()
        )
    except Exception:
        logger.debug("sentry: Cashew event preparation failed")
