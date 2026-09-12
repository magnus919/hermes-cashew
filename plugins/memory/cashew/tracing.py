"""Opt-in, content-free OpenTelemetry spans for Cashew operations."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Any, Generator

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
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "generation",
        "state",
        "fallback",
        "duration_bucket_ms",
        "count",
        "queue_depth",
        "input_length",
        "result_count",
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
_OTEL: Any = None


def _enabled() -> bool:
    return os.environ.get("HERMES_CASHEW_OTEL_ENABLED") == "1"


def _tracer_for_call() -> Any:
    global _OTEL
    if not _enabled():
        return None
    if _OTEL is None:
        try:
            from opentelemetry import trace

            _OTEL = trace.get_tracer("hermes-cashew")
        except Exception:
            _OTEL = False
    return None if _OTEL is False else _OTEL


def _duration_bucket(value: object) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return None
    milliseconds = max(0.0, min(float(value), 86_400_000.0))
    for ceiling in (1, 10, 100, 1_000, 10_000, 60_000, 600_000, 3_600_000, 86_400_000):
        if milliseconds <= ceiling:
            return ceiling
    return 86_400_000


def _safe_value(key: str, value: Any) -> int | str | None:
    if key in {"state", "fallback"}:
        allowed = _SAFE_STATES if key == "state" else _SAFE_FALLBACKS
        return value if isinstance(value, str) and value in allowed else None
    if key in {"generation", "count", "queue_depth", "input_length", "result_count"}:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return max(0, min(value, 1_000_000_000))
    if key == "duration_bucket_ms":
        return _duration_bucket(value)
    return None


def _safe_error_class(error: BaseException) -> str:
    name = type(error).__name__
    return name if name in _SAFE_ERROR_CLASSES else "Exception"


class _SafeSpan:
    """Prevent callers from sending attributes or exception details by accident."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        safe = _safe_value(key, value)
        if safe is None:
            return
        try:
            self._span.set_attribute(key, safe)
        except Exception:
            pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        del name, attributes

    def record_exception(self, exception: Exception) -> None:
        self._record_exception(exception)

    def _record_exception(self, exception: BaseException) -> None:
        try:
            self._span.add_event(
                "exception",
                {
                    "exception.type": _safe_error_class(exception),
                    "exception.message": "redacted",
                },
            )
        except Exception:
            pass

    def set_status(self, status: Any) -> None:
        del status

    def _set_error(self, exception: BaseException) -> None:
        self._record_exception(exception)
        try:
            from opentelemetry.trace import Status, StatusCode

            self._span.set_status(Status(StatusCode.ERROR, "cashew operation failed"))
        except Exception:
            pass


@contextmanager
def trace_operation(
    name: str, attributes: dict[str, Any] | None = None, kind: Any = None
) -> Generator[Any, None, None]:
    """Yield a safe proxy and never let optional telemetry affect Cashew."""
    tracer = _tracer_for_call()
    if tracer is None:
        yield _NoOpSpan()
        return
    operation = name if name in _SAFE_OPERATIONS else "cashew.unknown"
    try:
        if kind is None:
            try:
                from opentelemetry.trace import SpanKind

                kind = SpanKind.INTERNAL
            except ImportError:
                kind = None
        kwargs = {
            "record_exception": False,
            "set_status_on_exception": False,
        }
        if kind is not None:
            kwargs["kind"] = kind
        context = tracer.start_as_current_span(operation, **kwargs)
        raw_span = context.__enter__()
    except Exception:
        yield _NoOpSpan()
        return
    span = _SafeSpan(raw_span)
    for key, value in (attributes or {}).items():
        span.set_attribute(key, value)
    try:
        yield span
    except BaseException as error:
        # Automatic exception recording is disabled at span creation, so ensure
        # the safe proxy records the fixed event before context teardown.
        span._set_error(error)
        try:
            context.__exit__(type(error), error, error.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception:
            pass


def record_exception(span: Any, exception: Exception) -> None:
    if isinstance(span, _SafeSpan):
        span._record_exception(exception)


def set_error(span: Any, exception: Exception) -> None:
    if isinstance(span, _SafeSpan):
        span._set_error(exception)


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def set_status(self, status: Any) -> None:
        del status

    def record_exception(self, exception: Exception) -> None:
        del exception

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        del name, attributes
