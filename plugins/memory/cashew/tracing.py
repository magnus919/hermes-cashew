"""OpenTelemetry tracing instrumentation for the cashew memory provider.

Provides lightweight span creation helpers that wrap key plugin operations.
When OpenTelemetry is not installed (the common case), all functions are
graceful no-ops with zero overhead — no spans are created and no exceptions
are raised.

When a tracer provider IS configured (e.g., by a host application like
Hermes Agent exporting to an OTel collector), spans created here compose
with the host's trace context automatically.

Install with: ``pip install hermes-cashew[tracing]``

Usage in plugin code::

    from .tracing import trace_operation

    with trace_operation("cashew.query") as span:
        span.set_attribute("query.length", len(query))
        result = _do_query(query)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _tracer = otel_trace.get_tracer(__name__)
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


@contextmanager
def trace_operation(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: Any = None,
) -> Generator[Any, None, None]:
    """Create a span around a block of code.

    Args:
        name: Span name (e.g., ``"cashew.query"``).
        attributes: Optional dict of span attributes to set on creation.
        kind: Optional ``SpanKind`` value. Ignored when OTel is absent.

    Yields:
        The active span object (or a no-op proxy when OTel is absent).
        The span is ended automatically when the context exits.

    Example::

        with trace_operation("cashew.sync", {"turn.length": 120}) as span:
            _drain_once(turn)
            span.set_attribute("success", True)
    """
    if not _HAS_OTEL:
        # No-op path: yield a dummy that accepts .set_attribute() calls
        yield _NoOpSpan()
        return

    if kind is None:
        kind = SpanKind.INTERNAL

    with _tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def record_exception(span: Any, exception: Exception) -> None:
    """Record an exception on a span without marking it as an error.

    Use this when the exception is expected and handled (e.g., Cashew
    silent-degrade paths). The span remains OK but carries the exception
    event for debugging.

    Args:
        span: The span object from ``trace_operation()``.
        exception: The caught exception.
    """
    if not _HAS_OTEL:
        return
    span.record_exception(exception)


def set_error(span: Any, exception: Exception) -> None:
    """Mark a span as failed and record the exception.

    Use this when the exception represents a genuine operational failure.

    Args:
        span: The span object from ``trace_operation()``.
        exception: The caught exception.
    """
    if not _HAS_OTEL:
        return
    span.record_exception(exception)
    span.set_status(Status(StatusCode.ERROR, str(exception)))


class _NoOpSpan:
    """No-op span proxy used when OpenTelemetry is not installed.

    Accepts all the same attribute-setting calls as a real span but
    does nothing. This avoids conditional code at every call site.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, _status: Any) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass
