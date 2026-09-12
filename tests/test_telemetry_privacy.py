"""Production-boundary privacy contracts for optional Cashew telemetry."""

from __future__ import annotations

import io
import json
import logging
import queue
import sys
import types
from contextlib import contextmanager

from plugins.memory.cashew import (
    CashewMemoryProvider,
    error_tracking,
    log_filter,
    tracing,
)


class _SentryClient:
    constructed: list[dict] = []
    events: list[dict] = []
    closed: list[object] = []

    def __init__(self, **kwargs):
        self.constructed.append(kwargs)

    def capture_event(self, event, scope=None):
        self.events.append({"event": event, "scope": scope})

    def close(self, timeout=None):
        self.closed.append(timeout)


class _SentryScope:
    pass


def _fake_sentry(monkeypatch):
    fake = types.SimpleNamespace(Client=_SentryClient, Scope=_SentryScope)
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake)
    _SentryClient.constructed.clear()
    _SentryClient.events.clear()
    _SentryClient.closed.clear()


def test_generic_sentry_configuration_does_nothing(monkeypatch):
    _fake_sentry(monkeypatch)
    monkeypatch.delenv("HERMES_CASHEW_SENTRY_DSN", raising=False)
    monkeypatch.setenv("SENTRY_DSN", "https://host.example/123")

    assert error_tracking.start_sentry_telemetry() is None
    error_tracking.capture_exception(
        RuntimeError("CONTENT-CANARY"), operation="cashew.query"
    )
    assert _SentryClient.constructed == []
    assert _SentryClient.events == []


def test_explicit_sentry_event_is_allowlisted_and_per_provider(monkeypatch):
    _fake_sentry(monkeypatch)
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://cashew.example/123")
    first = error_tracking.start_sentry_telemetry()
    second = error_tracking.start_sentry_telemetry()
    assert first is not None and second is not None and first is not second

    error_tracking.capture_exception(
        RuntimeError(
            "CONTENT-CANARY path=/private/profile/brain.db api_key=SECRET-CANARY"
        ),
        operation="cashew.query",
        session_id="SESSION-CANARY",
        extra={
            "query": "QUERY-CANARY",
            "db_path": "/private/profile/brain.db",
            "generation": 4,
            "state": "degraded",
            "fallback": "keyword",
            "duration_ms": 12.5,
        },
        telemetry=first,
    )

    payload = json.dumps([item["event"] for item in _SentryClient.events])
    for canary in (
        "CONTENT-CANARY",
        "SECRET-CANARY",
        "SESSION-CANARY",
        "QUERY-CANARY",
        "/private/profile",
    ):
        assert canary not in payload
    assert _SentryClient.constructed[0] == {
        "dsn": "https://cashew.example/123",
        "send_default_pii": False,
        "default_integrations": False,
        "max_breadcrumbs": 0,
        "traces_sample_rate": 0.0,
    }
    event = _SentryClient.events[0]["event"]
    assert event["tags"] == {"operation": "cashew.query"}
    assert event["contexts"]["cashew"] == {
        "generation": 4,
        "state": "degraded",
        "fallback": "keyword",
        "duration_bucket_ms": 100,
    }
    assert event["breadcrumbs"] == {"values": []}
    assert event["exception"]["values"] == [
        {"type": "RuntimeError", "value": "redacted"}
    ]

    error_tracking.close_sentry_telemetry(first)
    error_tracking.close_sentry_telemetry(first)
    assert _SentryClient.closed == [0]
    error_tracking.capture_exception(RuntimeError("CONTENT-CANARY"), telemetry=first)
    assert len(_SentryClient.events) == 1
    error_tracking.close_sentry_telemetry(second)
    assert _SentryClient.closed == [0, 0]


def test_sentry_sdk_failure_is_a_noop(monkeypatch):
    class BrokenSentry:
        class Client:
            def __init__(self, **kwargs):
                raise RuntimeError("SECRET-CANARY")

    monkeypatch.setitem(sys.modules, "sentry_sdk", BrokenSentry)
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://cashew.example/123")
    assert error_tracking.start_sentry_telemetry() is None


def test_provider_runtime_cleanup_closes_only_its_telemetry(monkeypatch):
    _fake_sentry(monkeypatch)
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://cashew.example/123")
    first = error_tracking.start_sentry_telemetry()
    second = error_tracking.start_sentry_telemetry()
    assert first is not None and second is not None
    provider = CashewMemoryProvider()
    provider._sync_queue = queue.Queue()
    provider._sentry_telemetry = first
    monkeypatch.setattr(
        provider, "_close_embedding_runtime", lambda *args, **kwargs: None
    )

    provider._clear_runtime_state(None, provider._sync_queue)

    assert _SentryClient.closed == [0]
    assert provider._sentry_telemetry is None
    error_tracking.capture_exception(RuntimeError("CONTENT-CANARY"), telemetry=second)
    assert len(_SentryClient.events) == 1
    error_tracking.close_sentry_telemetry(second)


def test_provider_shutdown_and_reinitialize_own_distinct_sentry_clients(
    tmp_path, monkeypatch
):
    _fake_sentry(monkeypatch)
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://cashew.example/123")
    provider = CashewMemoryProvider()

    provider.initialize("SESSION-CANARY", hermes_home=str(tmp_path))
    first = provider._sentry_telemetry
    assert first is not None
    provider.shutdown()
    assert provider._sentry_telemetry is None
    assert _SentryClient.closed == [0]

    provider.initialize("SESSION-CANARY-SECOND", hermes_home=str(tmp_path))
    second = provider._sentry_telemetry
    try:
        assert second is not None and second is not first
    finally:
        provider.shutdown()
    assert _SentryClient.closed == [0, 0]


def test_late_child_log_record_is_sanitized_at_actual_emission():
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG)
    try:
        parent = logging.getLogger("plugins.memory.cashew.telemetry_boundary")
        parent.handlers.clear()
        parent.propagate = True
        log_filter.add_scrub_filter(parent)
        child = logging.getLogger("plugins.memory.cashew.telemetry_boundary.late_child")
        child.error(
            "CONTENT-CANARY path=/private/profile/brain.db session=SESSION-CANARY api_key=SECRET-CANARY",
            exc_info=RuntimeError("CONTENT-CANARY path=/private/profile/brain.db"),
        )
        emitted = stream.getvalue()
    finally:
        parent.handlers.clear()
        parent.propagate = True
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
    for canary in (
        "CONTENT-CANARY",
        "SECRET-CANARY",
        "SESSION-CANARY",
        "/private/profile",
    ):
        assert canary not in emitted
    assert "cashew local log event error=RuntimeError" in emitted
    assert "cashew error RuntimeError" in emitted


def test_otel_proxy_blocks_raw_content_and_disables_automatic_exception_recording(
    monkeypatch,
):
    class Span:
        def __init__(self):
            self.attributes = {}
            self.events = []
            self.status = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def add_event(self, name, attributes):
            self.events.append((name, attributes))

        def set_status(self, status):
            self.status = status

    class Tracer:
        def start_as_current_span(self, name, **kwargs):
            self.name, self.kwargs = name, kwargs
            span = Span()

            @contextmanager
            def context():
                self.span = span
                yield span

            return context()

    tracer = Tracer()
    monkeypatch.setenv("HERMES_CASHEW_OTEL_ENABLED", "1")
    monkeypatch.setattr(tracing, "_OTEL", tracer)

    with tracing.trace_operation(
        "cashew.query",
        {"input_length": 50, "query": "CONTENT-CANARY", "generation": 2},
    ) as span:
        span.set_attribute("session_id", "SESSION-CANARY")
        span.add_event("raw", {"credential": "SECRET-CANARY"})
        tracing.set_error(span, RuntimeError("CONTENT-CANARY /private/profile"))

    assert tracer.name == "cashew.query"
    assert tracer.kwargs["record_exception"] is False
    assert tracer.kwargs["set_status_on_exception"] is False
    assert tracer.span.attributes == {"input_length": 50, "generation": 2}
    payload = json.dumps(
        {"events": tracer.span.events, "status": str(tracer.span.status)}
    )
    for canary in (
        "CONTENT-CANARY",
        "SECRET-CANARY",
        "SESSION-CANARY",
        "/private/profile",
    ):
        assert canary not in payload
    assert tracer.span.events == [
        (
            "exception",
            {
                "exception.type": "RuntimeError",
                "exception.message": "redacted",
            },
        )
    ]


def test_otel_malfunction_and_escaped_operation_error_preserve_functional_path(
    monkeypatch,
):
    class BrokenTracer:
        def start_as_current_span(self, *args, **kwargs):
            raise RuntimeError("CONTENT-CANARY")

    monkeypatch.setenv("HERMES_CASHEW_OTEL_ENABLED", "1")
    monkeypatch.setattr(tracing, "_OTEL", BrokenTracer())
    with tracing.trace_operation("cashew.query") as span:
        span.set_attribute("input_length", 1)

    class ExitFailingContext:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            raise RuntimeError("SECRET-CANARY")

    class ExitFailingTracer:
        def start_as_current_span(self, *args, **kwargs):
            return ExitFailingContext()

    monkeypatch.setattr(tracing, "_OTEL", ExitFailingTracer())
    try:
        with tracing.trace_operation("cashew.query"):
            raise ValueError("CONTENT-CANARY")
    except ValueError as error:
        assert str(error) == "CONTENT-CANARY"
    else:
        raise AssertionError("operation exception was swallowed")


def test_generic_otel_configuration_does_nothing(monkeypatch):
    monkeypatch.delenv("HERMES_CASHEW_OTEL_ENABLED", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://host.example")
    monkeypatch.setattr(tracing, "_OTEL", object())
    with tracing.trace_operation("cashew.query", {"generation": 1}) as span:
        span.set_attribute("generation", 1)
