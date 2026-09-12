"""Production-boundary privacy contracts for optional Cashew telemetry."""

from __future__ import annotations

import io
import json
import logging
import queue
import sys
from contextlib import contextmanager

from plugins.memory.cashew import (
    CashewMemoryProvider,
    error_tracking,
    log_filter,
    tracing,
)


def _decode_event(payload: bytes) -> dict:
    from sentry_sdk.envelope import Envelope

    envelope = Envelope.deserialize(payload)
    assert len(envelope.items) == 1
    assert envelope.items[0].type == "event"
    event = envelope.get_event()
    assert event is not None
    return event


def _sentry_host_state() -> tuple[object, object, object, dict[str, object]]:
    import sentry_sdk

    hub = sentry_sdk.Hub.current
    return (
        hub,
        hub.client,
        sentry_sdk.get_current_scope(),
        dict(sentry_sdk.__dict__),
    )


def test_generic_sentry_configuration_does_nothing(monkeypatch):
    monkeypatch.delenv("HERMES_CASHEW_SENTRY_DSN", raising=False)
    monkeypatch.setenv("SENTRY_DSN", "https://host.example/123")

    assert error_tracking.start_sentry_telemetry() is None


def test_locked_sentry_final_envelope_is_allowlisted_and_host_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://public@example.invalid/123")
    before = _sentry_host_state()
    event = error_tracking._event(
        RuntimeError("CONTENT-CANARY path=/private/profile api_key=SECRET-CANARY"),
        "cashew.query",
        extra={
            "query": "QUERY-CANARY",
            "session_id": "SESSION-CANARY",
            "db_path": "/private/profile/brain.db",
            "generation": 4,
            "state": "degraded",
            "fallback": "keyword",
            "duration_ms": 12.5,
        },
    )
    payload = error_tracking._record_final_envelope_for_test(
        event, "https://public@example.invalid/123"
    )
    after = _sentry_host_state()

    assert before[:3] == after[:3]
    assert before[3] == after[3]
    assert "sentry_sdk" in sys.modules  # The test imported it; production did not.
    decoded = _decode_event(payload)
    assert set(decoded) == {
        "level",
        "platform",
        "message",
        "tags",
        "contexts",
        "breadcrumbs",
        "exception",
    }
    serialized = json.dumps(decoded, sort_keys=True)
    for canary in (
        "CONTENT-CANARY",
        "SECRET-CANARY",
        "SESSION-CANARY",
        "QUERY-CANARY",
        "/private/profile",
    ):
        assert canary not in serialized
    assert decoded["tags"] == {"operation": "cashew.query"}
    assert decoded["contexts"] == {
        "cashew": {
            "generation": 4,
            "state": "degraded",
            "fallback": "keyword",
            "duration_bucket_ms": 100,
        }
    }
    assert decoded["breadcrumbs"] == {"values": []}
    assert decoded["exception"] == {
        "values": [{"type": "RuntimeError", "value": "redacted"}]
    }


def test_sentry_worker_is_bounded_and_provider_owned(monkeypatch):
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://public@example.invalid/123")
    before_modules = set(sys.modules)
    first = error_tracking.start_sentry_telemetry()
    second = error_tracking.start_sentry_telemetry()
    assert first is not None and second is not None and first is not second
    for _ in range(64):
        error_tracking.capture_exception(
            RuntimeError("CONTENT-CANARY"), telemetry=first
        )
    error_tracking.close_sentry_telemetry(first)
    error_tracking.close_sentry_telemetry(first)
    error_tracking.close_sentry_telemetry(second)
    assert "sentry_sdk" not in set(sys.modules) - before_modules
    assert not first.process.is_alive()
    assert not second.process.is_alive()


def test_provider_runtime_cleanup_closes_only_its_telemetry(monkeypatch):
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://public@example.invalid/123")
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

    assert not first.process.is_alive()
    assert provider._sentry_telemetry is None
    assert second.process.is_alive()
    error_tracking.close_sentry_telemetry(second)


def test_provider_shutdown_and_reinitialize_own_distinct_sentry_workers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_CASHEW_SENTRY_DSN", "https://public@example.invalid/123")
    provider = CashewMemoryProvider()

    provider.initialize("SESSION-CANARY", hermes_home=str(tmp_path))
    first = provider._sentry_telemetry
    assert first is not None
    provider.shutdown()
    assert provider._sentry_telemetry is None
    assert not first.process.is_alive()

    provider.initialize("SESSION-CANARY-SECOND", hermes_home=str(tmp_path))
    second = provider._sentry_telemetry
    try:
        assert second is not None and second is not first
    finally:
        provider.shutdown()
    assert not second.process.is_alive()


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


def test_pinned_upstream_root_emissions_and_late_handlers_are_sanitized(
    tmp_path, monkeypatch
):
    """Pinned core uses root logging; source-path boundary handles late handlers."""
    import sqlite3

    from core import embeddings, session

    root = logging.getLogger()
    core_logger = logging.getLogger("core")
    original_root, original_core = root.handlers[:], core_logger.handlers[:]
    original_levels = root.level, core_logger.level
    root_stream, core_stream, host_stream = io.StringIO(), io.StringIO(), io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    core_handler = logging.StreamHandler(core_stream)
    host_handler = logging.StreamHandler(host_stream)
    try:
        log_filter.acquire_provider_scrub_filters()
        # Added after Cashew's boundary: both must receive the already-scrubbed
        # shared record. The unrelated host logger remains byte-identical.
        root.addHandler(root_handler)
        core_logger.addHandler(core_handler)
        host = logging.getLogger("host.unrelated")
        host.addHandler(host_handler)
        root.setLevel(logging.DEBUG)
        core_logger.setLevel(logging.DEBUG)
        host.warning("HOST-CANARY %s", "unchanged")

        db_path = tmp_path / "graph.db"
        session.end_session(str(db_path), "SESSION-CANARY", "too short")
        connection = sqlite3.connect(db_path)
        monkeypatch.setattr(embeddings, "_vec_available", True)
        monkeypatch.setattr(
            embeddings.sqlite_vec,
            "load",
            lambda _connection: (_ for _ in ()).throw(
                RuntimeError("CONTENT-CANARY /private/profile SECRET-CANARY")
            ),
        )
        embeddings._load_vec(connection)
        connection.close()
        # A late direct ``core`` handler sees the same source-scoped boundary.
        core_logger.handle(
            core_logger.makeRecord(
                "core",
                logging.WARNING,
                session.__file__,
                1,
                "CONTENT-CANARY core handler",
                (),
                (RuntimeError, RuntimeError("SECRET-CANARY /private/profile"), None),
            )
        )
    finally:
        host.removeHandler(host_handler)
        root.handlers[:] = original_root
        core_logger.handlers[:] = original_core
        root.setLevel(original_levels[0])
        core_logger.setLevel(original_levels[1])
        log_filter.release_provider_scrub_filters()

    emitted = root_stream.getvalue() + core_stream.getvalue()
    for canary in (
        "SESSION-CANARY",
        "CONTENT-CANARY",
        "SECRET-CANARY",
        "/private/profile",
    ):
        assert canary not in emitted
    assert emitted.count("cashew local log event") >= 2
    assert host_stream.getvalue() == "HOST-CANARY unchanged\n"


def test_log_scrub_boundaries_are_reference_counted_and_reversible():
    root = logging.getLogger()
    core_logger = logging.getLogger("core")
    plugin_logger = logging.getLogger("plugins.memory.cashew")
    before = {
        current: (current.handlers[:], current.propagate)
        for current in (root, core_logger, plugin_logger)
    }
    log_filter.acquire_provider_scrub_filters()
    log_filter.acquire_provider_scrub_filters()
    try:
        for current in before:
            assert any(
                getattr(handler, "_cashew_sanitizing_handler", False)
                for handler in current.handlers
            )
        log_filter.release_provider_scrub_filters()
        assert any(
            getattr(handler, "_cashew_sanitizing_handler", False)
            for handler in root.handlers
        )
    finally:
        log_filter.release_provider_scrub_filters()
    for current, (handlers, propagate) in before.items():
        assert current.handlers == handlers
        assert current.propagate is propagate


def test_otel_raised_provider_operation_records_safe_error_before_exit(monkeypatch):
    class Span:
        def __init__(self):
            self.events = []
            self.status = None

        def set_attribute(self, key, value):
            del key, value

        def add_event(self, name, attributes):
            self.events.append((name, attributes))

        def set_status(self, status):
            self.status = status

    class Context:
        def __init__(self, span):
            self.span = span
            self.exit_args = None

        def __enter__(self):
            return self.span

        def __exit__(self, *args):
            self.exit_args = args

    class Tracer:
        def start_as_current_span(self, *_args, **_kwargs):
            self.span = Span()
            self.context = Context(self.span)
            return self.context

    tracer = Tracer()
    monkeypatch.setenv("HERMES_CASHEW_OTEL_ENABLED", "1")
    monkeypatch.setattr(tracing, "_OTEL", tracer)
    try:
        with tracing.trace_operation("cashew.sync"):
            raise RuntimeError("CONTENT-CANARY /private/profile")
    except RuntimeError as error:
        assert str(error) == "CONTENT-CANARY /private/profile"
    else:
        raise AssertionError("functional exception was swallowed")
    assert tracer.context.exit_args[0] is RuntimeError
    assert tracer.span.events == [
        (
            "exception",
            {"exception.type": "RuntimeError", "exception.message": "redacted"},
        )
    ]
    assert tracer.span.status.status_code.name == "ERROR"
    assert tracer.span.status.description == "cashew operation failed"
