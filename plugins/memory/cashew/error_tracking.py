"""Opt-in, content-free Sentry diagnostics owned by each Cashew provider."""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import threading
import urllib.request
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, cast

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
        duration_ms = extra.get(
            "duration_ms", extra.get("duration_bucket_ms", duration_ms)
        )
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


def _canonical_event(event: object) -> dict[str, Any]:
    """Re-apply Cashew's envelope allowlist after SDK enrichment."""
    source = event if isinstance(event, dict) else {}
    tags = source.get("tags")
    contexts = source.get("contexts")
    exception = source.get("exception")
    operation = tags.get("operation") if isinstance(tags, dict) else None
    raw_context = contexts.get("cashew") if isinstance(contexts, dict) else None
    raw_values = exception.get("values") if isinstance(exception, dict) else None
    first = raw_values[0] if isinstance(raw_values, list) and raw_values else {}
    error_type = first.get("type") if isinstance(first, dict) else None
    return {
        "level": "error",
        "platform": "python",
        "message": "cashew operation failed",
        "tags": {"operation": _operation(operation)},
        "contexts": {
            "cashew": _safe_metadata(
                extra=raw_context if isinstance(raw_context, dict) else None
            )
        },
        "breadcrumbs": {"values": []},
        "exception": {
            "values": [
                {
                    "type": (
                        error_type
                        if isinstance(error_type, str)
                        and error_type in _SAFE_ERROR_CLASSES
                        else "Exception"
                    ),
                    "value": "redacted",
                }
            ]
        },
    }


def _final_envelope_bytes(envelope: Any) -> bytes:
    """Drop every SDK-added item/header before the only outbound transport."""
    from sentry_sdk.envelope import Envelope

    final = Envelope()
    final.add_event(cast(Any, _canonical_event(envelope.get_event())))
    return final.serialize()


def _post_final_envelope(payload: bytes, dsn: str) -> None:
    """Deliver a pre-scrubbed envelope from the isolated diagnostics worker."""
    from sentry_sdk.utils import Dsn

    auth = Dsn(dsn).to_auth("hermes-cashew")
    request = urllib.request.Request(
        auth.get_api_url(),
        data=payload,
        headers={
            "Content-Type": "application/x-sentry-envelope",
            "X-Sentry-Auth": auth.to_header(),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1):
        pass


def _sentry_worker(
    events: Any, dsn: str, record_connection: Connection | None = None
) -> None:
    """Own the high-level SDK in a child so host SDK globals never change."""
    try:
        import sentry_sdk
        from sentry_sdk.transport import Transport

        class FinalTransport(Transport):
            def capture_envelope(self, envelope: Any) -> None:
                payload = _final_envelope_bytes(envelope)
                if record_connection is not None:
                    record_connection.send(payload)
                    return
                try:
                    _post_final_envelope(payload, dsn)
                except Exception:
                    pass

        client = sentry_sdk.Client(
            dsn=dsn,
            transport=FinalTransport,
            send_default_pii=False,
            default_integrations=False,
            max_breadcrumbs=0,
            traces_sample_rate=0.0,
            send_client_reports=False,
        )
    except Exception:
        if record_connection is not None:
            record_connection.close()
        return
    try:
        while True:
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event is None:
                return
            try:
                client.capture_event(
                    cast(Any, _canonical_event(event)), scope=sentry_sdk.Scope()
                )
                client.flush(timeout=0)
            except Exception:
                pass
    finally:
        try:
            client.close(timeout=0)
        except Exception:
            pass
        if record_connection is not None:
            record_connection.close()


def _record_final_envelope_for_test(event: dict[str, Any], dsn: str) -> bytes:
    """Exercise the locked SDK's enriched envelope in an isolated child."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    events = context.Queue(maxsize=1)
    process = context.Process(target=_sentry_worker, args=(events, dsn, child))
    process.daemon = True
    process.start()
    child.close()
    try:
        events.put_nowait(_canonical_event(event))
        if not parent.poll(5):
            raise RuntimeError("Sentry recording worker did not respond")
        return cast(bytes, parent.recv())
    finally:
        try:
            events.put_nowait(None)
        except queue.Full:
            pass
        parent.close()
        process.join(timeout=0.5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.5)


@dataclass
class SentryTelemetry:
    """One provider-owned diagnostics worker, safe to close exactly once."""

    events: Any
    process: Any
    _lock: threading.Lock
    _closed: bool = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            events, process = self.events, self.process
        try:
            events.put_nowait(None)
        except queue.Full:
            pass
        except Exception:
            pass
        try:
            events.close()
        except Exception:
            pass
        try:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.2)
        except Exception:
            logger.debug("sentry: Cashew worker close failed")

    def capture(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            events, process = self.events, self.process
        if not process.is_alive():
            return
        try:
            events.put_nowait(_canonical_event(event))
        except queue.Full:
            return
        except Exception:
            logger.debug("sentry: Cashew capture unavailable")


def start_sentry_telemetry() -> SentryTelemetry | None:
    """Start an isolated client only after explicit Cashew opt-in."""
    dsn = os.environ.get(_SENTRY_DSN_ENV, "").strip()
    if not dsn:
        return None
    try:
        context = multiprocessing.get_context("spawn")
        events = context.Queue(maxsize=16)
        process = context.Process(target=_sentry_worker, args=(events, dsn))
        process.daemon = True
        process.start()
        return SentryTelemetry(events, process, threading.Lock())
    except Exception:
        logger.debug("sentry: Cashew client unavailable")
        return None


def close_sentry_telemetry(telemetry: SentryTelemetry | None) -> None:
    """Release one provider's diagnostics worker without touching host SDK state."""
    if telemetry is not None:
        telemetry.close()


def capture_exception(
    error: Exception,
    operation: str = "unknown",
    session_id: str = "",
    extra: dict[str, Any] | None = None,
    *,
    telemetry: SentryTelemetry | None = None,
    **metadata: Any,
) -> None:
    """Capture a bounded event through the calling provider's worker only."""
    del session_id
    if telemetry is None:
        return
    try:
        telemetry.capture(_event(error, operation, extra=extra, **metadata))
    except Exception:
        logger.debug("sentry: Cashew event preparation failed")
