"""Content-free local logging for the Cashew provider hierarchy."""

from __future__ import annotations

import logging
import re
import threading
import weakref
from pathlib import Path

_SAFE_EXCEPTION_CLASSES = frozenset(
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
_PACKAGE_DIR = Path(__file__).resolve().parent
_SAFE_ENV_NAME = re.compile(r"CASHEW_[A-Z0-9_]+$")
_SAFE_EMBED_OPERATIONS = frozenset({"start", "encode", "close"})
_SAFE_EMBED_REASONS = frozenset(
    {
        "busy",
        "closed",
        "exited",
        "owned",
        "protocol",
        "startup",
        "timeout",
    }
)
_SAFE_IDENTITY_STATES = frozenset({"could not be verified", "inspection deferred"})


def _is_owned_source(record: logging.LogRecord) -> bool:
    """Only preserve templates emitted by a checked-in Cashew source module."""
    try:
        return Path(record.pathname).resolve().is_relative_to(_PACKAGE_DIR)
    except (OSError, ValueError):
        return False


def _safe_argument(value: object) -> object:
    """Keep numeric counters and Cashew setting names; redact every other arg."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and _SAFE_ENV_NAME.fullmatch(value):
        return value
    return "<redacted>"


def _safe_arguments(args: object) -> object:
    if isinstance(args, dict):
        return {key: _safe_argument(value) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(_safe_argument(value) for value in args)
    return ()


def _safe_embedding_failure_arguments(args: object) -> object:
    if not isinstance(args, tuple) or len(args) != 10:
        return _safe_arguments(args)
    (
        operation,
        generation,
        _model,
        _requested,
        _effective,
        _request,
        count,
        reason,
        exit_code,
        elapsed,
    ) = args
    return (
        operation if operation in _SAFE_EMBED_OPERATIONS else "<redacted>",
        _safe_argument(generation),
        "<redacted>",
        "<redacted>",
        "<redacted>",
        "<redacted>",
        _safe_argument(count),
        reason if reason in _SAFE_EMBED_REASONS else "<redacted>",
        _safe_argument(exit_code),
        _safe_argument(elapsed),
    )


def _safe_identity_lock_arguments(args: object) -> object:
    if not isinstance(args, tuple) or len(args) != 2:
        return _safe_arguments(args)
    state, _lock_path = args
    return (state if state in _SAFE_IDENTITY_STATES else "<redacted>", "<redacted>")


def _safe_error_class(error: BaseException | None) -> str:
    if error is None:
        return "Exception"
    name = type(error).__name__
    return name if name in _SAFE_EXCEPTION_CLASSES else "Exception"


class ScrubFilter(logging.Filter):
    """Replace raw Cashew log content and exception details with safe codes."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_cashew_scrubbed", False):
            return True
        error = record.exc_info[1] if record.exc_info else None
        suffix = f" error={_safe_error_class(error)}" if error is not None else ""
        # Checked-in provider modules use static operation templates. Keep only
        # that source-owned template (never its arguments); externally emitted
        # child records are reduced to a generic code because their provenance
        # is not a privacy boundary the adapter can prove.
        template = record.msg if isinstance(record.msg, str) else ""
        if _is_owned_source(record):
            record.msg = template + suffix
            if template.startswith("embedding worker failure:"):
                record.args = _safe_embedding_failure_arguments(record.args)
            elif template.startswith("embedding identity %s; unable to acquire"):
                record.args = _safe_identity_lock_arguments(record.args)
            else:
                record.args = _safe_arguments(record.args)
        else:
            record.msg = f"cashew local log event{suffix}"
            record.args = None
        if error is not None:
            record.exc_info = (
                Exception,
                Exception(f"cashew error {_safe_error_class(error)}"),
                None,
            )
        record.exc_text = None
        record.stack_info = None
        record._cashew_scrubbed = True
        return True


class _ForwardToInheritedHandlers(logging.Handler):
    """Sanitize once at the Cashew boundary before ordinary host propagation."""

    _cashew_sanitizing_handler = True

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__(level=logging.NOTSET)
        self._logger_ref = weakref.ref(logger)
        self._emit_lock = threading.RLock()
        self.addFilter(ScrubFilter())

    def emit(self, record: logging.LogRecord) -> None:
        logger = self._logger_ref()
        if logger is None:
            return
        # ``Logger`` filters do not run for propagated children. Reproduce the
        # normal ancestor-handler walk after this owned handler scrubs the one
        # shared record, without modifying a host logger or its configuration.
        with self._emit_lock:
            ancestor = logger.parent
            found = False
            while ancestor is not None:
                for handler in ancestor.handlers:
                    found = True
                    if record.levelno >= handler.level:
                        handler.handle(record)
                if not ancestor.propagate:
                    break
                ancestor = ancestor.parent
            if (
                not found
                and logging.lastResort is not None
                and record.levelno >= logging.lastResort.level
            ):
                logging.lastResort.handle(record)


def add_scrub_filter(logger: logging.Logger) -> None:
    """Route all Cashew descendant records through one owned sanitizer.

    The parent logger is the hierarchy boundary. Descendants can be created
    after registration; their propagated records still reach this handler.
    Existing handlers receive the same scrubber before they see a record.
    """
    for handler in logger.handlers:
        if getattr(handler, "_cashew_sanitizing_handler", False):
            return
    for handler in logger.handlers:
        handler.addFilter(ScrubFilter())
    logger.addHandler(_ForwardToInheritedHandlers(logger))
    logger.propagate = False
