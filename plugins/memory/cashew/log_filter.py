"""Content-free local logging for the Cashew provider hierarchy."""

from __future__ import annotations

import importlib.util
import logging
import re
import threading
import weakref
from pathlib import Path
from typing import Any, Mapping, cast

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


def _upstream_source_dir() -> Path | None:
    """Locate the pinned Cashew package without importing or configuring it."""
    try:
        spec = importlib.util.find_spec("core")
        if spec is not None and spec.origin is not None:
            return Path(spec.origin).resolve().parent
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    return None


_UPSTREAM_SOURCE_DIR = _upstream_source_dir()
_SOURCE_DIRS = tuple(
    directory
    for directory in (_PACKAGE_DIR, _UPSTREAM_SOURCE_DIR)
    if directory is not None
)
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
    """Match only this adapter or the pinned Cashew source tree by pathname."""
    try:
        source = Path(record.pathname).resolve()
        return any(source.is_relative_to(directory) for directory in _SOURCE_DIRS)
    except (OSError, ValueError):
        return False


def _is_adapter_source(record: logging.LogRecord) -> bool:
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


def _safe_arguments(args: object) -> tuple[object, ...] | Mapping[str, object]:
    if isinstance(args, Mapping):
        return {str(key): _safe_argument(value) for key, value in args.items()}
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

    def __init__(self, *, source_scoped: bool = False) -> None:
        super().__init__()
        self._source_scoped = source_scoped

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_cashew_scrubbed", False):
            return True
        if self._source_scoped and not _is_owned_source(record):
            return True
        error = (
            record.exc_info[1]
            if isinstance(record.exc_info, tuple) and len(record.exc_info) > 1
            else None
        )
        suffix = f" error={_safe_error_class(error)}" if error is not None else ""
        # Checked-in provider modules use static operation templates. Keep only
        # that source-owned template (never its arguments); externally emitted
        # child records are reduced to a generic code because their provenance
        # is not a privacy boundary the adapter can prove.
        template = record.msg if isinstance(record.msg, str) else ""
        if _is_adapter_source(record):
            record.msg = template + suffix
            if template.startswith("embedding worker failure:"):
                record.args = cast(Any, _safe_embedding_failure_arguments(record.args))
            elif template.startswith("embedding identity %s; unable to acquire"):
                record.args = cast(Any, _safe_identity_lock_arguments(record.args))
            else:
                record.args = cast(Any, _safe_arguments(record.args))
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

    def __init__(self, logger: logging.Logger, *, source_scoped: bool) -> None:
        super().__init__(level=logging.NOTSET)
        self._logger_ref = weakref.ref(logger)
        self._emit_lock = threading.RLock()
        self.addFilter(ScrubFilter(source_scoped=source_scoped))

    def emit(self, record: logging.LogRecord) -> None:
        logger = self._logger_ref()
        if logger is None:
            return
        # ``Logger`` filters do not run for propagated children. Reproduce the
        # normal ancestor-handler walk after this owned handler scrubs the one
        # shared record. At the root, the normal handler loop already reaches
        # later siblings after this handler, so there is nothing to forward.
        if logger.parent is None:
            return
        with self._emit_lock:
            ancestor: logging.Logger | None = logger.parent
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


_SCRUB_LOCK = threading.RLock()
_SCRUB_REFS = 0
_SCRUB_INSTALLS: list[
    tuple[
        logging.Logger,
        bool,
        list[tuple[logging.Handler, ScrubFilter]],
        _ForwardToInheritedHandlers,
    ]
] = []


def acquire_provider_scrub_filters() -> None:
    """Install one reversible boundary for all live Cashew providers."""
    global _SCRUB_REFS
    with _SCRUB_LOCK:
        _SCRUB_REFS += 1
        if _SCRUB_REFS != 1:
            return
        # ``core`` covers upstream named records; root covers upstream modules
        # that call ``logging.warning`` directly. Both are source-path scoped.
        for logger, source_scoped in (
            (logging.getLogger("plugins.memory.cashew"), False),
            (logging.getLogger("core"), True),
            (logging.getLogger(), True),
        ):
            installed: list[tuple[logging.Handler, ScrubFilter]] = []
            for handler in logger.handlers:
                scrubber = ScrubFilter(source_scoped=source_scoped)
                handler.addFilter(scrubber)
                installed.append((handler, scrubber))
            boundary = _ForwardToInheritedHandlers(logger, source_scoped=source_scoped)
            logger.addHandler(boundary)
            _SCRUB_INSTALLS.append((logger, logger.propagate, installed, boundary))
            logger.propagate = False


def release_provider_scrub_filters() -> None:
    """Remove exactly the handlers and filters installed by Cashew."""
    global _SCRUB_REFS
    with _SCRUB_LOCK:
        if _SCRUB_REFS == 0:
            return
        _SCRUB_REFS -= 1
        if _SCRUB_REFS:
            return
        while _SCRUB_INSTALLS:
            logger, original_propagate, installed, boundary = _SCRUB_INSTALLS.pop()
            logger.removeHandler(boundary)
            for handler, scrubber in installed:
                handler.removeFilter(scrubber)
            # Cashew set this only while its boundary was active. If the host
            # changed it during that window, preserve the host's newer value.
            if logger.propagate is False:
                logger.propagate = original_propagate


def add_scrub_filter(logger: logging.Logger, *, source_scoped: bool = False) -> None:
    """Deprecated test helper for a standalone hierarchy boundary."""
    for handler in logger.handlers:
        if getattr(handler, "_cashew_sanitizing_handler", False):
            return
    for handler in logger.handlers:
        handler.addFilter(ScrubFilter(source_scoped=source_scoped))
    logger.addHandler(_ForwardToInheritedHandlers(logger, source_scoped=source_scoped))
    logger.propagate = False
