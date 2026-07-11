"""Log scrubbing filter for the cashew memory provider.

Redacts credential-shaped values and bounds log-message length via a standard
library logging.Filter. Call sites must never pass ordinary user content: a
generic logging filter cannot reliably distinguish it from operational text.
"""

from __future__ import annotations

import logging
import re

# Patterns that look like API keys or tokens.
_SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"(api_key|apikey|secret|token|password|passwd)\s*[:=]\s*\S+", r"\1=<REDACTED>"),
    (r"(sk-[a-zA-Z0-9]{20,})", "<REDACTED_KEY>"),
    (r"(ghp_[a-zA-Z0-9]{36})", "<REDACTED_GH_TOKEN>"),
    (r"(Bearer\s+)\S+", r"\1<REDACTED>"),
]

# Maximum length for user/assistant message content in logs.
_MAX_CONTENT_LENGTH = 200


def _scrub(value: str) -> str:
    """Apply secret patterns to a string value."""
    for pattern, replacement in _SECRET_PATTERNS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


class ScrubFilter(logging.Filter):
    """Logging filter that redacts sensitive values from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # Truncate long content fields
        if len(msg) > _MAX_CONTENT_LENGTH:
            msg = msg[:_MAX_CONTENT_LENGTH] + "..."
        record.msg = _scrub(msg)
        record.args = None  # prevent double-formatting after msg is rewritten
        return True


def add_scrub_filter(logger: logging.Logger) -> None:
    """Install the scrub filter on a logger instance."""
    logger.addFilter(ScrubFilter())
