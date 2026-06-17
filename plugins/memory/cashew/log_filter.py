"""Log scrubbing processor for structlog.

Redacts sensitive values (API keys, tokens, user message content) from log
output. Used as a structlog processor to sanitize log entries before they
reach the renderer.
"""

from __future__ import annotations

import re
from typing import Any

# Patterns that look like API keys or tokens.
_SECRET_PATTERNS: list[tuple[str, str]] = [
    (r"(api_key|apikey|secret|token|password|passwd)\s*[:=]\s*\S+", r"\1=<REDACTED>"),
    (r"(sk-[a-zA-Z0-9]{20,})", "<REDACTED_KEY>"),
    (r"(ghp_[a-zA-Z0-9]{36})", "<REDACTED_GH_TOKEN>"),
    (r"(Bearer\s+)\S+", r"\1<REDACTED>"),
]

# Maximum length for user/assistant message content in logs.
_MAX_CONTENT_LENGTH = 200


def scrub_value(value: str) -> str:
    """Apply secret patterns to a string value."""
    for pattern, replacement in _SECRET_PATTERNS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value


def _scrub_event_dict(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that scrubs sensitive values from the event dict."""
    for key in ("user_content", "assistant_content", "message", "prompt"):
        if key in event_dict and isinstance(event_dict[key], str):
            if len(event_dict[key]) > _MAX_CONTENT_LENGTH:
                event_dict[key] = event_dict[key][:_MAX_CONTENT_LENGTH] + "..."
    # Scrub string values that may contain secrets
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = scrub_value(value)
    return event_dict


def get_scrub_processor() -> Any:
    """Return the structlog processor for log scrubbing."""
    return _scrub_event_dict
