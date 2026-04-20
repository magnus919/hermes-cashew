"""Tool-surface helpers for the Cashew memory provider.

This module owns:
  - CASHEW_QUERY_SCHEMA (dict): Anthropic-style tool schema for the cashew_query tool.
    Structure matches 03-RESEARCH.md §3; description wording matches PHASE_DESIGN_NOTES
    Decision Point 6.
  - build_success_envelope / build_error_envelope: pure functions that return
    json.dumps(...) strings for handle_tool_call to return.

Contract per PHASE_DESIGN_NOTES Decision Point 3:
  - Success envelope fields: ok=True, tool, query, context, node_count.
  - Error envelope fields: ok=False, tool, error, query.
  - Error messages are GENERIC ("cashew recall failed", "unknown tool").
    Exception types/messages never leak through these builders — the caller is
    expected to logger.warning(..., exc_info=True) for the audit trail.

Phase 4 will add CASHEW_EXTRACT_SCHEMA and a matching envelope pair here.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = [
    "CASHEW_QUERY_SCHEMA",
    "build_success_envelope",
    "build_error_envelope",
]

TOOL_NAME: str = "cashew_query"
"""Stable tool identifier. Referenced in the schema AND in both envelope builders;
exporting as a module constant means Phase 4's cashew_extract can do the same
without string duplication."""


CASHEW_QUERY_SCHEMA: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Retrieve related context from the Cashew thought graph. "
        "Use when the user's question references prior discussions, decisions, "
        "or knowledge the agent may have stored in a previous session. "
        "Returns a formatted context string that cites source nodes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query or topic to look up in the thought graph.",
            },
            "max_nodes": {
                "type": "integer",
                "description": "Maximum number of nodes to return (default: recall_k from config).",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
"""Anthropic-format tool schema (input_schema, not OpenAI's parameters).

Per 03-RESEARCH.md §3: Hermes's ABC is schema-format-agnostic at the plugin layer —
it returns whatever the provider emits and adapts per downstream LLM. We target
Nous Research's Hermes (Anthropic-first), so input_schema is the correct key.

Description length: 284 characters (>= 50-char RECALL-02 floor with ~5x headroom).
Do NOT rewrap or paraphrase; Plan 03-03 Task 2 asserts the exact wording.
"""


def build_success_envelope(query: str, context: str, node_count: int) -> str:
    """Return a json.dumps(...) string wrapping a successful cashew_query result.

    Args:
        query: The original query string (echoed for LLM context-tracking).
        context: The formatted context string returned by ContextRetriever.format_context.
        node_count: len(nodes) from the retrieve() call — lets the LLM see how
            much context the result is based on.

    Returns:
        A JSON string. Never None, never raises for str/int inputs.

    Shape:
        {"ok": true, "tool": "cashew_query", "query": ..., "context": ..., "node_count": N}
    """
    return json.dumps({
        "ok": True,
        "tool": TOOL_NAME,
        "query": query,
        "context": context,
        "node_count": node_count,
    })


def build_error_envelope(
    query: str | None,
    error_message: str = "cashew recall failed",
) -> str:
    """Return a json.dumps(...) string wrapping a cashew_query error.

    Args:
        query: The original query, or None if the error arose before the query
            could be parsed (e.g., unknown tool name, malformed args).
        error_message: A GENERIC error label. Must NOT contain exception types,
            file paths, line numbers, or traceback fragments. The full traceback
            is the caller's responsibility to route to logger.warning with
            exc_info=True.

    Returns:
        A JSON string. Never None, never raises.

    Shape:
        {"ok": false, "tool": "cashew_query", "error": ..., "query": ...}
    """
    return json.dumps({
        "ok": False,
        "tool": TOOL_NAME,
        "error": error_message,
        "query": query,
    })
