# tests/test_tool_schema.py
# Phase 3 Plan 03-03 Task 2: CASHEW_QUERY_SCHEMA + get_tool_schemas tests — RECALL-02.
from __future__ import annotations

import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.tools import CASHEW_QUERY_SCHEMA


CANONICAL_DESCRIPTION_PREFIX = (
    "Retrieve related context from the Cashew thought graph. "
    "Use when the user's question references prior discussions, decisions, "
    "or knowledge the agent may have stored in a previous session."
)


def test_schema_has_required_top_level_keys():
    """Schema structural contract — OpenAI tool-schema shape."""
    assert set(CASHEW_QUERY_SCHEMA.keys()) == {"name", "description", "parameters"}
    assert CASHEW_QUERY_SCHEMA["name"] == "cashew_query"


def test_schema_description_meets_50_char_floor():
    """RECALL-02: description >= 50 chars."""
    assert len(CASHEW_QUERY_SCHEMA["description"]) >= 50


def test_schema_description_matches_canonical_wording():
    """PHASE_DESIGN_NOTES Decision Point 6: description is the exact 03-RESEARCH.md §3 wording.

    Sub-string check — tolerant of trailing whitespace / line-break reflow but requires
    the canonical phrasing intact."""
    assert CANONICAL_DESCRIPTION_PREFIX in CASHEW_QUERY_SCHEMA["description"], (
        f"description was paraphrased; expected canonical prefix {CANONICAL_DESCRIPTION_PREFIX!r}"
    )


def test_schema_parameters_required_is_query_only():
    """RECALL-02: `required` is explicitly the one-element list ["query"]."""
    params = CASHEW_QUERY_SCHEMA["parameters"]
    assert params["required"] == ["query"]


def test_schema_parameters_additional_properties_false():
    """Defensive: unknown parameters are rejected at schema-validation layer (T-03-02-04)."""
    assert CASHEW_QUERY_SCHEMA["parameters"]["additionalProperties"] is False


def test_schema_query_property_is_string():
    props = CASHEW_QUERY_SCHEMA["parameters"]["properties"]
    assert props["query"]["type"] == "string"
    assert props["query"].get("description"), "query property needs a description"


def test_schema_max_nodes_property_is_bounded_integer():
    props = CASHEW_QUERY_SCHEMA["parameters"]["properties"]
    mn = props["max_nodes"]
    assert mn["type"] == "integer"
    assert mn["minimum"] == 1
    assert mn["maximum"] == 20


def test_schema_uses_openai_parameters_key_naming():
    """OpenAI-format: parameters key, NOT Anthropic's input_schema."""
    assert "parameters" in CASHEW_QUERY_SCHEMA
    assert "input_schema" not in CASHEW_QUERY_SCHEMA
    assert "function" not in CASHEW_QUERY_SCHEMA


def test_schema_passes_draft7_meta_validation():
    """JSON-Schema structural validity — catches typos / invalid keywords."""
    jsonschema = pytest.importorskip("jsonschema")
    # Validate the parameters portion (it's the actual JSON-Schema document).
    jsonschema.Draft7Validator.check_schema(CASHEW_QUERY_SCHEMA["parameters"])


def test_valid_args_pass_jsonschema_validation():
    """Positive: a well-formed LLM call validates."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate({"query": "hello"}, CASHEW_QUERY_SCHEMA["parameters"])
    jsonschema.validate(
        {"query": "hello", "max_nodes": 5},
        CASHEW_QUERY_SCHEMA["parameters"],
    )


def test_missing_query_fails_jsonschema_validation():
    """Negative: required field enforcement."""
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({}, CASHEW_QUERY_SCHEMA["parameters"])


def test_unknown_parameter_fails_jsonschema_validation():
    """Negative: additionalProperties=False rejects unknown fields."""
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"query": "x", "bogus": 1},
            CASHEW_QUERY_SCHEMA["parameters"],
        )


def test_max_nodes_above_cap_fails_jsonschema_validation():
    """Negative: max_nodes has an upper bound of 20 to cap retrieval load (T-03-02-03)."""
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"query": "x", "max_nodes": 100},
            CASHEW_QUERY_SCHEMA["parameters"],
        )


def test_provider_get_tool_schemas_returns_single_cashew_query_schema():
    """RECALL-02 + SYNC-03: get_tool_schemas exposes both tool schemas.

    Phase 3 shipped one schema (cashew_query); Phase 4 Plan 04-02 appended
    cashew_extract. The first-slot identity invariant (cashew_query is
    schemas[0]) is preserved so any downstream iteration order assumption
    remains stable. Both schemas must be module constants, not copies —
    callers must not mutate them.
    """
    schemas = CashewMemoryProvider().get_tool_schemas()
    assert len(schemas) == 2, "Phase 4 expects two schemas: cashew_query + cashew_extract"
    assert schemas[0] is CASHEW_QUERY_SCHEMA, "get_tool_schemas must return the module constant, not a copy"
