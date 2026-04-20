# tests/test_cashew_extract.py
# Phase 4 Plan 04-03 Task 3: cashew_extract schema + handler + stack-trace-leak guard.
# Covers SYNC-03, SYNC-04.
from __future__ import annotations

import json
import logging
import sqlite3
import types
from typing import Any

import jsonschema
import pytest

from plugins.memory.cashew import CashewMemoryProvider
from plugins.memory.cashew.tools import (
    CASHEW_EXTRACT_SCHEMA,
    EXTRACT_TOOL_NAME,
    build_extract_error_envelope,
    build_extract_success_envelope,
)


# -- Module-level helpers --


def fake_end_session_ok(calls_list: list) -> Any:
    def _fake(*args: Any, **kwargs: Any) -> Any:
        calls_list.append(kwargs)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    return _fake


def fake_end_session_raises(exc: Exception) -> Any:
    def _fake(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return _fake


def make_initialized_provider(tmp_path) -> CashewMemoryProvider:
    p = CashewMemoryProvider()
    p.save_config({}, str(tmp_path))
    p.initialize("test-sync", hermes_home=str(tmp_path))
    return p


# -- Schema tests --


def test_extract_schema_shape():
    assert set(CASHEW_EXTRACT_SCHEMA.keys()) >= {"name", "description", "input_schema"}
    assert CASHEW_EXTRACT_SCHEMA["name"] == "cashew_extract"
    assert CASHEW_EXTRACT_SCHEMA["name"] == EXTRACT_TOOL_NAME


def test_extract_schema_description_length():
    assert len(CASHEW_EXTRACT_SCHEMA["description"]) >= 50


def test_extract_schema_required_fields():
    ischema = CASHEW_EXTRACT_SCHEMA["input_schema"]
    assert ischema["required"] == ["user_content", "assistant_content"]
    assert ischema["additionalProperties"] is False
    assert ischema["type"] == "object"
    assert "user_content" in ischema["properties"]
    assert "assistant_content" in ischema["properties"]


def test_extract_schema_passes_jsonschema_draft7():
    ischema = CASHEW_EXTRACT_SCHEMA["input_schema"]
    # Meta-validation: the schema itself is a valid JSON Schema.
    jsonschema.Draft7Validator.check_schema(ischema)
    # Positive instance validation.
    jsonschema.validate(
        instance={"user_content": "u", "assistant_content": "a"},
        schema=ischema,
    )
    # Negative: missing field.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"user_content": "u"}, schema=ischema)
    # Negative: extra field.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"user_content": "u", "assistant_content": "a", "extra": 1},
            schema=ischema,
        )
    # Negative: wrong type.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            instance={"user_content": 123, "assistant_content": "a"},
            schema=ischema,
        )


# -- Handler tests --


def test_extract_happy_path_returns_success_envelope(tmp_path, monkeypatch):
    fake_result = types.SimpleNamespace(
        new_nodes=["n1", "n2", "n3"], new_edges=["e1"], updated_nodes=[]
    )
    monkeypatch.setattr(
        "core.session.end_session", lambda **k: fake_result, raising=False
    )
    p = make_initialized_provider(tmp_path)
    try:
        result = p.handle_tool_call(
            "cashew_extract",
            {"user_content": "hi", "assistant_content": "hello"},
        )
        assert isinstance(result, str)
        d = json.loads(result)
        assert d == {
            "ok": True,
            "tool": "cashew_extract",
            "new_nodes": 3,
            "new_edges": 1,
        }
    finally:
        p.shutdown()


def test_extract_calls_end_session_with_canonical_kwargs(tmp_path, monkeypatch):
    recorded: list[dict] = []

    def _fake(**kwargs):
        recorded.append(kwargs)
        return types.SimpleNamespace(new_nodes=[], new_edges=[], updated_nodes=[])

    monkeypatch.setattr("core.session.end_session", _fake, raising=False)
    p = make_initialized_provider(tmp_path)
    try:
        p.handle_tool_call(
            "cashew_extract",
            {"user_content": "foo", "assistant_content": "bar"},
        )
        assert len(recorded) == 1
        kw = recorded[0]
        assert kw["db_path"] == str(p._db_path)
        assert kw["session_id"] == "test-sync"
        assert kw["conversation_text"] == "User: foo\nAssistant: bar"
        assert kw["model_fn"] is None
    finally:
        p.shutdown()


def test_extract_bypasses_sync_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.session.end_session",
        lambda **k: types.SimpleNamespace(
            new_nodes=[], new_edges=[], updated_nodes=[]
        ),
        raising=False,
    )
    p = make_initialized_provider(tmp_path)
    try:
        qsize_before = p._sync_queue.qsize()
        unfinished_before = p._sync_queue.unfinished_tasks
        p.handle_tool_call(
            "cashew_extract",
            {"user_content": "x", "assistant_content": "y"},
        )
        qsize_after = p._sync_queue.qsize()
        unfinished_after = p._sync_queue.unfinished_tasks
        assert qsize_after == qsize_before == 0
        assert unfinished_after == unfinished_before == 0
    finally:
        p.shutdown()


def test_extract_half_state_returns_error_envelope_no_log(caplog):
    p = CashewMemoryProvider()  # never initialized
    with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
        result = p.handle_tool_call(
            "cashew_extract",
            {"user_content": "u", "assistant_content": "a"},
        )
    assert json.loads(result) == {
        "ok": False,
        "tool": "cashew_extract",
        "error": "cashew extract failed",
    }
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_extract_missing_arg_returns_error_envelope_and_logs_once(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(
        "core.session.end_session",
        lambda **k: types.SimpleNamespace(
            new_nodes=[], new_edges=[], updated_nodes=[]
        ),
        raising=False,
    )
    p = make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call(
                "cashew_extract",
                {"user_content": "u"},  # missing assistant_content
            )
        d = json.loads(result)
        assert d["ok"] is False
        assert d["tool"] == "cashew_extract"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_extract_cashew_failure_returns_error_envelope_and_logs_once(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(
        "core.session.end_session",
        lambda **k: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
        raising=False,
    )
    p = make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call(
                "cashew_extract",
                {"user_content": "u", "assistant_content": "a"},
            )
        d = json.loads(result)
        assert d["ok"] is False
        assert d["error"] == "cashew extract failed"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert warnings[0].exc_info is not None
    finally:
        p.shutdown()


def test_extract_error_envelope_has_no_stack_trace_substrings(
    tmp_path, monkeypatch, caplog
):
    SECRET = "/some/secret/path/brain.db"
    monkeypatch.setattr(
        "core.session.end_session",
        lambda **k: (_ for _ in ()).throw(
            sqlite3.OperationalError(f"database is locked {SECRET}")
        ),
        raising=False,
    )
    p = make_initialized_provider(tmp_path)
    try:
        with caplog.at_level(logging.WARNING, logger="plugins.memory.cashew"):
            result = p.handle_tool_call(
                "cashew_extract",
                {"user_content": "u", "assistant_content": "a"},
            )
        # The returned JSON string must NOT contain traceback fragments.
        forbidden = ["Traceback", 'File "', "sqlite3.", "at line", SECRET]
        for f in forbidden:
            assert f not in result, f"error envelope leaks {f!r}: {result!r}"
        # But the logger record DOES have the full traceback (operator audit).
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings and warnings[0].exc_info is not None
    finally:
        p.shutdown()
