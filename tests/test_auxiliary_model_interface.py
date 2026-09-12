"""Hermes auxiliary-model boundary tests for the Cashew adapter."""

from __future__ import annotations

import contextvars
import logging
import sqlite3
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import Any

import pytest

import plugins.memory.cashew.config as config_module
from plugins.memory.cashew.config import CashewConfig, resolve_model_fn


class _FakeClient:
    def __init__(self, responder):
        self._responder = responder
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responder(kwargs)


def _response(content: Any) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture
def fake_host(monkeypatch):
    """Install only the public Hermes interfaces the adapter consumes."""
    active_home: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "fake_hermes_home", default=None
    )
    profiles: dict[str, dict[str, Any]] = {}
    resolved_homes: list[str | None] = []
    resets: list[object] = []
    client_holder: dict[str, Any] = {"client": None, "model": "fake-model"}

    constants = types.ModuleType("hermes_constants")

    def set_home(path):
        return active_home.set(str(path))

    def reset_home(token):
        resets.append(token)
        active_home.reset(token)

    constants.set_hermes_home_override = set_home
    constants.reset_hermes_home_override = reset_home

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []
    host_config = types.ModuleType("hermes_cli.config")
    host_config.read_raw_config_readonly = lambda: profiles.get(active_home.get(), {})

    aux = types.ModuleType("agent.auxiliary_client")

    def get_text_auxiliary_client(role):
        resolved_homes.append(active_home.get())
        return client_holder["client"], client_holder["model"]

    aux.get_text_auxiliary_client = get_text_auxiliary_client

    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", host_config)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux)
    return SimpleNamespace(
        active_home=active_home,
        profiles=profiles,
        resolved_homes=resolved_homes,
        resets=resets,
        client_holder=client_holder,
    )


def _enabled_model(fake_host, home, client):
    fake_host.profiles[str(home)] = {"auxiliary": {"memory": {"provider": "fake"}}}
    fake_host.client_holder["client"] = client
    model_fn = resolve_model_fn(home, CashewConfig(llm_aux_role="memory"))
    assert model_fn is not None
    return model_fn


def test_resolver_uses_profile_scoped_host_client_and_fixed_bounds(fake_host, tmp_path):
    client = _FakeClient(lambda _: _response("[]"))
    model_fn = _enabled_model(fake_host, tmp_path, client)

    assert model_fn("extract this") == "[]"
    assert fake_host.resolved_homes == [str(tmp_path)]
    assert fake_host.active_home.get() is None
    assert fake_host.resets
    assert client.calls == [
        {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "extract this"}],
            "max_tokens": config_module._AUXILIARY_MAX_TOKENS,
            "timeout": config_module._AUXILIARY_DEADLINE_SECONDS,
        }
    ]


def test_prompt_budget_preserves_instructions_and_recent_context(fake_host, tmp_path):
    client = _FakeClient(lambda _: _response("[]"))
    model_fn = _enabled_model(fake_host, tmp_path, client)
    prompt = "I" * config_module._AUXILIARY_PROMPT_PREFIX + "middle" * 9_000 + "R" * 64

    assert model_fn(prompt) == "[]"
    sent_prompt = client.calls[0]["messages"][0]["content"]
    assert len(sent_prompt) == config_module._AUXILIARY_PROMPT_LIMIT
    assert sent_prompt.startswith("I" * config_module._AUXILIARY_PROMPT_PREFIX)
    assert "[Cashew truncated older prompt content.]" in sent_prompt
    assert sent_prompt.endswith("R" * 64)


def test_role_changed_to_null_after_closure_never_auto_routes(fake_host, tmp_path):
    client = _FakeClient(lambda _: _response("[]"))
    model_fn = _enabled_model(fake_host, tmp_path, client)
    fake_host.profiles[str(tmp_path)]["auxiliary"]["memory"] = None

    assert model_fn("must not call") == ""
    assert fake_host.resolved_homes == []
    assert client.calls == []


def test_failed_request_resets_profile_context_and_does_not_log_prompt(
    fake_host, tmp_path, caplog
):
    secret_prompt = "do-not-log-this-sensitive-prompt"
    model_fn = _enabled_model(
        fake_host,
        tmp_path,
        _FakeClient(lambda _: (_ for _ in ()).throw(RuntimeError("transport failed"))),
    )

    assert model_fn(secret_prompt) == ""
    assert fake_host.active_home.get() is None
    assert fake_host.resets
    assert secret_prompt not in caplog.text


def test_closing_callable_rejects_new_calls(fake_host, tmp_path):
    client = _FakeClient(lambda _: _response("[]"))
    model_fn = _enabled_model(fake_host, tmp_path, client)

    close = getattr(model_fn, "_cashew_close")
    close()

    assert model_fn("after shutdown") == ""
    assert client.calls == []


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "null content"),
        ("", "empty content"),
        ("   ", "empty content"),
        (42, "non-text content"),
    ],
)
def test_non_json_content_outcomes_are_operationally_distinct(
    fake_host, tmp_path, caplog, content, reason
):
    caplog.set_level(logging.INFO)
    model_fn = _enabled_model(
        fake_host, tmp_path, _FakeClient(lambda _: _response(content))
    )

    assert model_fn("extract") == ""
    assert reason in caplog.text


def test_malformed_response_is_not_coerced_to_text(fake_host, tmp_path, caplog):
    model_fn = _enabled_model(fake_host, tmp_path, _FakeClient(lambda _: object()))

    assert model_fn("extract") == ""
    assert "malformed response" in caplog.text


def test_valid_empty_json_is_preserved_for_upstream(fake_host, tmp_path):
    model_fn = _enabled_model(
        fake_host, tmp_path, _FakeClient(lambda _: _response("[]"))
    )
    assert model_fn("extract") == "[]"


def test_process_wide_gate_caps_hung_calls_across_new_closures(
    fake_host, tmp_path, monkeypatch
):
    monkeypatch.setattr(config_module, "_AUXILIARY_DEADLINE_SECONDS", 0.02)
    monkeypatch.setattr(config_module, "_OUTSTANDING_AUXILIARY_CALLS", 0)
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()

    def block(_):
        nonlocal calls
        with calls_lock:
            calls += 1
            if calls >= config_module._MAX_OUTSTANDING_AUXILIARY_CALLS:
                started.set()
        release.wait(timeout=2)
        return _response("[]")

    client = _FakeClient(block)
    functions = [_enabled_model(fake_host, tmp_path, client) for _ in range(100)]
    try:
        for model_fn in functions:
            assert model_fn("hung") == ""

        assert started.wait(timeout=1)
        assert calls == config_module._MAX_OUTSTANDING_AUXILIARY_CALLS
        assert (
            config_module._OUTSTANDING_AUXILIARY_CALLS
            == config_module._MAX_OUTSTANDING_AUXILIARY_CALLS
        )

        release.set()
        deadline = time.monotonic() + 1
        while (
            config_module._OUTSTANDING_AUXILIARY_CALLS and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert config_module._OUTSTANDING_AUXILIARY_CALLS == 0

        assert functions[0]("recovered") == "[]"
        assert calls == config_module._MAX_OUTSTANDING_AUXILIARY_CALLS + 1
    finally:
        release.set()
        deadline = time.monotonic() + 1
        while (
            config_module._OUTSTANDING_AUXILIARY_CALLS and time.monotonic() < deadline
        ):
            time.sleep(0.01)


def test_empty_response_uses_upstream_heuristic_while_json_array_does_not(
    fake_host, tmp_path, monkeypatch
):
    import core.session as session

    monkeypatch.setattr(session, "embed_nodes", lambda _: None)
    monkeypatch.setattr(session, "_find_similar_nodes", lambda *_: [])
    fallback_calls: list[str] = []

    def fallback(_):
        fallback_calls.append("called")
        return [{"content": "heuristic fallback fact", "type": "observation"}]

    monkeypatch.setattr(session, "_extract_with_heuristics", fallback)
    empty_model = _enabled_model(
        fake_host, tmp_path, _FakeClient(lambda _: _response(""))
    )
    empty_db = tmp_path / "empty.db"
    empty = session.end_session(
        str(empty_db), "s", "conversation long enough", empty_model
    )
    assert len(empty.new_nodes) == 1
    with sqlite3.connect(empty_db) as conn:
        assert (
            conn.execute("SELECT content FROM thought_nodes").fetchone()[0]
            == "heuristic fallback fact"
        )
    assert fallback_calls == ["called"]

    fallback_calls.clear()
    json_model = _enabled_model(
        fake_host, tmp_path, _FakeClient(lambda _: _response("[]"))
    )
    json_db = tmp_path / "json.db"
    empty = session.end_session(
        str(json_db), "s", "conversation long enough", json_model
    )
    assert empty.new_nodes == []
    assert fallback_calls == []
