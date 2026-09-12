"""Issue #202 diagnostics and provider-local outcome contracts."""

from __future__ import annotations

import json
import threading
import time
import types

from plugins.memory.cashew import CashewMemoryProvider


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_health_status_is_bounded_and_read_only(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    before = provider.health_status()

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("health_status must not probe storage")

    monkeypatch.setattr(provider, "_load_think_counter", fail_if_called)
    monkeypatch.setattr(provider, "_save_think_counter", fail_if_called)
    after = provider.health_status()

    assert before == after
    assert after == {
        "state": "unconfigured",
        "reason_code": "not_initialized",
        "generation": 0,
        "runtime": {
            "config_loaded": False,
            "retriever_ready": False,
            "write_enabled": True,
            "worker_running": False,
        },
        "fallback": "none",
        "cron": "disabled",
        "last_error": None,
        "work": {
            "accepted": 0,
            "completed": 0,
            "failed": 0,
            "dropped": 0,
            "rejected": 0,
            "pending": 0,
            "in_flight": 0,
            "reconciled": True,
        },
        "tools": {
            "query_completed": 0,
            "query_failed": 0,
            "query_empty": 0,
            "extract_completed": 0,
            "extract_failed": 0,
            "extract_empty": 0,
        },
    }
    assert list(tmp_path.iterdir()) == []


def test_missing_config_uses_defaults_and_reports_runtime(tmp_path):
    provider = CashewMemoryProvider()
    provider.initialize("health-defaults", hermes_home=str(tmp_path))
    try:
        status = provider.health_status()
        assert status["state"] in {"ready", "degraded"}
        assert status["reason_code"] != "config_invalid"
        assert status["runtime"]["config_loaded"] is True
        assert status["runtime"]["retriever_ready"] is True
        assert status["generation"] == 1
    finally:
        provider.shutdown()


def test_init_vector_unavailable_reports_keyword_degradation(tmp_path, monkeypatch):
    # Vec work is deliberately finalized after backup-backed embedding repair.
    # Keep this degradation seam at that finalization boundary.
    def vec_unavailable(provider, _db_path):
        provider._vector_available = False

    embedded_queries: list[dict] = []
    persisted_extracts: list[dict] = []

    def record_embedding_route(**kwargs):
        embedded_queries.append(kwargs)
        return []

    def persist_extract(**kwargs):
        persisted_extracts.append(kwargs)
        return types.SimpleNamespace(new_nodes=["written"], new_edges=[])

    monkeypatch.setattr(CashewMemoryProvider, "_finalize_vec_schema", vec_unavailable)
    monkeypatch.setattr("core.retrieval.retrieve_recursive_bfs", record_embedding_route)
    monkeypatch.setattr("core.session.end_session", persist_extract, raising=False)
    provider = CashewMemoryProvider()
    provider.initialize("vector-health", hermes_home=str(tmp_path))
    try:
        status = provider.health_status()
        assert status["state"] == "degraded"
        assert status["reason_code"] == "vector_unavailable"
        assert status["fallback"] == "keyword"
        assert provider._embedding_identity_ready is True

        extract = json.loads(
            provider.handle_tool_call(
                "cashew_extract",
                {"user_content": "write", "assistant_content": "admitted"},
            )
        )
        assert extract["ok"] is True
        assert persisted_extracts

        provider.prefetch("embedding route")
        assert embedded_queries == [
            {
                "db_path": str(provider._db_path),
                "query": "embedding route",
                "top_k": provider._config.recall_k,
                "domain": None,
                "tags": None,
                "exclude_tags": None,
            }
        ]
    finally:
        provider.shutdown()


def test_init_failure_is_distinct_from_discovery(monkeypatch, tmp_path):
    provider = CashewMemoryProvider()
    provider.save_config({}, str(tmp_path))
    monkeypatch.setattr("plugins.memory.cashew.ContextRetriever", None)

    provider.initialize("broken", hermes_home=str(tmp_path))

    status = provider.health_status()
    assert status["state"] == "failed"
    assert status["reason_code"] == "dependency_missing"
    assert status["runtime"]["config_loaded"] is False
    assert status["last_error"]["class"] == "RuntimeError"
    assert provider.is_available() is True


def test_query_outcomes_distinguish_empty_success_from_failure(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.initialize("query-health", hermes_home=str(tmp_path))
    monkeypatch.setattr(provider, "_keyword_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: [],
    )
    try:
        result = json.loads(
            provider.handle_tool_call("cashew_query", {"query": "none"})
        )
        assert result["ok"] is True
        assert result["node_count"] == 0
        status = provider.health_status()
        assert status["tools"]["query_completed"] == 1
        assert status["tools"]["query_empty"] == 1
        assert status["tools"]["query_failed"] == 0
    finally:
        provider.shutdown()


def test_vector_failure_reports_keyword_degradation_without_tool_error(
    tmp_path, monkeypatch
):
    provider = CashewMemoryProvider()
    provider.initialize("keyword-health", hermes_home=str(tmp_path))
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vector detail")),
    )
    monkeypatch.setattr(provider, "_keyword_search", lambda *args, **kwargs: [])
    try:
        result = json.loads(
            provider.handle_tool_call("cashew_query", {"query": "none"})
        )
        assert result["ok"] is True
        status = provider.health_status()
        assert status["state"] == "degraded"
        assert status["reason_code"] == "vector_unavailable"
        assert status["fallback"] == "keyword"
        assert status["tools"]["query_completed"] == 1
    finally:
        provider.shutdown()


def test_prefetch_fallback_updates_health_without_changing_return_contract(
    tmp_path, monkeypatch
):
    provider = CashewMemoryProvider()
    provider.initialize("prefetch-health", hermes_home=str(tmp_path))
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vector detail")),
    )
    monkeypatch.setattr(provider, "_keyword_search", lambda *args, **kwargs: [])
    try:
        assert provider.prefetch("nothing") == ""
        status = provider.health_status()
        assert status["state"] == "degraded"
        assert status["reason_code"] == "vector_unavailable"
        assert status["fallback"] == "keyword"
    finally:
        provider.shutdown()


def test_prefetch_keyword_failure_reports_backend_error(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.initialize("prefetch-backend", hermes_home=str(tmp_path))
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vector detail")),
    )
    monkeypatch.setattr(
        provider,
        "_keyword_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db detail")),
    )
    try:
        assert provider.prefetch("nothing") == ""
        status = provider.health_status()
        assert status["state"] == "degraded"
        assert status["reason_code"] == "backend_error"
    finally:
        provider.shutdown()


def test_write_disabled_runtime_is_not_reported_unconfigured(tmp_path):
    provider = CashewMemoryProvider()
    provider.initialize(
        "secondary-health", hermes_home=str(tmp_path), agent_context="secondary"
    )
    try:
        status = provider.health_status()
        assert status["state"] in {"ready", "degraded"}
        assert status["state"] != "unconfigured"
        assert status["runtime"]["write_enabled"] is False
        assert (
            json.loads(provider.handle_tool_call("cashew_extract", {}))["ok"] is False
        )
    finally:
        provider.shutdown()


def test_sync_accounting_reconciles_and_resets_per_generation(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.initialize("sync-health", hermes_home=str(tmp_path))
    monkeypatch.setattr(provider, "_drain_once", lambda turn: True)
    try:
        provider.sync_turn("user", "assistant")
        _wait_for(lambda: provider.health_status()["work"]["completed"] == 1)
        status = provider.health_status()
        assert status["work"]["accepted"] == 1
        assert status["work"]["completed"] == 1
        assert status["work"]["reconciled"] is True
    finally:
        provider.shutdown()

    provider.initialize("sync-health-next", hermes_home=str(tmp_path))
    try:
        status = provider.health_status()
        assert status["generation"] == 2
        assert status["work"]["accepted"] == 0
        assert status["work"]["completed"] == 0
        assert status["work"]["reconciled"] is True
    finally:
        provider.shutdown()


def test_late_old_generation_health_finding_cannot_mutate_new_generation(tmp_path):
    provider = CashewMemoryProvider()
    provider.initialize("generation-a", hermes_home=str(tmp_path / "a"))
    old_status = provider.health_status()
    with provider._sync_state_lock:
        old_ledger = provider._outcomes
        old_generation = provider._health_generation
    provider.shutdown()
    provider.initialize("generation-b", hermes_home=str(tmp_path / "b"))
    try:
        before = provider.health_status()
        provider._mark_health_if_current(
            old_ledger,
            old_generation,
            "degraded",
            "backend_error",
        )
        after = provider.health_status()
        assert after["generation"] == old_status["generation"] + 1
        assert after["generation"] == before["generation"]
        assert after["reason_code"] == before["reason_code"]
        assert after["state"] == before["state"]
    finally:
        provider.shutdown()


def test_late_query_failure_cannot_mutate_reinitialized_generation(
    tmp_path, monkeypatch
):
    provider = CashewMemoryProvider()
    provider.initialize("query-a", hermes_home=str(tmp_path / "a"))
    entered = threading.Event()
    release = threading.Event()
    result_holder = {}

    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: [],
    )

    def late_failure(*args, **kwargs):
        del args, kwargs
        entered.set()
        assert release.wait(timeout=2.0)
        raise RuntimeError("old generation backend")

    monkeypatch.setattr(provider, "_keyword_search", late_failure)
    query_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", provider.handle_tool_call("cashew_query", {"query": "old"})
        )
    )
    query_thread.start()
    assert entered.wait(timeout=2.0)
    provider.shutdown()
    provider.initialize("query-b", hermes_home=str(tmp_path / "b"))
    try:
        before = provider.health_status()
        release.set()
        query_thread.join(timeout=2.0)
        assert not query_thread.is_alive()
        assert json.loads(result_holder["result"])["ok"] is False
        after = provider.health_status()
        assert after["generation"] == before["generation"]
        assert after["tools"]["query_failed"] == 0
        assert after["reason_code"] == before["reason_code"]
    finally:
        release.set()
        provider.shutdown()


def test_late_query_completion_cannot_mutate_reinitialized_generation(
    tmp_path, monkeypatch
):
    provider = CashewMemoryProvider()
    provider.initialize("query-complete-a", hermes_home=str(tmp_path / "a"))
    entered = threading.Event()
    release = threading.Event()
    result_holder = {}
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: [],
    )

    def late_success(*args, **kwargs):
        del args, kwargs
        entered.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(provider, "_keyword_search", late_success)
    query_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            provider.handle_tool_call("cashew_query", {"query": "old-complete"}),
        )
    )
    query_thread.start()
    assert entered.wait(timeout=2.0)
    provider.shutdown()
    provider.initialize("query-complete-b", hermes_home=str(tmp_path / "b"))
    try:
        before = provider.health_status()
        release.set()
        query_thread.join(timeout=2.0)
        assert not query_thread.is_alive()
        assert json.loads(result_holder["result"])["ok"] is True
        after = provider.health_status()
        assert after["generation"] == before["generation"]
        assert after["tools"]["query_completed"] == 0
        assert after["reason_code"] == before["reason_code"]
    finally:
        release.set()
        provider.shutdown()


def test_late_query_fallback_success_cannot_reverse_completed_shutdown(
    tmp_path, monkeypatch
):
    provider = CashewMemoryProvider()
    provider.initialize("query-terminal", hermes_home=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    result_holder = {}
    monkeypatch.setattr(
        "core.retrieval.retrieve_recursive_bfs",
        lambda *args, **kwargs: [],
    )

    def blocked_fallback(*args, **kwargs):
        del args, kwargs
        entered.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(provider, "_keyword_search", blocked_fallback)
    query_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", provider.handle_tool_call("cashew_query", {"query": "late"})
        )
    )
    query_thread.start()
    assert entered.wait(timeout=2.0)
    provider.shutdown()
    assert provider.health_status()["state"] == "stopped"
    release.set()
    query_thread.join(timeout=2.0)
    try:
        assert not query_thread.is_alive()
        assert json.loads(result_holder["result"])["ok"] is True
        status = provider.health_status()
        assert status["state"] == "stopped"
        assert status["reason_code"] == "shutdown_complete"
        assert status["tools"]["query_completed"] == 0
    finally:
        release.set()
        provider.shutdown()


def test_late_extract_failure_cannot_reverse_completed_shutdown(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.initialize("extract-terminal", hermes_home=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    result_holder = {}

    def blocked_extract(**kwargs):
        del kwargs
        entered.set()
        assert release.wait(timeout=2.0)
        raise RuntimeError("late extract backend")

    monkeypatch.setattr("core.session.end_session", blocked_extract)
    extract_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            provider.handle_tool_call(
                "cashew_extract",
                {"user_content": "old", "assistant_content": "turn"},
            ),
        )
    )
    extract_thread.start()
    assert entered.wait(timeout=2.0)
    provider.shutdown()
    assert provider.health_status()["state"] == "stopped"
    release.set()
    extract_thread.join(timeout=2.0)
    try:
        assert not extract_thread.is_alive()
        assert json.loads(result_holder["result"])["ok"] is False
        status = provider.health_status()
        assert status["state"] == "stopped"
        assert status["reason_code"] == "shutdown_complete"
        assert status["tools"]["extract_failed"] == 0
    finally:
        release.set()
        provider.shutdown()


def test_worker_failure_during_stopping_cannot_overwrite_timeout(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.save_config({"sync_queue_timeout": 0}, str(tmp_path))
    provider.initialize("worker-stopping", hermes_home=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    clear_entered = threading.Event()
    release_clear = threading.Event()

    def late_failure(turn):
        del turn
        entered.set()
        assert release.wait(timeout=2.0)
        raise RuntimeError("worker failed after shutdown admission")

    real_clear = provider._clear_runtime_state

    def delayed_clear(*args, **kwargs):
        clear_entered.set()
        assert release_clear.wait(timeout=2.0)
        return real_clear(*args, **kwargs)

    monkeypatch.setattr(provider, "_drain_once", late_failure)
    monkeypatch.setattr(provider, "_clear_runtime_state", delayed_clear)
    try:
        provider.sync_turn("user", "assistant")
        assert entered.wait(timeout=2.0)
        provider.shutdown()
        assert provider.health_status()["reason_code"] == "worker_timeout"
        release.set()
        assert clear_entered.wait(timeout=2.0)
        stopping = provider.health_status()
        assert stopping["state"] == "stopping"
        assert stopping["reason_code"] == "worker_timeout"
        release_clear.set()
        _wait_for(lambda: provider.health_status()["state"] == "stopped")
    finally:
        release.set()
        release_clear.set()
        provider.shutdown()


def test_health_publication_is_inside_worker_lifecycle_handoff(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    real_start = provider._start_sync_worker
    start_entered = threading.Event()
    release_start = threading.Event()

    def gated_start():
        real_start()
        start_entered.set()
        assert release_start.wait(timeout=2.0)

    monkeypatch.setattr(provider, "_start_sync_worker", gated_start)
    initializer = threading.Thread(
        target=provider.initialize,
        args=("publication",),
        kwargs={"hermes_home": str(tmp_path)},
    )
    shutdown = threading.Thread(target=provider.shutdown)
    initializer.start()
    assert start_entered.wait(timeout=2.0)
    shutdown.start()
    time.sleep(0.02)
    assert shutdown.is_alive()
    release_start.set()
    initializer.join(timeout=2.0)
    shutdown.join(timeout=2.0)
    assert not initializer.is_alive()
    assert not shutdown.is_alive()
    assert provider.health_status()["state"] == "stopped"


def test_sync_failure_is_accounted_without_changing_hot_path(monkeypatch, tmp_path):
    provider = CashewMemoryProvider()
    provider.initialize("sync-failure", hermes_home=str(tmp_path))

    class UnboundedBackendFailureError(Exception):
        pass

    def fail(turn):
        del turn
        raise UnboundedBackendFailureError("private backend detail")

    monkeypatch.setattr(provider, "_drain_once", fail)
    try:
        provider.sync_turn("user", "assistant")
        _wait_for(lambda: provider.health_status()["work"]["failed"] == 1)
        status = provider.health_status()
        assert status["work"]["accepted"] == 1
        assert status["work"]["failed"] == 1
        assert status["work"]["reconciled"] is True
        assert status["reason_code"] == "backend_error"
        assert status["last_error"]["class"] == "Exception"
        assert "private backend detail" not in json.dumps(status)
    finally:
        provider.shutdown()


def test_shutdown_timeout_keeps_inflight_outcome_visible(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.save_config({"sync_queue_timeout": 0}, str(tmp_path))
    provider.initialize("timeout-health", hermes_home=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()

    def blocked(turn):
        del turn
        entered.set()
        assert release.wait(timeout=2.0)
        return True

    monkeypatch.setattr(provider, "_drain_once", blocked)
    try:
        provider.sync_turn("user", "assistant")
        assert entered.wait(timeout=2.0)
        provider.shutdown()
        stopping = provider.health_status()
        assert stopping["state"] == "stopping"
        assert stopping["reason_code"] == "worker_timeout"
        assert stopping["work"]["in_flight"] == 1
        release.set()
        _wait_for(lambda: provider.health_status()["state"] == "stopped")
    finally:
        release.set()
        provider.shutdown()


def test_sync_queue_drop_and_rejection_preserve_reconciliation(tmp_path, monkeypatch):
    provider = CashewMemoryProvider()
    provider.initialize("sync-accounting", hermes_home=str(tmp_path))
    entered = threading.Event()
    release = threading.Event()

    def blocked(turn):
        del turn
        entered.set()
        assert release.wait(timeout=2.0)
        return True

    monkeypatch.setattr(provider, "_drain_once", blocked)
    try:
        provider.sync_turn("first", "assistant")
        assert entered.wait(timeout=2.0)
        for index in range(17):
            provider.sync_turn(f"queued-{index}", "assistant")

        with provider._sync_state_lock:
            provider._shutdown_started.set()
        provider.sync_turn("rejected", "assistant")
        with provider._sync_state_lock:
            provider._shutdown_started.clear()

        before_release = provider.health_status()["work"]
        assert before_release["accepted"] == 18
        assert before_release["dropped"] == 1
        assert before_release["rejected"] == 1
        assert before_release["in_flight"] == 1
        assert before_release["pending"] == 16
        assert before_release["reconciled"] is True

        release.set()
        _wait_for(lambda: provider.health_status()["work"]["completed"] == 17)
        after_release = provider.health_status()["work"]
        assert after_release["accepted"] == 18
        assert after_release["completed"] == 17
        assert after_release["dropped"] == 1
        assert after_release["rejected"] == 1
        assert after_release["reconciled"] is True
    finally:
        release.set()
        provider.shutdown()
