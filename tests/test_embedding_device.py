"""Embedding process ownership, device selection, and upstream shim contracts."""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
import threading

import numpy as np
import pytest

import plugins.memory.cashew as cashew_module
from plugins.memory.cashew import (
    _UPSTREAM_COMPATIBILITY_SHIMS,
    CashewMemoryProvider,
    _bind_upstream_embedding,
    _GenerationBoundEmbeddingService,
)
from plugins.memory.cashew.embedding import (
    DEFAULT_EMBEDDING_DEVICE,
    normalize_embedding_device,
)
from plugins.memory.cashew.embedding_process import (
    EmbeddingFailure,
    EmbeddingSupervisor,
    EmbeddingUnavailable,
    NoInProcessEmbeddingBackend,
    ProcessEmbeddingBackend,
)


def test_normalize_embedding_device_defaults_to_cpu() -> None:
    assert normalize_embedding_device(None) == DEFAULT_EMBEDDING_DEVICE
    assert normalize_embedding_device("") == DEFAULT_EMBEDDING_DEVICE
    assert normalize_embedding_device(" MPS ") == "mps"


def test_embedding_configuration_routes_upstream_without_parent_fallback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.embedding_service

    class FakeSupervisor:
        dimension = 1024

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.started = False

        def start(self) -> int:
            self.started = True
            return self.dimension

        def serve_generation(self):
            from contextlib import nullcontext

            return nullcontext()

        def encode(
            self, texts: list[str], *, wait_timeout: float | None = None
        ) -> np.ndarray:
            raise EmbeddingUnavailable(EmbeddingFailure.TIMEOUT)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    supervisor = _bind_upstream_embedding(
        "thenlper/gte-large",
        "mps",
        cache_path=tmp_path / "embedding-cache.db",
    )
    assert isinstance(supervisor, FakeSupervisor)
    assert supervisor.started is True
    service = core.embedding_service._default_service
    assert isinstance(service._service.daemon, ProcessEmbeddingBackend)
    assert isinstance(service._service.local, NoInProcessEmbeddingBackend)
    with pytest.raises(EmbeddingUnavailable) as raised:
        service.embed_np(["no parent fallback"])
    assert raised.value.reason is EmbeddingFailure.TIMEOUT
    assert service.model == "thenlper/gte-large"
    assert service.dim == 1024


def test_upstream_cache_miss_uses_child_and_cache_hit_does_not(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.embedding_service

    class FakeSupervisor:
        dimension = 384

        def __init__(self, **kwargs: object) -> None:
            self.calls: list[list[str]] = []

        def start(self) -> int:
            return self.dimension

        def serve_generation(self):
            from contextlib import nullcontext

            return nullcontext()

        def encode(
            self, texts: list[str], *, wait_timeout: float | None = None
        ) -> np.ndarray:
            self.calls.append(texts)
            return np.ones((len(texts), self.dimension), dtype=np.float32)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    supervisor = _bind_upstream_embedding(
        "BAAI/bge-small-en-v1.5",
        "cpu",
        cache_path=tmp_path / "embedding-cache.db",
    )
    assert isinstance(supervisor, FakeSupervisor)
    service = core.embedding_service._default_service
    first = service.embed_np(["cached text"])
    second = service.embed_np(["cached text"])
    assert np.array_equal(first, second)
    assert supervisor.calls == [["cached text"]]


def test_rejected_second_provider_does_not_replace_active_upstream_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.config
    import core.embedding_service

    active_service = object()
    monkeypatch.setattr(core.config.config, "embedding_model", "active/model")
    monkeypatch.setattr(core.embedding_service, "_default_service", active_service)
    owner = EmbeddingSupervisor(
        model="active/model",
        device="cpu",
        dimension=4,
        cache_dir=tmp_path / "active-cache",
    )
    try:
        with pytest.raises(EmbeddingUnavailable) as raised:
            _bind_upstream_embedding(
                "second/model",
                "cpu",
                cache_path=tmp_path / "second-cache.db",
            )
        assert raised.value.reason is EmbeddingFailure.OWNED
        assert core.config.config.embedding_model == "active/model"
        assert core.embedding_service._default_service is active_service
    finally:
        owner.close()


def test_binding_requires_explicit_profile_cache_without_global_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.config
    import core.embedding_service

    service = object()
    monkeypatch.setattr(core.config.config, "embedding_model", "active/model")
    monkeypatch.setattr(core.embedding_service, "_default_service", service)
    with pytest.raises(ValueError, match="profile-scoped"):
        _bind_upstream_embedding("thenlper/gte-large", "cpu")
    assert core.config.config.embedding_model == "active/model"
    assert core.embedding_service._default_service is service


def test_closed_generation_rejects_cached_uncached_and_blank_service_paths(
    tmp_path,
) -> None:
    """Close must gate upstream shortcuts that never call a backend."""

    class RawService:
        model = "fake"
        dim = 4

        def embed_np(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 4), dtype=np.float32)

        def embed(self, text: str) -> list[float]:
            return [1.0, 1.0, 1.0, 1.0]

    supervisor = EmbeddingSupervisor(
        model="fake",
        device="cpu",
        dimension=4,
        cache_dir=tmp_path / "cache",
    )
    service = _GenerationBoundEmbeddingService(RawService(), supervisor)
    try:
        assert service.embed_np(["cached"]).shape == (1, 4)
        assert service.embed_np([]).shape == (0, 4)
        supervisor.close()
        for texts in (["cached"], ["uncached"], []):
            with pytest.raises(EmbeddingUnavailable) as raised:
                service.embed_np(texts)
            assert raised.value.reason is EmbeddingFailure.CLOSED
        with pytest.raises(EmbeddingUnavailable) as raised:
            service.embed("")
        assert raised.value.reason is EmbeddingFailure.CLOSED
    finally:
        supervisor.close()


def test_failed_child_handshake_publishes_no_upstream_global_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.config
    import core.embedding_service

    class FailingSupervisor:
        def __init__(self, **_kwargs: object) -> None:
            self.dimension = 0

        def start(self) -> int:
            raise EmbeddingUnavailable(EmbeddingFailure.STARTUP)

        def close(self) -> None:
            return None

    sentinel = object()
    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FailingSupervisor)
    monkeypatch.setattr(core.config.config, "embedding_model", "active/model")
    monkeypatch.setattr(core.embedding_service, "_default_service", sentinel)
    with pytest.raises(EmbeddingUnavailable) as raised:
        _bind_upstream_embedding(
            "unknown/model", "cpu", cache_path=tmp_path / "embedding-cache.db"
        )
    assert raised.value.reason is EmbeddingFailure.STARTUP
    assert core.config.config.embedding_model == "active/model"
    assert core.embedding_service._default_service is sentinel
    assert "unknown/model" not in core.embedding_service._KNOWN_DIMS


def test_unknown_model_dimension_is_published_only_after_child_handshake(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.embedding_service

    class VerifiedSupervisor:
        dimension = 7

        def __init__(self, **_kwargs: object) -> None:
            self.started = False

        def start(self) -> int:
            self.started = True
            return self.dimension

        def serve_generation(self):
            from contextlib import nullcontext

            return nullcontext()

        def close(self) -> None:
            return None

    model = "example/child-verified-model"
    monkeypatch.delitem(core.embedding_service._KNOWN_DIMS, model, raising=False)
    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", VerifiedSupervisor)
    supervisor = _bind_upstream_embedding(
        model, "cpu", cache_path=tmp_path / "embedding-cache.db"
    )
    assert isinstance(supervisor, VerifiedSupervisor)
    assert supervisor.started is True
    assert core.embedding_service._KNOWN_DIMS[model] == 7


def test_same_model_profiles_keep_distinct_cache_and_late_close_cannot_clobber_binding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late old-generation teardown cannot overwrite the published service."""
    import core.embedding_service

    release_old_close = threading.Event()
    old_close_started = threading.Event()

    class FakeSupervisor:
        dimension = 384
        instances: list["FakeSupervisor"] = []

        def __init__(self, **_kwargs: object) -> None:
            self.index = len(self.instances)
            self.instances.append(self)

        def start(self) -> int:
            return self.dimension

        def serve_generation(self):
            from contextlib import nullcontext

            return nullcontext()

        def close(self) -> None:
            if self.index == 0:
                old_close_started.set()
                release_old_close.wait(timeout=2)

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    old = _bind_upstream_embedding(
        "thenlper/gte-small", "cpu", cache_path=tmp_path / "one" / "cache.db"
    )
    assert isinstance(old, FakeSupervisor)
    old_service = core.embedding_service._default_service
    closer = threading.Thread(target=old.close)
    closer.start()
    assert old_close_started.wait(timeout=1)
    newer = _bind_upstream_embedding(
        "thenlper/gte-small", "cpu", cache_path=tmp_path / "two" / "cache.db"
    )
    assert isinstance(newer, FakeSupervisor)
    new_service = core.embedding_service._default_service
    assert new_service is not old_service
    assert new_service._service.cache.path != old_service._service.cache.path
    release_old_close.set()
    closer.join(timeout=1)
    assert not closer.is_alive()
    assert core.embedding_service._default_service is new_service


def test_same_model_profiles_do_not_cross_hit_the_upstream_embedding_cache(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Profile-scoped upstream caches stay separate even for one model name."""
    import core.embedding_service

    class FakeSupervisor:
        instances: list["FakeSupervisor"] = []

        def __init__(self, *, dimension: int, **_kwargs: object) -> None:
            self.dimension = dimension
            self.calls: list[list[str]] = []
            self.instances.append(self)

        def start(self) -> int:
            return self.dimension

        def serve_generation(self):
            from contextlib import nullcontext

            return nullcontext()

        def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
            self.calls.append(texts)
            return np.full(
                (len(texts), self.dimension), len(self.instances), np.float32
            )

        def _when_closed(self, callback) -> None:
            callback()

        def close(self, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    first = CashewMemoryProvider()
    second = CashewMemoryProvider()
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    first.save_config({"embedding_model": "thenlper/gte-small"}, str(home_a))
    second.save_config({"embedding_model": "thenlper/gte-small"}, str(home_b))
    try:
        first.initialize("a", hermes_home=str(home_a))
        service_a = core.embedding_service._default_service
        service_a.embed_np(["same text"])
        assert FakeSupervisor.instances[0].calls == [["same text"]]
        first.shutdown()

        second.initialize("b", hermes_home=str(home_b))
        service_b = core.embedding_service._default_service
        assert service_b._service.cache.path != service_a._service.cache.path
        service_b.embed_np(["same text"])
        assert FakeSupervisor.instances[1].calls == [["same text"]]
    finally:
        first.shutdown()
        second.shutdown()


def test_pinned_upstream_shims_are_bounded_and_have_retirement_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the dd57 compatibility seam against a return to class patching."""
    import core.config
    import core.embedding_service

    monkeypatch.setattr(core.config.config, "embedding_model", "thenlper/gte-large")

    assert _UPSTREAM_COMPATIBILITY_SHIMS == (
        "core.config.config.embedding_model",
        "core.embedding_service._default_service",
        "core.embedding_service._KNOWN_DIMS[model]",
    )
    source = inspect.getsource(cashew_module._bind_upstream_embedding)
    assert "reset_default_service" not in source
    assert "DEFAULT_MODEL" not in source
    assert "EMBEDDING_DIM" not in source
    assert "LocalBackend" not in source
    assert "DaemonBackend" not in source
    # The broad offline fixture intentionally replaces this production entry
    # point. Reload just the pinned module for its source-level contract.
    core_embeddings = importlib.reload(importlib.import_module("core.embeddings"))
    migration_script = importlib.reload(
        importlib.import_module("scripts.migrate_embeddings")
    )

    def calls(function) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(
                ast.parse(textwrap.dedent(inspect.getsource(function)))
            )
            if isinstance(node, ast.Call)
        ]

    def called_name(call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ""

    # The pinned dd57 call graph requires the three documented seams.  These
    # are AST checks so a comment or unrelated helper cannot satisfy them.
    assert "get_default_service" in {
        called_name(call) for call in calls(core_embeddings.embed_nodes)
    }
    migration_calls = calls(migration_script.detect_mismatch) + calls(
        migration_script.migrate_embeddings
    )
    assert any(
        called_name(call) == "resolve_embedding_dim"
        and not call.args
        and not call.keywords
        for call in migration_calls
    )
    assert "_resolve_default_model" in {called_name(call) for call in migration_calls}
    resolver_tree = ast.parse(
        textwrap.dedent(inspect.getsource(core.embedding_service.resolve_embedding_dim))
    )
    assert any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "LocalBackend"
        and node.attr == "dim"
        for node in ast.walk(resolver_tree)
    )

    def assigned_target(target: ast.expr) -> str | None:
        def dotted(node: ast.expr) -> str | None:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                parent = dotted(node.value)
                return f"{parent}.{node.attr}" if parent else None
            return None

        if isinstance(target, ast.Attribute):
            return dotted(target)
        if isinstance(target, ast.Subscript):
            base = dotted(target.value)
            return f"{base}[model]" if base else None
        return None

    shim_writes = [
        assigned_target(target)
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if assigned_target(target) is not None
    ]
    assert len(shim_writes) == 3
    assert set(shim_writes) == {
        "core.config.config.embedding_model",
        "core.embedding_service._KNOWN_DIMS[model]",
        "core.embedding_service._default_service",
    }
