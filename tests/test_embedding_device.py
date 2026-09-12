"""Embedding device selection and CPU fallback contracts."""

from __future__ import annotations

import numpy as np
import pytest

import plugins.memory.cashew as cashew_module
from plugins.memory.cashew import _patch_upstream_embedding
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

        def encode(
            self, texts: list[str], *, wait_timeout: float | None = None
        ) -> np.ndarray:
            raise EmbeddingUnavailable(EmbeddingFailure.TIMEOUT)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    supervisor = _patch_upstream_embedding(
        "thenlper/gte-large",
        "mps",
        cache_path=tmp_path / "embedding-cache.db",
    )
    assert isinstance(supervisor, FakeSupervisor)
    service = core.embedding_service._default_service
    assert isinstance(service.daemon, ProcessEmbeddingBackend)
    assert isinstance(service.local, NoInProcessEmbeddingBackend)
    with pytest.raises(EmbeddingUnavailable) as raised:
        service.embed_np(["no parent fallback"])
    assert raised.value.reason is EmbeddingFailure.TIMEOUT
    assert core.embedding_service.DEFAULT_MODEL == "thenlper/gte-large"
    assert core.embedding_service.EMBEDDING_DIM == 1024


def test_upstream_cache_miss_uses_child_and_cache_hit_does_not(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.embedding_service

    class FakeSupervisor:
        dimension = 384

        def __init__(self, **kwargs: object) -> None:
            self.calls: list[list[str]] = []

        def encode(
            self, texts: list[str], *, wait_timeout: float | None = None
        ) -> np.ndarray:
            self.calls.append(texts)
            return np.ones((len(texts), self.dimension), dtype=np.float32)

        def close(self) -> None:
            return None

    monkeypatch.setattr(cashew_module, "EmbeddingSupervisor", FakeSupervisor)
    supervisor = _patch_upstream_embedding(
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
    monkeypatch.setattr(core.embedding_service, "DEFAULT_MODEL", "active/model")
    monkeypatch.setattr(core.embedding_service, "EMBEDDING_DIM", 4)
    monkeypatch.setattr(core.embedding_service, "_default_service", active_service)
    owner = EmbeddingSupervisor(
        model="active/model",
        device="cpu",
        dimension=4,
        cache_dir=tmp_path / "active-cache",
    )
    try:
        with pytest.raises(EmbeddingUnavailable) as raised:
            _patch_upstream_embedding(
                "second/model",
                "cpu",
                cache_path=tmp_path / "second-cache.db",
            )
        assert raised.value.reason is EmbeddingFailure.OWNED
        assert core.config.config.embedding_model == "active/model"
        assert core.embedding_service.DEFAULT_MODEL == "active/model"
        assert core.embedding_service.EMBEDDING_DIM == 4
        assert core.embedding_service._default_service is active_service
    finally:
        owner.close()


def test_repeated_compatibility_configuration_resets_default_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.config
    import core.embedding_service

    first_service = object()
    monkeypatch.setattr(core.embedding_service, "_default_service", first_service)
    _patch_upstream_embedding("thenlper/gte-large", "cpu")
    assert core.config.config.embedding_model == "thenlper/gte-large"
    assert core.embedding_service.DEFAULT_MODEL == "thenlper/gte-large"
    assert core.embedding_service.EMBEDDING_DIM == 1024
    assert core.embedding_service._default_service is None

    second_service = object()
    monkeypatch.setattr(core.embedding_service, "_default_service", second_service)
    _patch_upstream_embedding("thenlper/gte-large", "cpu")
    assert core.embedding_service._default_service is None
