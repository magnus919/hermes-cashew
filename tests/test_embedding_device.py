"""Embedding device selection and CPU fallback contracts."""

from __future__ import annotations

import sys

import numpy as np

from plugins.memory.cashew import _patch_upstream_embedding
from plugins.memory.cashew.embedding import (
    DEFAULT_EMBEDDING_DEVICE,
    normalize_embedding_device,
)


def test_normalize_embedding_device_defaults_to_cpu() -> None:
    assert normalize_embedding_device(None) == DEFAULT_EMBEDDING_DEVICE
    assert normalize_embedding_device("") == DEFAULT_EMBEDDING_DEVICE
    assert normalize_embedding_device(" MPS ") == "mps"


def test_explicit_device_bypasses_daemon_and_retries_initialization_on_cpu(
    monkeypatch,
) -> None:
    import core.embedding_service

    attempts: list[str | None] = []

    class RecordingSentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            attempts.append(device)
            if device == "mps":
                raise RuntimeError("simulated MPS initialization failure")
            self.model_name = model_name

        def get_sentence_embedding_dimension(self) -> int:
            return 1024

        def encode(self, texts, **kwargs) -> np.ndarray:
            return np.ones((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(
        sys.modules["sentence_transformers"],
        "SentenceTransformer",
        RecordingSentenceTransformer,
    )

    try:
        _patch_upstream_embedding("thenlper/gte-large", "mps")
        backend = core.embedding_service.LocalBackend("thenlper/gte-large")
        vectors = backend.encode(["hello"])
        daemon = core.embedding_service.DaemonBackend(model_name="thenlper/gte-large")

        assert attempts == ["mps", "cpu"]
        assert vectors.shape == (1, 1024)
        assert daemon.encode(["hello"]).shape == (0, 1024)
    finally:
        _patch_upstream_embedding("thenlper/gte-large", DEFAULT_EMBEDDING_DEVICE)


def test_upstream_device_patch_is_idempotent() -> None:
    import core.embedding_service

    _patch_upstream_embedding("thenlper/gte-large", "cpu")
    ensure_model = core.embedding_service.LocalBackend._ensure_model
    daemon_encode = core.embedding_service.DaemonBackend.encode

    _patch_upstream_embedding("thenlper/gte-large", "cpu")

    assert core.embedding_service.LocalBackend._ensure_model is ensure_model
    assert core.embedding_service.DaemonBackend.encode is daemon_encode
