"""Embedding-device helpers shared by provider and sleep-cycle paths."""

from __future__ import annotations

DEFAULT_EMBEDDING_DEVICE = "cpu"


def normalize_embedding_device(device: object) -> str:
    """Return a stable SentenceTransformer device value.

    ``cpu`` is intentionally the fallback: automatic device selection can choose
    Apple MPS, whose intermittent native crashes cannot be caught by Python.
    """
    normalized = str(device or DEFAULT_EMBEDDING_DEVICE).strip().lower()
    return normalized or DEFAULT_EMBEDDING_DEVICE
