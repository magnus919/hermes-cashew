"""Embedding-device helpers shared by provider and sleep-cycle paths."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DEVICE = "cpu"


def normalize_embedding_device(device: object) -> str:
    """Return a stable SentenceTransformer device value.

    ``cpu`` is intentionally the fallback: automatic device selection can choose
    Apple MPS, whose intermittent native crashes cannot be caught by Python.
    """
    normalized = str(device or DEFAULT_EMBEDDING_DEVICE).strip().lower()
    return normalized or DEFAULT_EMBEDDING_DEVICE


def load_sentence_transformer(model_name: str, device: object) -> Any:
    """Load a SentenceTransformer on *device*, retrying initialization on CPU.

    ``auto`` preserves SentenceTransformer's own device selection. Any explicit
    non-CPU device that fails during construction gets one safe CPU retry.
    """
    from sentence_transformers import SentenceTransformer

    selected = normalize_embedding_device(device)
    kwargs = {} if selected == "auto" else {"device": selected}
    try:
        return SentenceTransformer(model_name, **kwargs)
    except Exception:
        if selected == DEFAULT_EMBEDDING_DEVICE:
            raise
        logger.warning(
            "failed to initialize embedding model %s on device %s; retrying on CPU",
            model_name,
            selected,
            exc_info=True,
        )
        return SentenceTransformer(model_name, device=DEFAULT_EMBEDDING_DEVICE)
