"""Compatibility boundary for the pinned Cashew embedding service.

The provider owns profile lifecycle and admission.  This module owns the
small amount of adapter code needed to present a profile-scoped embedding
service and cache to the pinned upstream Cashew release.  The three upstream
global assignments remain deliberately visible in the provider's binding
call; they can be removed only after the upstream replacement gate in #193.
"""

from __future__ import annotations

import pathlib
import sqlite3
from typing import TYPE_CHECKING, Any, cast

from .admission import OperationAdmissionError, admit_operation, current_admission

if TYPE_CHECKING:
    from .embedding_process import EmbeddingSupervisor

__all__ = [
    "UPSTREAM_KNOWN_DIMS",
    "UPSTREAM_COMPATIBILITY_SHIMS",
    "NoopEmbeddingCache",
    "GenerationBoundEmbeddingCache",
    "GenerationBoundEmbeddingService",
]


UPSTREAM_KNOWN_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "thenlper/gte-large": 1024,
    "thenlper/gte-base": 768,
    "thenlper/gte-small": 384,
    "all-mpnet-base-v2": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}

# These are the only private compatibility seams retained by the adapter for
# cashew-brain at the pinned composite source. See the provider binding call for
# the publication ordering and #193 for the upstream replacement gate.
UPSTREAM_COMPATIBILITY_SHIMS = (
    "core.config.config.embedding_model",
    "core.embedding_service._default_service",
    "core.embedding_service._KNOWN_DIMS[model]",
)


class NoopEmbeddingCache:
    """Exact cache surface used when the affected SQLite runtime is unsafe."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = str(path)

    def get_many(self, model: str, texts: list[str]) -> list[None]:
        del model
        return [None for _ in texts]

    def put_many(self, model: str, pairs: list[tuple[str, Any]]) -> int:
        del model, pairs
        return 0

    def get(self, model: str, text: str) -> None:
        del model, text
        return None

    def put(self, model: str, text: str, vector: Any) -> None:
        del model, text, vector

    def size(self, model: str | None = None) -> int:
        del model
        return 0

    def invalidate_model(self, model: str) -> None:
        del model


class GenerationBoundEmbeddingCache:
    """Reuse the admission-owned cache lease without opening another handle."""

    def __init__(
        self,
        cache: Any,
        *,
        path: pathlib.Path,
        model: str,
        embedding_dim: int,
        supervisor: Any,
        generation: str | int | None,
    ) -> None:
        self._cache = cache
        self.path = str(path)
        self._path = path.resolve(strict=False)
        self._model = model
        self._embedding_dim = embedding_dim
        self._supervisor = supervisor
        self._generation = generation

    def _check(self, model: str) -> None:
        admission = current_admission()
        if admission is None:
            raise OperationAdmissionError("embedding cache operation is not admitted")
        if isinstance(self._cache, NoopEmbeddingCache):
            # Affected-runtime profiles deliberately carry no cache lease; the
            # facade remains a harmless exact no-op for upstream calls.
            if model != self._model:
                raise OperationAdmissionError(
                    "embedding cache admission model mismatch"
                )
            return
        if admission.cache_path != self._path:
            raise OperationAdmissionError("embedding cache admission path mismatch")
        if admission.model != model or admission.model != self._model:
            raise OperationAdmissionError("embedding cache admission model mismatch")
        if admission.embedding_dim not in (None, self._embedding_dim):
            raise OperationAdmissionError(
                "embedding cache admission dimension mismatch"
            )
        if admission.supervisor is not self._supervisor:
            raise OperationAdmissionError(
                "embedding cache admission supervisor mismatch"
            )
        if admission.embedding_generation != self._generation:
            raise OperationAdmissionError(
                "embedding cache admission generation mismatch"
            )
        if admission.cache_lease is None:
            raise OperationAdmissionError("embedding cache lease is not owned")
        # The child handshake is the source of truth for dimensions.  Keep the
        # per-model cache record in the same file scoped to this admission so a
        # stale facade cannot publish a vector before identity is durable.
        try:
            with sqlite3.connect(str(self._path)) as conn:
                row = conn.execute(
                    "SELECT embedding_dim FROM hermes_cashew_cache_meta WHERE model=?",
                    (model,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise OperationAdmissionError(
                "embedding cache identity metadata is unavailable"
            ) from exc
        if row is None or int(row[0]) != self._embedding_dim:
            raise OperationAdmissionError(
                "embedding cache model dimension metadata mismatch"
            )

    def get_many(self, model: str, texts: list[str]) -> Any:
        self._check(model)
        return self._cache.get_many(model, texts)

    def put_many(self, model: str, pairs: list[tuple[str, Any]]) -> Any:
        self._check(model)
        return self._cache.put_many(model, pairs)

    def get(self, model: str, text: str) -> Any:
        self._check(model)
        return self._cache.get(model, text)

    def put(self, model: str, text: str, vector: Any) -> Any:
        self._check(model)
        return self._cache.put(model, text, vector)

    def size(self, model: str | None = None) -> Any:
        selected = model or self._model
        self._check(selected)
        return self._cache.size(selected)

    def invalidate_model(self, model: str) -> Any:
        self._check(model)
        return self._cache.invalidate_model(model)


class GenerationBoundEmbeddingService:
    """Make one upstream service unavailable once its owner closes.

    Upstream's service returns cache hits and zero vectors without consulting a
    backend.  The outer generation gate prevents those convenience paths from
    letting a closed profile serve another profile's global singleton.
    """

    def __init__(
        self,
        service: Any,
        supervisor: "EmbeddingSupervisor",
        *,
        cache_path: pathlib.Path | None = None,
        model: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self._service = service
        self._supervisor = supervisor
        self._cache_path = cache_path
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return cast(str, self._service.model)

    @property
    def dim(self) -> int:
        return cast(int, self._service.dim)

    def embed_np(self, texts: list[str]) -> Any:
        admission = current_admission()
        if admission is None and self._cache_path is not None:
            with admit_operation(
                cache_path=self._cache_path,
                model=self._model,
                embedding_dim=self._dimension,
                supervisor=self._supervisor,
                embedding_generation=getattr(self._supervisor, "generation", None),
                cache_exclusive=True,
                deadline=1.5,
            ):
                with self._supervisor.serve_generation():
                    return self._service.embed_np(texts)
        with self._supervisor.serve_generation():
            return self._service.embed_np(texts)

    def embed(self, text: Any) -> Any:
        """Keep upstream's public single-text route behind the same gate."""
        admission = current_admission()
        if admission is None and self._cache_path is not None:
            with admit_operation(
                cache_path=self._cache_path,
                model=self._model,
                embedding_dim=self._dimension,
                supervisor=self._supervisor,
                embedding_generation=getattr(self._supervisor, "generation", None),
                cache_exclusive=True,
                deadline=1.5,
            ):
                with self._supervisor.serve_generation():
                    return self._service.embed(text)
        with self._supervisor.serve_generation():
            return self._service.embed(text)
