"""Boundary contracts for the upstream sleep implementation (#193)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from core.db import ensure_schema

from plugins.memory.cashew import sleep_adapter
from plugins.memory.cashew.locking import try_maintenance_lock

MODEL = "thenlper/gte-large"
DIMENSION = 1024


class _EmbeddingClient:
    dimension = DIMENSION

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), DIMENSION), dtype=np.float32)


def _brain(tmp_path: Path) -> Path:
    db_path = tmp_path / "brain.db"
    ensure_schema(str(db_path))
    with sqlite3.connect(db_path) as conn:
        for index, node_id in enumerate(("left", "right", "third")):
            conn.execute(
                "INSERT INTO thought_nodes "
                "(id, content, node_type, domain, timestamp, source_file, permanent) "
                "VALUES (?, ?, 'observation', 'test', '2026-09-12T00:00:00Z', ?, ?)",
                (node_id, f"memory {node_id}", f"source-{index}", 1 if index == 0 else 0),
            )
            vector = np.zeros(DIMENSION, dtype=np.float32)
            vector[0] = 1.0
            conn.execute(
                "INSERT INTO embeddings (node_id, vector, model, updated_at) "
                "VALUES (?, ?, ?, '2026-09-12T00:00:00Z')",
                (node_id, vector.tobytes(), MODEL),
            )
    return db_path


def test_adapter_delegates_real_cycle_and_preserves_backup(tmp_path: Path) -> None:
    db_path = _brain(tmp_path)
    backup = tmp_path / "profile-backup.json"
    backup.write_text("profile backup")
    with sqlite3.connect(db_path) as conn:
        before_embeddings = dict(
            conn.execute("SELECT node_id, length(vector) FROM embeddings")
        )

    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=2,
        max_edges=1,
        model_fn=None,
        embedding_model=MODEL,
        embedding_device="cpu",
        embedding_client=_EmbeddingClient(),
    )

    assert result["status"] in {"completed", "partial", "unavailable"}
    assert result["dream_generation"] == "skipped"
    assert result["nodes_selected"] <= 2
    assert backup.read_text() == "profile backup"
    with sqlite3.connect(db_path) as conn:
        assert dict(
            conn.execute("SELECT node_id, length(vector) FROM embeddings")
        ) == before_embeddings
        audit_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(decay_audit)")
        }
        assert {"node_id", "decay_reason", "decay_timestamp"} <= audit_columns
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'decay_audit'"
        ).fetchone()
        assert conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0] <= 2


def test_adapter_skips_when_maintenance_lease_is_held(tmp_path: Path) -> None:
    db_path = _brain(tmp_path)
    with try_maintenance_lock(db_path) as lease:
        assert lease is not None
        assert sleep_adapter.run_sleep_cycle(str(db_path), model_fn=None) == {}


def test_configured_model_and_profile_are_forwarded(monkeypatch: Any, tmp_path: Path) -> None:
    db_path = _brain(tmp_path)
    captured: dict[str, Any] = {}

    def fake_upstream(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "completed", "dream_generation": "skipped"}

    monkeypatch.setattr(sleep_adapter, "_upstream_run_sleep_cycle", fake_upstream)
    client = _EmbeddingClient()
    def model_fn(prompt: str) -> str:
        del prompt
        return "{}"
    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=7,
        max_edges=3,
        model_fn=model_fn,
        embedding_model=MODEL,
        embedding_device="cpu",
        embedding_client=client,
    )

    assert result["status"] == "completed"
    assert captured["model_fn"] is model_fn
    assert captured["embedding_model"] == MODEL
    assert captured["expected_dimension"] == DIMENSION
    assert captured["journal_policy"] == "preserve"
    assert captured["limit"] == 7
    assert captured["max_edges"] == 3
