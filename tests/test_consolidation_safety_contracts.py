"""Boundary contracts for the upstream sleep implementation (#193)."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from core.db import ensure_schema

from plugins.memory.cashew import sleep_adapter, sleep_refactor
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
            if index == 0:
                vector[0] = 1.0
            else:
                vector[0] = 0.92
                vector[index] = np.sqrt(1.0 - 0.92**2)
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
    assert result["cross_link_directed_rows"] == result["cross_links_created"] * 2
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


def test_configured_model_fn_runs_for_real_cross_source_pair(tmp_path: Path) -> None:
    db_path = _brain(tmp_path)
    calls: list[str] = []

    def model_fn(prompt: str) -> str:
        calls.append(prompt)
        return "A shared invariant connects these two memories."

    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=3,
        max_edges=2,
        model_fn=model_fn,
        embedding_model=MODEL,
        embedding_client=_EmbeddingClient(),
    )

    assert result["dream_generation"] == "ran"
    assert calls and "SNIPPET A" in calls[0]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM thought_nodes WHERE node_type = 'dream'"
        ).fetchone()[0] == 1


def test_gc_audit_and_permanence_are_preserved_at_upstream_boundary(
    tmp_path: Path,
) -> None:
    db_path = _brain(tmp_path)
    with sqlite3.connect(db_path) as conn:
        for index in range(55):
            vector = np.zeros(DIMENSION, dtype=np.float32)
            vector[10 + index] = 1.0
            conn.execute(
                "INSERT INTO thought_nodes "
                "(id, content, node_type, domain, timestamp, source_file, permanent) "
                "VALUES (?, 'old', 'observation', 'test', '2020-01-01', ?, ?)",
                (f"stale-{index}", f"stale-{index}", 1 if index == 0 else 0),
            )
            conn.execute(
                "INSERT INTO embeddings (node_id, vector, model, updated_at) "
                "VALUES (?, ?, ?, '2020-01-01')",
                (f"stale-{index}", vector.tobytes(), MODEL),
            )

    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=60,
        model_fn=None,
        embedding_model=MODEL,
        embedding_client=_EmbeddingClient(),
    )

    assert result["nodes_gc_decayed"] == 50
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT decayed FROM thought_nodes WHERE id = 'stale-0'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM thought_nodes "
            "WHERE id LIKE 'stale-%' AND decayed = 1"
        ).fetchone()[0] == 50
        assert conn.execute("SELECT COUNT(*) FROM decay_audit").fetchone()[0] == 50


def test_real_vec_rows_survive_upstream_cycle_and_malformed_row_is_ignored(
    tmp_path: Path,
) -> None:
    sqlite_vec = __import__("sqlite_vec")
    db_path = _brain(tmp_path)
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[0] = 1.0
    with sqlite3.connect(db_path) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
            f"node_id TEXT PRIMARY KEY, embedding float[{DIMENSION}] distance_metric=cosine)"
        )
        conn.execute(
            "INSERT INTO vec_embeddings (node_id, embedding) VALUES (?, ?)",
            ("left", vector.tobytes()),
        )
        conn.execute(
            "INSERT INTO embeddings (node_id, vector, model, updated_at) "
            "VALUES ('malformed', ?, ?, '2026-09-12T00:00:00Z')",
            (b"bad", MODEL),
        )
    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=3,
        model_fn=None,
        embedding_model=MODEL,
        embedding_client=_EmbeddingClient(),
    )
    assert result["status"] in {"completed", "partial", "unavailable"}
    with sqlite3.connect(db_path) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        assert {
            row[0] for row in conn.execute("SELECT node_id FROM vec_embeddings")
        } == {"left", "right", "third"}
        assert conn.execute(
            "SELECT vector FROM embeddings WHERE node_id = 'malformed'"
        ).fetchone()[0] == b"bad"


def test_upstream_failure_degrades_silently_and_historical_import_keeps_contract(
    monkeypatch: Any, tmp_path: Path
) -> None:
    assert sleep_refactor.run_sleep_cycle is sleep_adapter.run_sleep_cycle
    assert set(inspect.signature(sleep_refactor.run_sleep_cycle).parameters) >= {
        "db_path",
        "model_fn",
        "embedding_client",
    }
    db_path = _brain(tmp_path)

    def fail(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(sleep_adapter, "_upstream_run_sleep_cycle", fail)
    assert sleep_refactor.run_sleep_cycle(str(db_path), model_fn=None) == {}


def test_background_dream_is_rejected_until_hermes_can_retain_its_lease(
    tmp_path: Path,
) -> None:
    result = sleep_adapter.run_sleep_cycle(str(_brain(tmp_path)), background_dream=True)
    assert result == {"status": "rejected", "error": "background_dream_unsupported"}


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
