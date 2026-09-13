"""Boundary contracts for the upstream sleep implementation (#193)."""

from __future__ import annotations

import hashlib
import inspect
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from core.backup import create_backup
from core.db import ensure_schema

from plugins.memory.cashew import sleep_adapter, sleep_refactor
from plugins.memory.cashew.config import resolve_db_path
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


def _snapshot(db_path: Path) -> dict[str, Any]:
    sqlite_vec = __import__("sqlite_vec")
    with sqlite3.connect(db_path) as conn:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return {
            "nodes": conn.execute(
                "SELECT id, content, node_type, domain, timestamp, access_count, "
                "last_accessed, source_file, decayed, metadata, permanent, tags "
                "FROM thought_nodes ORDER BY id"
            ).fetchall(),
            "meta": conn.execute(
                "SELECT key, value FROM hermes_provider_meta ORDER BY key"
            ).fetchall(),
            "embeddings": conn.execute(
                "SELECT node_id, vector, model, updated_at FROM embeddings ORDER BY node_id"
            ).fetchall(),
            "vec": conn.execute(
                "SELECT node_id, embedding FROM vec_embeddings ORDER BY node_id"
            ).fetchall(),
        }


def test_adapter_preserves_profile_backup_and_before_after_identity(
    tmp_path: Path,
) -> None:
    db_path = resolve_db_path(tmp_path, "cashew/brain.db")
    db_path.parent.mkdir(parents=True)
    ensure_schema(str(db_path))
    sqlite_vec = __import__("sqlite_vec")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hermes_provider_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO hermes_provider_meta (key, value) VALUES (?, ?)",
            [("embedding_model", MODEL), ("embedding_dim", str(DIMENSION)),
             ("vec_dim", str(DIMENSION)), ("maintenance_epoch", "1")],
        )
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
            f"node_id TEXT PRIMARY KEY, embedding float[{DIMENSION}] distance_metric=cosine)"
        )
        for index, node_id in enumerate(("left", "right", "third")):
            vector = np.zeros(DIMENSION, dtype=np.float32)
            if index == 0:
                vector[0] = 1.0
            else:
                vector[0] = 0.92
                vector[index] = np.sqrt(1.0 - 0.92**2)
            conn.execute(
                "INSERT INTO thought_nodes "
                "(id, content, node_type, domain, timestamp, source_file, permanent) "
                "VALUES (?, ?, 'observation', 'test', '2026-09-12T00:00:00Z', ?, ?)",
                (node_id, f"memory {node_id}", f"source-{index}", 1 if index == 0 else 0),
            )
            blob = vector.tobytes()
            conn.execute(
                "INSERT INTO embeddings (node_id, vector, model, updated_at) "
                "VALUES (?, ?, ?, '2026-09-12T00:00:00Z')",
                (node_id, blob, MODEL),
            )
            conn.execute(
                "INSERT INTO vec_embeddings (node_id, embedding) VALUES (?, ?)",
                (node_id, blob),
            )
    # Cashew, via core.backup.create_backup, owns this pre-change backup.
    backup_path = Path(create_backup(str(db_path), str(db_path.parent / "backups")))
    before = _snapshot(db_path)
    assert db_path == tmp_path / "cashew" / "brain.db"
    assert dict(before["meta"])["embedding_model"] == MODEL
    backup_hash = hashlib.sha256(backup_path.read_bytes()).hexdigest()

    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=3,
        max_edges=1,
        model_fn=None,
        embedding_model=MODEL,
        embedding_device="cpu",
        embedding_client=_EmbeddingClient(),
    )

    assert result["status"] in {"completed", "partial", "unavailable"}
    assert result["dream_generation"] == "skipped"
    assert result["nodes_selected"] <= 3
    assert result["cross_link_directed_rows"] == result["cross_links_created"] * 2
    after = _snapshot(db_path)
    before_nodes = {row[0]: row for row in before["nodes"]}
    after_nodes = {row[0]: row for row in after["nodes"]}
    assert set(after_nodes) == set(before_nodes)
    for node_id, before_row in before_nodes.items():
        after_row = after_nodes[node_id]
        # Upstream may promote a core memory during the cycle. Every other
        # persisted node field remains part of this before/after invariant.
        assert after_row[0:2] == before_row[0:2]
        assert after_row[3:] == before_row[3:]
        assert after_row[2] in {before_row[2], "core_memory"}
    assert after["meta"] == before["meta"]
    assert after["embeddings"] == before["embeddings"]
    assert after["vec"] == before["vec"]
    assert hashlib.sha256(backup_path.read_bytes()).hexdigest() == backup_hash
    assert backup_path.parent == db_path.parent / "backups"
    with sqlite3.connect(db_path) as conn:
        audit_columns = {row[1] for row in conn.execute("PRAGMA table_info(decay_audit)")}
        assert {"node_id", "decay_reason", "decay_timestamp"} <= audit_columns


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


def test_cross_link_cap_reports_truthful_persisted_directed_rows(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cap.db"
    ensure_schema(str(db_path))
    with sqlite3.connect(db_path) as conn:
        vectors = [("a", 0, None), ("b", 0, 1), ("c", 2, None), ("d", 2, 3)]
        for index, (node_id, axis, extra) in enumerate(vectors):
            vector = np.zeros(DIMENSION, dtype=np.float32)
            vector[axis] = 1.0 if extra is None else 0.92
            if extra is not None:
                vector[extra] = np.sqrt(1.0 - 0.92**2)
            conn.execute(
                "INSERT INTO thought_nodes "
                "(id, content, node_type, domain, timestamp, source_file) "
                "VALUES (?, ?, 'observation', 'test', '2026-09-12T00:00:00Z', ?)",
                (node_id, f"cap memory {node_id}", f"cap-source-{index}"),
            )
            conn.execute(
                "INSERT INTO embeddings (node_id, vector, model, updated_at) "
                "VALUES (?, ?, ?, '2026-09-12T00:00:00Z')",
                (node_id, vector.tobytes(), MODEL),
            )

    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=4,
        max_edges=1,
        model_fn=None,
        embedding_model=MODEL,
        embedding_client=_EmbeddingClient(),
    )

    assert result["cross_link_candidates"] >= 2
    assert result["cross_link_capped"] is True
    assert result["cross_links_created"] == 1
    assert result["cross_link_directed_rows"] == 2
    with sqlite3.connect(db_path) as conn:
        persisted = conn.execute(
            "SELECT COUNT(*) FROM derivation_edges WHERE reasoning LIKE 'cross%'"
        ).fetchone()[0]
    assert persisted == result["cross_link_directed_rows"]
    assert persisted <= 2


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
            "INSERT INTO thought_nodes "
            "(id, content, node_type, domain, timestamp, source_file) "
            "VALUES ('malformed', 'active malformed blob', 'observation', 'test', "
            "'2026-09-12T00:00:00Z', 'malformed')"
        )
        conn.execute(
            "INSERT INTO embeddings (node_id, vector, model, updated_at) "
            "VALUES ('malformed', ?, ?, '2026-09-12T00:00:00Z')",
            (b"bad", MODEL),
        )
        for node_id, blob in (
            ("nan-node", np.full(DIMENSION, np.nan, dtype=np.float32).tobytes()),
            ("wrong-dimension", np.ones(2, dtype=np.float32).tobytes()),
            ("zero-node", np.zeros(DIMENSION, dtype=np.float32).tobytes()),
        ):
            conn.execute(
                "INSERT INTO thought_nodes "
                "(id, content, node_type, domain, timestamp, source_file) "
                "VALUES (?, ?, 'observation', 'test', '2026-09-12T00:00:00Z', ?)",
                (node_id, f"active malformed {node_id}", node_id),
            )
            conn.execute(
                "INSERT INTO embeddings (node_id, vector, model, updated_at) "
                "VALUES (?, ?, ?, '2026-09-12T00:00:00Z')",
                (node_id, blob, MODEL),
            )
    result = sleep_adapter.run_sleep_cycle(
        str(db_path),
        limit=10,
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
        assert {
            row[0]
            for row in conn.execute(
                "SELECT node_id FROM embeddings WHERE node_id IN "
                "('left', 'right', 'third')"
            )
        } == {"left", "right", "third"}
        assert conn.execute(
            "SELECT vector FROM embeddings WHERE node_id = 'malformed'"
        ).fetchone()[0] == b"bad"
        assert conn.execute(
            "SELECT vector FROM embeddings WHERE node_id = 'nan-node'"
        ).fetchone()[0] == np.full(DIMENSION, np.nan, dtype=np.float32).tobytes()
        assert conn.execute(
            "SELECT length(vector) FROM embeddings WHERE node_id = 'wrong-dimension'"
        ).fetchone()[0] == 8


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
