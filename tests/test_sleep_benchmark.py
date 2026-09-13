"""Contracts for the opt-in sleep measurement harness."""

from __future__ import annotations

import inspect

import pytest

from plugins.memory.cashew.sleep_benchmark import run_benchmark


def test_benchmark_public_api_owns_storage(tmp_path) -> None:
    """The public benchmark cannot mutate a caller-selected database path."""
    assert "db_path" not in inspect.signature(run_benchmark).parameters

    caller_path = tmp_path / "caller.db"
    with pytest.raises(TypeError):
        run_benchmark(node_count=2, orphan_count=0, db_path=caller_path)
    assert not caller_path.exists()


def test_benchmark_is_deterministic_and_records_phase_and_contention_evidence() -> None:
    report = run_benchmark(
        node_count=4,
        orphan_count=2,
        delay_s=0.001,
        max_edges=10,
    )

    assert report["fixture"] == {
        "nodes": 4,
        "orphans": 2,
        "delay_ms": 1.0,
        "limit": 4,
        "max_edges": 10,
        "pair_similarity": None,
    }
    assert report["summary"]["nodes_selected"] == 4
    assert report["summary"]["orphans_embedded"] == 2
    # The pinned upstream pipeline batches the two eligible orphans into one
    # encode call; this is the transaction-boundary behavior being measured.
    assert report["embedding_calls"] == 1
    assert report["orphan_rows_with_embeddings"] == 2
    assert report["bounded_integrity"] is True
    assert report["participating_writer"]["observed"] is True
    assert report["participating_writer"]["admitted"] is False
    assert report["participating_writer"]["sqlite_writer_commit_ms"] >= 0
    assert {"find_candidates", "compute_metrics", "embed_orphans"}.issubset(
        report["phase_ms"]
    )


def test_benchmark_catches_edge_cap_commit_gap() -> None:
    report = run_benchmark(
        node_count=4,
        orphan_count=0,
        delay_s=0,
        max_edges=1,
        # gte-large's calibrated cross-link threshold is 0.90; 0.92 creates
        # real candidates without crossing its 0.94 dedup threshold.
        pair_similarity=0.92,
    )

    assert report["summary"]["cross_link_capped"] is True
    assert report["cross_link_rows"] == 2
    assert report["bounded_integrity"] is True
