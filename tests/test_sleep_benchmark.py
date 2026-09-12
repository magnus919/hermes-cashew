"""Contracts for the opt-in sleep measurement harness."""

from __future__ import annotations

import pytest

from plugins.memory.cashew.sleep_benchmark import run_benchmark


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
    assert report["embedding_calls"] == 2
    assert report["orphan_rows_with_embeddings"] == 2
    assert report["bounded_integrity"] is True
    assert report["participating_writer"]["observed"] is True
    assert report["participating_writer"]["admitted"] is False
    assert report["participating_writer"]["sqlite_writer_commit_ms"] >= 0
    assert {"find_candidates", "compute_metrics", "embed_orphans"}.issubset(
        report["phase_ms"]
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#193 upstream replacement must flush pending edges at the cap",
)
def test_benchmark_catches_edge_cap_commit_gap() -> None:
    report = run_benchmark(
        node_count=4,
        orphan_count=0,
        delay_s=0,
        max_edges=1,
        pair_similarity=0.8,
    )

    assert report["summary"]["cross_link_capped"] is True
    assert report["bounded_integrity"] is True
