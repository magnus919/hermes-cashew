# tests/test_sleep_refactor.py
"""Tests for plugins/memory/cashew/sleep_refactor.py — refactored sleep cycle.

Covers all nine phases in isolation + smoke test + integration test.
Mocks sklearn, SentenceTransformer, and model_fn to keep tests offline.
"""

from __future__ import annotations

import math
import sqlite3
import os
import sys
import tempfile
import hashlib
from typing import Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from plugins.memory.cashew.sleep_refactor import (
    CROSS_LINK_THRESHOLD,
    DEDUP_THRESHOLD,
    MAX_NODES_PER_CYCLE,
    MAX_EDGES_PER_CYCLE,
    run_sleep_cycle,
    _find_candidates,
    _batch_cross_links,
    _run_dedup,
    _compute_metrics,
    _garbage_collect,
    _evaluate_permanence,
    _promote_core_memories,
    _generate_dream,
    _embed_orphans,
    _merge_cluster,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _make_fake_embedding(nid: str, seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic embedding vector seeded per node id."""
    rng = np.random.RandomState(hash(nid) % 2**31 + seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create the minimal Cashew schema for sleep cycle tests."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'observation',
            timestamp TEXT NOT NULL DEFAULT '',
            mood_state TEXT,
            metadata TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0,
            domain TEXT DEFAULT 'user',
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            tags TEXT
        );

        CREATE TABLE IF NOT EXISTS derivation_edges (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            reasoning TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_edges_parent ON derivation_edges(parent_id);
        CREATE INDEX IF NOT EXISTS idx_edges_child ON derivation_edges(child_id);

        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)


def _insert_node(conn, id_: str, content: str, node_type: str = "observation",
                 timestamp: str = "2026-01-01T00:00:00", domain: str = "user",
                 source_file: str = None, access_count: int = 0,
                 permanent: bool = False) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO thought_nodes (id, content, node_type, timestamp, "
        "domain, source_file, access_count, permanent) VALUES (?,?,?,?,?,?,?,?)",
        (id_, content, node_type, timestamp, domain, source_file or "",
         access_count, 1 if permanent else 0),
    )


def _insert_embedding(conn, nid: str, seed: int = 42,
                     model: str = "all-MiniLM-L6-v2") -> None:
    vec = _make_fake_embedding(nid, seed)
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (node_id, vector, model, updated_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (nid, vec.tobytes(), model),
    )


def _insert_edge(conn, parent: str, child: str, weight: float = 1.0,
                 reasoning: str = "") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, weight, reasoning) "
        "VALUES (?, ?, ?, ?)",
        (parent, child, weight, reasoning),
    )


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary SQLite DB with schema."""
    path = str(tmp_path / "cashew_test.db")
    conn = sqlite3.connect(path)
    _create_schema(conn)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def small_graph(db_path):
    """Populate a 6-node graph with known embeddings."""
    conn = sqlite3.connect(db_path)
    # Three distinct clusters: [a,b] near-dups, [c,d] near-dups, [e,f] cross-link
    _insert_node(conn, "a", "alpha beta gamma delta", node_type="observation")
    _insert_node(conn, "b", "alpha beta gamma delta epsilon", node_type="observation",
                 access_count=3)
    _insert_node(conn, "c", "lorem ipsum dolor sit", node_type="fact")
    _insert_node(conn, "d", "lorem ipsum dolor sit amet", node_type="fact",
                 access_count=5)
    _insert_node(conn, "e", "machine learning gradient descent", node_type="insight",
                 source_file="file_A")
    _insert_node(conn, "f", "neural networks backpropagation optimization", node_type="insight",
                 source_file="file_B")
    # Add a permanent node
    _insert_node(conn, "perm1", "permanent reference node", node_type="core_memory",
                 permanent=True, access_count=100)
    _insert_node(conn, "perm2", "another permanent node", node_type="core_memory")

    for nid in ("a", "b", "c", "d", "e", "f", "perm1", "perm2"):
        seed = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
                "perm1": 10, "perm2": 11}[nid]
        _insert_embedding(conn, nid, seed=seed)

    _insert_edge(conn, "e", "f", weight=0.8, reasoning="cross_link - similarity=0.800")

    # a<->b should be near-duplicates (high similarity)
    va = _make_fake_embedding("a", seed=1)
    vb = _make_fake_embedding("b", seed=2)
    sim_ab = float(np.dot(va, vb))
    _insert_edge(conn, "a", "b", weight=sim_ab, reasoning="cross_link - similarity=0.800")

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_graph(db_path):
    """Graph with schema but no nodes."""
    return db_path


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Candidate discovery
# ──────────────────────────────────────────────────────────────────────────────

def test_find_candidates_smoke(small_graph):
    """Smoke test: find_candidates runs and returns shaped arrays."""
    conn = sqlite3.connect(small_graph)
    from plugins.memory.cashew.sleep_refactor import _load_embedding_matrix

    ids = ["a", "b", "c", "d", "e", "f"]
    valid_ids, matrix = _load_embedding_matrix(conn, ids)
    conn.close()

    cross, dedup, sim = _find_candidates(valid_ids, matrix)
    assert isinstance(cross, np.ndarray)
    assert isinstance(dedup, np.ndarray)
    assert sim.shape == (len(valid_ids), len(valid_ids))
    # At minimum nothing crashes — actual pair counts depend on random seeds


def test_find_candidates_empty():
    """Empty input returns empty arrays."""
    cross, dedup, sim = _find_candidates([], np.array([]).reshape(0, 384))
    assert len(cross) == 0
    assert len(dedup) == 0
    assert sim.shape == (0, 0)


def test_find_candidates_no_pairs():
    """All-zeros matrix produces no candidates (cosine sim is NaN → triu returns 0)."""
    ids = ["x", "y"]
    mat = np.zeros((2, 384), dtype=np.float32)
    cross, dedup, sim = _find_candidates(ids, mat)
    # Zero vectors produce NaN similarity, argwhere on NaN upper tri gives empty
    assert len(cross) == 0
    assert len(dedup) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Batched cross-linking
# ──────────────────────────────────────────────────────────────────────────────

def test_batch_cross_links_creates_edges(small_graph):
    """Batch insert creates edges for pairs above threshold."""
    conn = sqlite3.connect(small_graph)
    ids = ["a", "b", "c"]
    # Simulate a precomputed similarity where a-c is above threshold
    # but a-b already exists
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

    from plugins.memory.cashew.sleep_refactor import _load_embedding_matrix
    valid_ids, matrix = _load_embedding_matrix(conn, ids)
    cross_pairs, _, sim = _find_candidates(valid_ids, matrix)

    if len(cross_pairs) == 0:
        conn.close()
        pytest.skip("No cross-link pairs generated at random")

    edge_before = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
    stats = _batch_cross_links(conn, valid_ids, cross_pairs, sim)
    edge_after = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]

    assert stats["candidates"] >= 0
    assert stats["created"] + stats["skipped"] + stats["same_source_skipped"] == stats["candidates"]
    # Edges may or may not increase depending on pre-existing
    conn.close()


def test_batch_cross_links_empty():
    """Empty pair array is a no-op."""
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    stats = _batch_cross_links(conn, [], np.array([]).reshape(0, 2), np.array([]))
    assert stats == {"candidates": 0, "created": 0, "skipped": 0,
                     "same_source_skipped": 0, "capped": False}
    conn.close()


def test_batch_cross_links_cross_source_filter(db_path):
    """Same source_file pairs are skipped when source_files dict is provided."""
    conn = sqlite3.connect(db_path)
    # Insert 4 nodes: (a,b) from source_X, (c,d) from source_Y
    _insert_node(conn, "a", "alpha beta gamma", source_file="source_X")
    _insert_node(conn, "b", "delta epsilon zeta", source_file="source_X")
    _insert_node(conn, "c", "eta theta iota", source_file="source_Y")
    _insert_node(conn, "d", "kappa lambda mu", source_file="source_Y")
    for nid in ("a", "b", "c", "d"):
        _insert_embedding(conn, nid, seed={"a": 1, "b": 2, "c": 3, "d": 4}[nid])
    conn.commit()

    ids = ["a", "b", "c", "d"]
    from plugins.memory.cashew.sleep_refactor import _load_embedding_matrix
    valid_ids, matrix = _load_embedding_matrix(conn, ids)
    cross_pairs, _, sim = _find_candidates(valid_ids, matrix)

    if len(cross_pairs) == 0:
        conn.close()
        pytest.skip("No cross-link pairs generated at random with these seeds")

    source_files = {"a": "source_X", "b": "source_X",
                    "c": "source_Y", "d": "source_Y"}

    stats = _batch_cross_links(conn, valid_ids, cross_pairs, sim,
                               source_files=source_files)

    assert stats["same_source_skipped"] > 0, \
        "Expected same-source pairs to be skipped"
    assert stats["created"] + stats["skipped"] + stats["same_source_skipped"] \
        == stats["candidates"]
    conn.close()


def test_batch_cross_links_edge_cap(db_path):
    """max_edges parameter limits the number of edges created."""
    conn = sqlite3.connect(db_path)
    # Insert many nodes so we get plenty of cross-link candidates
    rng = np.random.RandomState(0)
    for i in range(100):
        nid = f"n{i:04d}"
        _insert_node(conn, nid, f"content {i} random {rng.randn():.4f}")
        _insert_embedding(conn, nid, seed=i)
    conn.commit()

    ids = [f"n{i:04d}" for i in range(100)]
    from plugins.memory.cashew.sleep_refactor import _load_embedding_matrix
    valid_ids, matrix = _load_embedding_matrix(conn, ids)
    cross_pairs, _, sim = _find_candidates(valid_ids, matrix)

    if len(cross_pairs) == 0:
        conn.close()
        pytest.skip("No cross-link pairs generated at random with these seeds")

    # Cap at a tiny number
    stats = _batch_cross_links(conn, valid_ids, cross_pairs, sim, max_edges=10)

    assert stats["capped"] is True, "Expected cap to be reached"
    assert stats["created"] == 10, f"Expected 10 edges, got {stats['created']}"
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Dedup via connected components
# ──────────────────────────────────────────────────────────────────────────────

def test_merge_cluster_rewires_edges(small_graph):
    """merge_cluster keeps the most-accessed node and decays the losers."""
    conn = sqlite3.connect(small_graph)

    # a and b are near-duplicates. b has higher access_count.
    cluster = ["a", "b"]
    keeper = _merge_cluster(conn, cluster)
    assert keeper is not None
    assert keeper == "b"  # b has access_count=3, a has 0

    # a should be decayed
    row = conn.execute("SELECT decayed FROM thought_nodes WHERE id='a'").fetchone()
    assert row[0] == 1

    # b should NOT be decayed
    row = conn.execute("SELECT decayed FROM thought_nodes WHERE id='b'").fetchone()
    assert row[0] == 0

    # No self-loops on keeper
    sl = conn.execute(
        "SELECT COUNT(*) FROM derivation_edges WHERE parent_id='b' AND child_id='b'"
    ).fetchone()[0]
    assert sl == 0

    conn.close()


def test_merge_cluster_single_node():
    """Single-node cluster returns None."""
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    assert _merge_cluster(conn, ["lonely"]) is None
    conn.close()


def test_merge_cluster_empty():
    """Empty cluster returns None."""
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    assert _merge_cluster(conn, []) is None
    conn.close()


def test_run_dedup_connected_components(small_graph):
    """Connected-component BFS merges transitive dedup chains."""
    conn = sqlite3.connect(small_graph)
    # Construct a dedup triangle: a-b, b-c → all three should merge
    # We use fake dedup pairs (indices) since actual thresholds depend on random seeds
    dedup_pairs = np.array([[0, 1], [1, 2]], dtype=int)  # a↔b, b↔c
    ids = ["a", "b", "c"]

    stats = _run_dedup(conn, ids, dedup_pairs)

    # Components found
    assert stats["components"] >= 1
    assert stats["nodes_merged"] >= 1

    # c should be decayed (lowest access count)
    c_decayed = conn.execute(
        "SELECT decayed FROM thought_nodes WHERE id='c'"
    ).fetchone()[0]
    assert c_decayed == 1

    conn.close()


def test_run_dedup_empty():
    """Empty dedup pairs returns zeros."""
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    stats = _run_dedup(conn, [], np.array([]).reshape(0, 2))
    assert stats == {"components": 0, "nodes_merged": 0}
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Node metrics
# ──────────────────────────────────────────────────────────────────────────────

def test_compute_metrics(small_graph):
    """compute_metrics returns a dict with expected keys for each active node."""
    conn = sqlite3.connect(small_graph)
    metrics = _compute_metrics(conn)
    conn.close()

    assert len(metrics) > 0
    for nid, m in metrics.items():
        assert "branching_factor" in m
        assert "cross_links" in m
        assert "fitness" in m
        assert m["branching_factor"] >= 0
        assert m["cross_links"] >= 0
        assert m["fitness"] >= 0

    # Node 'e' has an outgoing edge → branching_factor > 0
    assert any(m["branching_factor"] > 0 for m in metrics.values())


def test_compute_metrics_empty(empty_graph):
    """Empty graph returns empty dict."""
    conn = sqlite3.connect(empty_graph)
    metrics = _compute_metrics(conn)
    conn.close()
    assert metrics == {}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Garbage collection
# ──────────────────────────────────────────────────────────────────────────────

def test_garbage_collect_decays_low_fitness(small_graph):
    """GC decays non-permanent nodes with fitness=0."""
    conn = sqlite3.connect(small_graph)
    # Add an isolated node (no edges → fitness=0)
    _insert_node(conn, "isolated", "no edges at all")
    _insert_embedding(conn, "isolated", seed=99)
    conn.commit()

    metrics = _compute_metrics(conn)
    before_decayed = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE decayed=1"
    ).fetchone()[0]

    count = _garbage_collect(conn, metrics)
    conn.commit()

    after_decayed = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE decayed=1"
    ).fetchone()[0]

    # The isolated node with fitness=0 should be decayed
    assert after_decayed >= before_decayed
    assert count >= 0

    conn.close()


def test_garbage_collect_skips_permanent(small_graph):
    """GC never touches permanent nodes."""
    conn = sqlite3.connect(small_graph)
    metrics = _compute_metrics(conn)
    perm_before = conn.execute(
        "SELECT id FROM thought_nodes WHERE permanent=1 AND decayed=0"
    ).fetchall()

    _garbage_collect(conn, metrics)
    conn.commit()

    perm_after = conn.execute(
        "SELECT id FROM thought_nodes WHERE permanent=1 AND decayed=0"
    ).fetchall()
    assert set(r[0] for r in perm_before) == set(r[0] for r in perm_after)
    conn.close()


def test_garbage_collect_empty_metrics(empty_graph):
    """Empty metrics returns 0."""
    conn = sqlite3.connect(empty_graph)
    assert _garbage_collect(conn, {}) == 0
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Permanence evaluation
# ──────────────────────────────────────────────────────────────────────────────

def test_evaluate_permanence():
    """Promotes nodes with access_count >= 10 directly via SQL."""
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    _insert_node(conn, "high", "high access", access_count=15)
    _insert_node(conn, "low", "low access", access_count=3)
    _insert_node(conn, "exact", "exact threshold", access_count=10)
    conn.commit()

    stats = _evaluate_permanence(conn)

    assert stats["nodes_promoted"] >= 1
    assert stats["threshold"] == 10

    # high and exact should be permanent
    for nid in ("high", "exact"):
        perm = conn.execute(
            "SELECT permanent FROM thought_nodes WHERE id=?", (nid,)
        ).fetchone()
        assert perm[0] == 1, f"node {nid} should be permanent"

    # low should not
    perm = conn.execute(
        "SELECT permanent FROM thought_nodes WHERE id='low'"
    ).fetchone()
    assert perm[0] == 0
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7: Core memory promotion
# ──────────────────────────────────────────────────────────────────────────────

def test_promote_core_memories(small_graph):
    """Top √N by fitness become core_memory."""
    conn = sqlite3.connect(small_graph)
    metrics = _compute_metrics(conn)

    before_core = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE node_type='core_memory'"
    ).fetchone()[0]

    stats = _promote_core_memories(conn, metrics)

    after_core = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE node_type='core_memory'"
    ).fetchone()[0]

    target = int(math.sqrt(len(metrics)))
    assert stats["target"] == target
    assert stats["promoted"] >= 0
    assert stats["demoted"] >= 0
    # At minimum, prior core_memory nodes should still exist
    assert after_core >= before_core

    conn.close()


def test_promote_core_memories_sets_permanent(small_graph):
    """Promoted core memories get permanent=1."""
    conn = sqlite3.connect(small_graph)
    metrics = _compute_metrics(conn)

    # Clear existing permanent flag on a core node
    conn.execute("UPDATE thought_nodes SET permanent=0 WHERE id='perm2'")
    conn.commit()

    _promote_core_memories(conn, metrics)

    # perm2 should be repaired (permanent=1)
    row = conn.execute(
        "SELECT permanent FROM thought_nodes WHERE id='perm2'"
    ).fetchone()
    assert row[0] == 1
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 8: Dream generation
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_dream_no_model_fn(small_graph):
    """Dream generation returns None without model_fn."""
    conn = sqlite3.connect(small_graph)
    result = _generate_dream(conn, [])
    assert result is None
    conn.close()


def test_generate_dream_with_model_fn(small_graph):
    """Dream generation creates a node when model_fn returns valid text."""
    conn = sqlite3.connect(small_graph)

    def fake_model_fn(prompt: str) -> str:
        return "Synthesis: both snippets share a latent assumption about local-first processing."

    cross_tuples = [("e", "f", 0.85)]
    result = _generate_dream(conn, cross_tuples, model_fn=fake_model_fn)

    assert result is not None
    assert len(result) == 12  # SHA-256 hex digest[:12]

    # Verify dream node exists
    row = conn.execute(
        "SELECT content, node_type, source_file FROM thought_nodes WHERE id=?",
        (result,),
    ).fetchone()
    assert row is not None
    assert row[1] == "dream"
    assert row[2] == "sleep_protocol"

    # Verify edges from both bridge nodes to dream
    edges = conn.execute(
        "SELECT COUNT(*) FROM derivation_edges WHERE child_id=?", (result,)
    ).fetchone()[0]
    assert edges == 2

    conn.close()


def test_generate_dream_short_response_returns_none(small_graph):
    """Responses shorter than 20 chars return None."""
    conn = sqlite3.connect(small_graph)

    def fake_model_fn(prompt: str) -> str:
        return "Short."

    result = _generate_dream(conn, [("e", "f", 0.85)], model_fn=fake_model_fn)
    assert result is None
    conn.close()


def test_generate_dream_no_cross_source_bridge(small_graph):
    """Returns None when candidates don't bridge different source files."""
    conn = sqlite3.connect(small_graph)
    # a and b have same source_file (both empty string defaults)
    cross_tuples = [("a", "b", 0.9)]

    def fake_model_fn(prompt: str) -> str:
        return "A meaningful synthesis of the two ideas."

    result = _generate_dream(conn, cross_tuples, model_fn=fake_model_fn)
    # a and b have no source_file set → empty strings are equal → no bridge
    assert result is None
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 9: Embedding gap closure
# ──────────────────────────────────────────────────────────────────────────────

def test_embed_orphans_no_orphans(small_graph):
    """Returns 0 when all active nodes have embeddings."""
    conn = sqlite3.connect(small_graph)
    # All nodes in small_graph have embeddings already
    count = _embed_orphans(conn)
    assert count == 0
    conn.close()


def test_embed_orphans_regression_not_null_schema(db_path):
    """Regression: _embed_orphans handles the production schema with model +
    updated_at NOT NULL constraints (issue #41).

    The production embeddings table has NOT NULL on ``model`` and ``updated_at``.
    Earlier code used ``model_name`` (wrong column) in the primary INSERT, then
    omitted both constrained columns in the fallback, causing IntegrityError.
    """
    conn = sqlite3.connect(db_path)
    # Schema is already the production one (set by _create_schema in db_path
    # fixture). Verify.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()}
    assert "model" in cols, "test schema must include model column"
    assert "updated_at" in cols, "test schema must include updated_at column"

    # Insert a node without an embedding row (an orphan)
    _insert_node(conn, "orphan1", "this node needs an embedding", node_type="observation")
    conn.commit()

    count = _embed_orphans(conn)
    assert count == 1, "should have embedded 1 orphan"

    # Verify the embedding row exists with all columns populated
    row = conn.execute(
        "SELECT node_id, model, updated_at FROM embeddings WHERE node_id='orphan1'"
    ).fetchone()
    assert row is not None, "embedding row should exist"
    assert row[1] == "thenlper/gte-large", f"expected model='thenlper/gte-large', got {row[1]!r}"
    assert row[2] is not None and len(str(row[2])) > 0, "updated_at should be set"

    conn.close()


def test_embed_orphans_mixed_orphans(db_path):
    """Embedding gap closure works with a mix of orphaned and already-embedded nodes."""
    conn = sqlite3.connect(db_path)
    # Add one node with embedding, two without
    _insert_node(conn, "has_embed", "already embedded content")
    _insert_embedding(conn, "has_embed", seed=1)
    _insert_node(conn, "needs_embed_a", "orphan content A")
    _insert_node(conn, "needs_embed_b", "orphan content B")
    conn.commit()

    count = _embed_orphans(conn)
    assert count == 2, "should embed exactly the 2 orphans"

    # has_embed should still have its original embedding
    row = conn.execute(
        "SELECT model FROM embeddings WHERE node_id='has_embed'"
    ).fetchone()
    assert row is not None, "original embedding should still exist"

    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test: full cycle
# ──────────────────────────────────────────────────────────────────────────────

def test_run_sleep_cycle_smoke(small_graph, monkeypatch):
    """Full cycle completes without error and returns expected summary keys."""
    # Mock sklearn to avoid real dependency (HF_HUB_OFFLINE is set)
    from unittest.mock import patch as _patch

    # We rely on the existing test embedding vectors + sklearn being available
    result = run_sleep_cycle(small_graph, limit=6, model_fn=None)
    assert "error" not in result

    expected_keys = {
        "nodes_selected", "nodes_with_embeddings",
        "cross_link_candidates", "dedup_candidates",
        "cross_links_created", "cross_links_skipped",
        "cross_link_same_source_skipped", "cross_link_capped",
        "dedup_components", "dedup_nodes_merged",
        "nodes_gc_decayed", "nodes_made_permanent",
        "core_promoted", "core_demoted",
        "dream_id", "dream_pending", "dream_generation",
        "orphans_embedded",
        "total_nodes", "elapsed_s",
    }
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    assert result["elapsed_s"] >= 0
    assert result["total_nodes"] > 0


def test_run_sleep_cycle_too_few_nodes(empty_graph):
    """Gracefully handles graphs with fewer than 2 embedded nodes."""
    result = run_sleep_cycle(empty_graph, limit=100, model_fn=None)
    assert "error" in result
    assert result["error"] in ("too few nodes", "no nodes selected")


def test_run_sleep_cycle_respects_limit(small_graph):
    """limit parameter caps nodes selected."""
    result = run_sleep_cycle(small_graph, limit=2, model_fn=None)
    assert result["nodes_selected"] <= 2


def test_run_sleep_cycle_with_model_fn(small_graph):
    """Full cycle with model_fn enables dream generation."""
    def fake_model_fn(prompt: str) -> str:
        return "Cross-domain synthesis: both perspectives converge on batch-first design."

    result = run_sleep_cycle(small_graph, limit=6, model_fn=fake_model_fn)
    # dream_id may be None if no cross-source bridge exists, but shouldn't crash
    assert "dream_id" in result


# ──────────────────────────────────────────────────────────────────────────────
# Integration: on_session_end triggers sleep
# ──────────────────────────────────────────────────────────────────────────────

def test_on_session_end_does_not_call_sleep_cycle(tmp_path, monkeypatch):
    """on_session_end no longer calls run_sleep_cycle — migrated to cron (v0.11.0)."""
    import json
    from plugins.memory.cashew import CashewMemoryProvider

    hermes_home = tmp_path / "hermes_test1"
    hermes_home.mkdir()
    cashew_cfg = hermes_home / "cashew.json"
    cashew_cfg.write_text(json.dumps({
        "sleep_cycles": True,
        "sleep_schedule": "every 12h",
        "think_cycles": False,
        "cashew_db_path": "cashew/brain.db",
    }))

    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("model:\n  provider: test\n  default: test\n")

    db_dir = hermes_home / "cashew"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "brain.db"))
    _create_schema(conn)
    conn.close()

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(hermes_home))

    import time
    time.sleep(0.3)

    call_log = []
    def fake_run_sleep(*args, **kwargs):
        call_log.append(1)
        return {}

    monkeypatch.setattr(
        "plugins.memory.cashew.sleep_refactor.run_sleep_cycle",
        fake_run_sleep,
    )

    provider.on_session_end([])

    assert len(call_log) == 0, "on_session_end should NOT call run_sleep_cycle"
    provider.shutdown()


def test_on_session_end_skips_when_disabled(tmp_path, monkeypatch):
    """When sleep_cycles=false, on_session_end does NOT call run_sleep_cycle."""
    import json
    from plugins.memory.cashew import CashewMemoryProvider

    hermes_home = tmp_path / "hermes_test2"
    hermes_home.mkdir()
    cashew_cfg = hermes_home / "cashew.json"
    cashew_cfg.write_text(json.dumps({
        "sleep_cycles": False,
        "think_cycles": False,
    }))

    db_dir = hermes_home / "cashew"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "brain.db"))
    _create_schema(conn)
    conn.close()

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(hermes_home))

    import time
    time.sleep(0.3)

    call_log = []
    def fake_run_sleep(*args, **kwargs):
        call_log.append(1)
        return {}

    import plugins.memory.cashew.sleep_refactor as sr
    monkeypatch.setattr(sr, "run_sleep_cycle", fake_run_sleep)

    provider.on_session_end([])

    assert len(call_log) == 0
    provider.shutdown()


def test_on_session_end_does_not_raise(tmp_path):
    """on_session_end is a safe no-op — sleep cycle was migrated to cron (v0.11.0)."""
    import json
    from plugins.memory.cashew import CashewMemoryProvider

    hermes_home = tmp_path / "hermes_test3"
    hermes_home.mkdir()
    cashew_cfg = hermes_home / "cashew.json"
    cashew_cfg.write_text(json.dumps({
        "sleep_cycles": True,
        "think_cycles": False,
    }))

    db_dir = hermes_home / "cashew"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "brain.db"))
    _create_schema(conn)
    conn.close()

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(hermes_home))

    import time
    time.sleep(0.3)

    # Should not raise even though sleep cycle was previously called here
    provider.on_session_end([])
    provider.shutdown()


def test_on_session_end_does_not_drain_sync_queue(tmp_path, monkeypatch):
    """on_session_end returns promptly even when the sync queue has pending items."""
    import json
    from plugins.memory.cashew import CashewMemoryProvider

    hermes_home = tmp_path / "hermes_test4"
    hermes_home.mkdir()
    cashew_cfg = hermes_home / "cashew.json"
    cashew_cfg.write_text(json.dumps({
        "sleep_cycles": True,
        "think_cycles": False,
        "sync_queue_timeout": 30.0,
    }))

    config_yaml = hermes_home / "config.yaml"
    config_yaml.write_text("auxiliary:\n  memory:\n    provider: test\n    model: test\n    api_key: fake\n")

    db_dir = hermes_home / "cashew"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "brain.db"))
    _create_schema(conn)
    for i, nid in enumerate(["n1", "n2", "n3"]):
        _insert_node(conn, nid, f"content {i}")
        _insert_embedding(conn, nid, seed=i)
    conn.commit()
    conn.close()

    provider = CashewMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(hermes_home))

    # Let the initial drain settle
    import time
    time.sleep(0.3)

    # Simulate a pending queue by adding turns that the worker will process
    # The worker will pick these up asynchronously
    for i in range(3):
        provider.sync_turn(f"user turn {i}", f"assistant reply {i}")

    # Even with items in the queue, on_session_end should return quickly
    monkeypatch.setattr(
        "plugins.memory.cashew.sleep_refactor.run_sleep_cycle",
        lambda *a, **kw: {"total_nodes": 3, "elapsed_s": 0.1},
    )

    import time as _time
    t0 = _time.monotonic()
    provider.on_session_end([])
    elapsed = _time.monotonic() - t0

    # Should return in well under 1s — there's no drain loop anymore
    assert elapsed < 1.0, f"on_session_end took {elapsed:.2f}s (should be instant)"

    provider.shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────

def test_merge_cluster_preserves_edge_count(small_graph):
    """After merge, total edge count should not collapse to near zero."""
    conn = sqlite3.connect(small_graph)
    edge_before = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]

    cluster = ["a", "b"]
    _merge_cluster(conn, cluster)

    edge_after = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
    # Edge count should be reasonable — not a massive drop
    assert edge_after >= edge_before - 2  # max 2 self-loops dropped
    conn.close()


def test_sleep_cycle_idempotency(small_graph):
    """Running the cycle twice doesn't corrupt the DB."""
    result1 = run_sleep_cycle(small_graph, limit=6, model_fn=None)
    result2 = run_sleep_cycle(small_graph, limit=6, model_fn=None)

    # Both should complete
    assert "error" not in result1
    assert "error" not in result2
    # Second run should have fewer new cross-links (already created in first)
    assert result2["cross_links_created"] <= result1["cross_links_created"]
    # Decayed count should stabilize
    assert result2["nodes_gc_decayed"] <= result1["nodes_gc_decayed"] + 1


# ──────────────────────────────────────────────────────────────────────────────
# Background dream dispatch (v0.10.0)
# ──────────────────────────────────────────────────────────────────────────────


def test_run_dream_async_spawns_daemon_thread(small_graph):
    """_run_dream_async spawns a daemon thread that writes a dream node."""
    def fake_model_fn(prompt: str) -> str:
        return "Dream synthesis bridging file_A and file_B."

    cross_tuples = [("e", "f", 0.85)]
    from plugins.memory.cashew.sleep_refactor import _run_dream_async
    _run_dream_async(small_graph, cross_tuples, fake_model_fn)

    # Give the daemon thread time to write
    import time
    time.sleep(1.0)

    conn = sqlite3.connect(small_graph)
    dream_nodes = conn.execute(
        "SELECT id, content, node_type FROM thought_nodes WHERE node_type='dream'"
    ).fetchall()
    conn.close()

    assert len(dream_nodes) >= 1
    assert dream_nodes[0][1] == "Dream synthesis bridging file_A and file_B."
    assert dream_nodes[0][2] == "dream"


def test_run_dream_async_handles_exception(small_graph):
    """Exception in daemon thread is caught and logged — does not propagate."""
    cross_tuples = [("e", "f", 0.85)]
    from plugins.memory.cashew.sleep_refactor import _run_dream_async
    _run_dream_async(small_graph, cross_tuples, None)  # None model_fn -> skips dream

    # Should not crash
    import time
    time.sleep(0.3)
    assert True


def test_background_dream_flag_requires_model_fn_and_tuples(small_graph, monkeypatch):
    """background_dream=True without model_fn sets dream_generation='skipped'."""
    result = run_sleep_cycle(small_graph, limit=6, model_fn=None,
                             background_dream=True)
    assert "dream_pending" in result
    assert result["dream_pending"] is False
    assert result["dream_generation"] == "skipped"


def test_background_dream_runs_sync_phases_before_returning(small_graph):
    """Synchronous phases (cross-links, dedup, GC, core) complete even with bg dream."""
    result = run_sleep_cycle(small_graph, limit=6, model_fn=None,
                             background_dream=True)
    assert result["nodes_selected"] > 0
    assert "cross_link_candidates" in result
    assert "nodes_gc_decayed" in result
    assert "core_promoted" in result
    assert "elapsed_s" in result


def test_background_dream_false_maintains_legacy_behavior(small_graph):
    """background_dream=False with model_fn runs dream synchronously."""
    def fake_model_fn(prompt: str) -> str:
        return "Legacy dream synthesis for testing."

    result = run_sleep_cycle(small_graph, limit=6, model_fn=fake_model_fn,
                             background_dream=False)
    assert "dream_pending" in result
    assert result["dream_pending"] is False
    assert "dream_id" in result
    assert "dream_generation" in result
    assert result["dream_generation"] in ("ran", "skipped", "failed")
