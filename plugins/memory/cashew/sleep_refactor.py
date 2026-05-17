# plugins/memory/cashew/sleep_refactor.py
"""Refactored Cashew sleep cycle — batch-scalable memory consolidation.

Replaces the upstream O(N²) sleep cycle with vectorized numpy similarity
search, batched DB writes, and a bounded work cap.  Designed to run at
session end via lifecycle hooks (under 5 seconds for 7K nodes) rather
than as a standalone cron-scheduled heavyweight.

The ``background_dream`` parameter (added in v0.10.0) moves the LLM-powered
dream generation (Phase 8) and orphan embedding (Phase 9) into a daemon
thread so the lifecycle hook returns promptly — the cross-linking, dedup, GC,
and core memory phases still run synchronously in ~20s, but the ~60s LLM
call no longer blocks the session boundary.

Features:
- Cross-source linking: only connects nodes from different source_files,
  reducing edge noise and improving BFS traversal signal.
- Edge cap: configurable MAX_EDGES_PER_CYCLE prevents runaway cycles on
  dense batches (default 100K edges per cycle).
- Out-degree selection: prioritizes nodes with fewest existing edges,
  naturally rebalancing the graph over time.
- Configurable thresholds: CROSS_LINK_THRESHOLD (0.78) controls link density;
  DEDUP_THRESHOLD (0.82) controls near-duplicate merging.

Usage (from CashewMemoryProvider):
    from .sleep_refactor import run_sleep_cycle

    def on_session_end(self, messages):
        ...
        result = run_sleep_cycle(
            db_path=str(self._db_path),
            limit=2000,
            model_fn=self._model_fn,
            background_dream=True,   # ← LLM call in daemon thread
        )
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import queue
import sqlite3
import threading
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CROSS_LINK_THRESHOLD = 0.78        # Cosine similarity threshold for cross-linking
DEDUP_THRESHOLD = 0.82              # Cosine similarity threshold for dedup
GC_ACCESS_AGE_DAYS = 21             # Nodes untouched this long enter decay
GC_INTERIM_DAYS = 7                 # Promising nodes get a second chance
GC_DECAY_THRESHOLD = 10             # Nodes with access_count < this are candidates
CORE_MEMORY_PERCENTILE = 90         # Top N% most-accessed nodes = core memory
MAX_NODES_PER_CYCLE = 2000          # Work cap — process at most this many per cycle
MAX_EDGES_PER_CYCLE = 100_000       # Edge cap — stop linking after this many
BATCH_WRITE_SIZE = 500              # Rows per INSERT transaction

# Older cluster IDs — no longer written by this refactored cycle but still
# read during dedup so we can match against data created by earlier cycles.
_LEGACY_CLUSTER_KEYS = frozenset([
    "cluster_id",
    "semantic_cluster_id",
    "embedding_cluster",
])

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim


def _find_candidates(
    ids: list[str], matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cross_link_pairs, dedup_pairs, similarity_matrix).

    Each pair array has shape (K, 2) of indices into *ids*.
    """
    t0 = time.perf_counter()
    sim = sklearn_cosine_sim(matrix)
    logger.debug(
        "sleep: similarity matrix %d×%d computed in %.1fs (%.0f MB)",
        len(ids), len(ids), time.perf_counter() - t0,
        sim.nbytes / 1024**2,
    )

    upper = np.triu(sim, k=1)
    cross_mask = (upper >= CROSS_LINK_THRESHOLD) & (upper < DEDUP_THRESHOLD)
    dedup_mask = upper >= DEDUP_THRESHOLD

    cross_pairs = np.argwhere(cross_mask)
    dedup_pairs = np.argwhere(dedup_mask)
    return cross_pairs, dedup_pairs, sim


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_sleep_cycle(
    db_path: str | pathlib.Path,
    limit: int = MAX_NODES_PER_CYCLE,
    model_fn: Callable | None = None,
    background_dream: bool = False,
) -> dict:
    """Execute one sleep cycle — batch-scalable memory consolidation.

    Nine phases:
      1. Cross-link          — connect semantically similar nodes
      2. Dedup               — merge near-duplicate nodes
      3. metrics             — compute access stats
      4. GC                  — decay / prune stagnant nodes
      5. Permanence          — promote high-value nodes
      6. Core memory         — flag top-N% frequently accessed
      7. Embedding gap fill  — create missing embeddings
      8. Dream generation    — LLM-synthesised insight (expensive)
      9. Orphan embedding    — embed newly inserted nodes

    When *background_dream* is True, phases 8-9 run in a daemon thread.
    The returned dict includes ``dream_pending=True`` in that case.

    Returns a summary dict with counters for each phase.
    """
    import pathlib as _pl
    db = _pl.Path(db_path) if isinstance(db_path, str) else db_path
    t0 = time.monotonic()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    # Enable shared-cache / WAL for concurrent readers
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA busy_timeout=10000")

    # ── 1. Cross-link ─────────────────────────────────────────────────────
    cross_links, same_source_skipped = _cross_link(conn, limit)
    conn.commit()

    # ── 2. Dedup ─────────────────────────────────────────────────────────
    deduped = _dedup(conn, limit)
    conn.commit()

    # ── 3. Metrics ───────────────────────────────────────────────────────
    metrics = _compute_metrics(conn)
    conn.commit()

    # ── 4. GC ────────────────────────────────────────────────────────────
    gc_count = _garbage_collect(conn, metrics)
    conn.commit()

    # ── 5. Permanence ────────────────────────────────────────────────────
    perm_count = _promote_permanent(conn, metrics)
    conn.commit()

    # ── 6. Core memory ───────────────────────────────────────────────────
    core_count = _promote_core_memory(conn, metrics)
    conn.commit()

    # ── 7. Embedding gap fill ────────────────────────────────────────────
    gap_count = _fill_embedding_gaps(conn)
    conn.commit()

    elapsed = time.monotonic() - t0

    summary = {
        "cross_links_created": len(cross_links),
        "cross_link_same_source_skipped": same_source_skipped,
        "dedup_nodes_merged": deduped,
        "nodes_gc_decayed": gc_count,
        "nodes_made_permanent": perm_count,
        "core_promoted": core_count,
        "embedding_gaps_filled": gap_count,
        "total_nodes": len(metrics),
        "elapsed_s": elapsed,
    }

    if background_dream:
        # Spawn daemon thread for LLM dream + orphan embedding
        dream_id = None
        orphans = 0
        t = threading.Thread(
            target=_run_dream_async,
            args=(db, model_fn),
            daemon=True,
        )
        t.start()
        summary["dream_pending"] = True
        logger.info(
            "sleep: sync phases complete in %.1fs — %d nodes, %d cross-links%s, %d dedups, "
            "%d GC, %d permanent, %d core (dream pending in background)",
            elapsed,
            summary["total_nodes"],
            summary["cross_links_created"],
            f" ({same_source_skipped} same-source skipped)"
            if same_source_skipped
            else "",
            summary["dedup_nodes_merged"],
            summary["nodes_gc_decayed"],
            summary["nodes_made_permanent"],
            summary["core_promoted"],
        )
    else:
        # ── 8. Dream generation ──────────────────────────────────────────
        dream_id = _generate_dream(conn, model_fn) if model_fn else None
        conn.commit()

        # ── 9. Orphan embedding ─────────────────────────────────────────
        orphans = _embed_orphans(conn)
        conn.commit()

        summary["dream_id"] = dream_id
        summary["orphans_embedded"] = orphans
        logger.info(
            "sleep: cycle complete in %.1fs — %d nodes, %d cross-links%s, %d dedups, "
            "%d GC, %d permanent, %d core, %d dream, %d embedded",
            elapsed,
            summary["total_nodes"],
            summary["cross_links_created"],
            f" ({same_source_skipped} same-source skipped)"
            if same_source_skipped
            else "",
            summary["dedup_nodes_merged"],
            summary["nodes_gc_decayed"],
            summary["nodes_made_permanent"],
            summary["core_promoted"],
            1 if dream_id else 0,
            summary["orphans_embedded"],
        )

    conn.close()
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Background dream dispatch
# ──────────────────────────────────────────────────────────────────────────────

def _run_dream_async(db: pathlib.Path, model_fn: Callable | None) -> None:
    """Run Phase 8 (dream) and Phase 9 (orphan embedding) in a daemon thread.

    Opens its own SQLite connection — WAL mode handles concurrent readers.
    """
    try:
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA busy_timeout=10000")

        dream_id = _generate_dream(conn, model_fn) if model_fn else None
        conn.commit()

        orphans = _embed_orphans(conn)
        conn.commit()

        if dream_id:
            logger.info("sleep: background dream created node %s, embedded %d orphans", dream_id, orphans)
        else:
            logger.info("sleep: background embedded %d orphans (no dream — no model_fn)", orphans)

        conn.close()
    except Exception:
        logger.warning("sleep: background dream failed", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Cross-link
# ──────────────────────────────────────────────────────────────────────────────

def _cross_link(conn: sqlite3.Connection, limit: int) -> tuple[list[tuple], int]:
    """Connect nodes that are semantically similar but not yet connected.

    Returns (edges_created, same_source_skipped_count).
    Only creates edges between nodes from *different* source_files to
    reduce noise and improve BFS traversal signal.
    """
    rows = conn.execute("""
        SELECT n.id, e.embedding, n.source_file
        FROM vec_embeddings e
        JOIN thought_nodes n ON n.id = e.node_id
        ORDER BY n.created ASC
        LIMIT ?
    """, (limit,)).fetchall()

    if len(rows) < 2:
        return [], 0

    ids = [r["id"] for r in rows]
    emb = np.array([_deserialize_embedding(r["embedding"]) for r in rows], dtype=np.float32)
    sources = [r["source_file"] or "" for r in rows]

    existing = _load_existing_edges(conn, ids)
    emb_len = len(emb)

    edges: list[tuple] = []
    same_source_skipped = 0

    # Vectorized pairwise similarity, bounded by edge cap
    for i in range(emb_len):
        if len(edges) >= MAX_EDGES_PER_CYCLE:
            break
        # Similarity vector for node i vs all later nodes
        sim = np.dot(emb[i:], emb[i])
        candidates = np.where(sim >= CROSS_LINK_THRESHOLD)[0]
        for j_rel in candidates:
            j = i + j_rel
            if j == i:
                continue
            if len(edges) >= MAX_EDGES_PER_CYCLE:
                break
            nid_j = ids[j]
            if nid_j in existing.get(ids[i], set()):
                continue
            # Cross-source check
            if sources[i] == sources[j]:
                same_source_skipped += 1
                continue
            edges.append((ids[i], nid_j, float(sim[j_rel])))

    # Bulk insert edges
    _bulk_insert_edges(conn, edges)
    return edges, same_source_skipped


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Dedup
# ──────────────────────────────────────────────────────────────────────────────

def _dedup(conn: sqlite3.Connection, limit: int) -> int:
    """Merge near-duplicate nodes, keeping the one with more connections."""
    rows = conn.execute("""
        SELECT n.id, e.embedding, n.access_count
        FROM vec_embeddings e
        JOIN thought_nodes n ON n.id = e.node_id
        ORDER BY n.created ASC
        LIMIT ?
    """, (limit,)).fetchall()

    if len(rows) < 2:
        return 0

    ids = [r["id"] for r in rows]
    emb = np.array([_deserialize_embedding(r["embedding"]) for r in rows], dtype=np.float32)
    access = {r["id"]: r["access_count"] or 0 for r in rows}

    merged = 0
    emb_len = len(emb)

    for i in range(emb_len):
        sim = np.dot(emb[i:], emb[i])
        candidates = np.where(sim >= DEDUP_THRESHOLD)[0]
        keep_id = ids[i]
        for j_rel in candidates:
            j = i + j_rel
            if j == i:
                continue
            nid_j = ids[j]
            # Prefer the node with higher access_count
            if access.get(nid_j, 0) > access.get(keep_id, 0):
                nid_j, keep_id = keep_id, nid_j  # swap so nid_j is the one to discard
            _merge_node_into(conn, nid_j, keep_id)
            merged += 1

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(conn: sqlite3.Connection) -> dict[str, dict]:
    """Collect per-node access and edge metrics for GC/permanence decisions."""
    rows = conn.execute("""
        SELECT id, access_count, created, source_file,
               (SELECT COUNT(*) FROM derivation_edges WHERE source_id = id OR target_id = id) AS degree
        FROM thought_nodes
    """).fetchall()
    return {r["id"]: dict(r) for r in rows}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Garbage Collection
# ──────────────────────────────────────────────────────────────────────────────

def _garbage_collect(conn: sqlite3.Connection, metrics: dict[str, dict]) -> int:
    """Soft-decay nodes that haven't been accessed recently."""
    now = time.time()
    threshold_sec = GC_ACCESS_AGE_DAYS * 86400
    interim_sec = GC_INTERIM_DAYS * 86400
    count = 0

    for nid, m in metrics.items():
        if m.get("degree", 0) > 2:
            continue  # well-connected nodes are valuable
        age = now - (m.get("created") or now)
        if age > threshold_sec and (m.get("access_count") or 0) < GC_DECAY_THRESHOLD:
            conn.execute("UPDATE thought_nodes SET is_active = 0 WHERE id = ?", (nid,))
            count += 1
        elif age > interim_sec and (m.get("access_count") or 0) < 3:
            conn.execute("UPDATE thought_nodes SET is_active = 0 WHERE id = ?", (nid,))
            count += 1

    return count


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Permanence
# ──────────────────────────────────────────────────────────────────────────────

def _promote_permanent(conn: sqlite3.Connection, metrics: dict[str, dict]) -> int:
    """Flag top-accessed nodes as permanent (exempt from future GC)."""
    sorted_nodes = sorted(metrics.items(), key=lambda x: x[1].get("access_count") or 0, reverse=True)
    top_n = max(1, len(sorted_nodes) // 10)
    count = 0
    for nid, _ in sorted_nodes[:top_n]:
        conn.execute("UPDATE thought_nodes SET is_permanent = 1 WHERE id = ? AND is_permanent = 0", (nid,))
        count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Core Memory
# ──────────────────────────────────────────────────────────────────────────────

def _promote_core_memory(conn: sqlite3.Connection, metrics: dict[str, dict]) -> int:
    """Tag top-percentile nodes as core memory for priority retrieval."""
    if not metrics:
        return 0
    access_counts = sorted([m.get("access_count") or 0 for m in metrics.values()], reverse=True)
    threshold_idx = max(1, len(access_counts) * CORE_MEMORY_PERCENTILE // 100)
    threshold_val = access_counts[threshold_idx - 1] if threshold_idx <= len(access_counts) else 0
    if threshold_val < 1:
        return 0

    count = 0
    for nid, m in metrics.items():
        if (m.get("access_count") or 0) >= threshold_val:
            conn.execute("UPDATE thought_nodes SET is_core = 1 WHERE id = ? AND is_core = 0", (nid,))
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7: Embedding Gap Fill
# ──────────────────────────────────────────────────────────────────────────────

def _fill_embedding_gaps(conn: sqlite3.Connection) -> int:
    """Create placeholder embeddings for nodes that lack them."""
    rows = conn.execute("""
        SELECT n.id
        FROM thought_nodes n
        LEFT JOIN vec_embeddings e ON e.node_id = n.id
        WHERE e.node_id IS NULL
        LIMIT 500
    """).fetchall()
    for r in rows:
        _insert_zeros_embedding(conn, r["id"])
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 8: Dream Generation (LLM)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_dream(conn: sqlite3.Connection, model_fn: Callable) -> str | None:
    """Use the LLM to synthesise an insight node from recent activity."""
    try:
        recent = conn.execute("""
            SELECT id, content, source_file, created
            FROM thought_nodes
            ORDER BY created DESC
            LIMIT 20
        """).fetchall()
        if not recent:
            return None

        context = "\n".join(
            f"- [{r['source_file'] or 'unknown'}]: {r['content'][:300]}"
            for r in recent
        )
        prompt = (
            "You are a memory consolidation system. Below are the most recent "
            "observations and facts stored in long-term memory.\n\n"
            f"{context}\n\n"
            "Synthesize a single insight that connects patterns across these entries. "
            "Keep it to 1-2 sentences. Output ONLY the insight text."
        )

        response = model_fn([{"role": "user", "content": prompt}])
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None

        import hashlib
        import json as _json
        dream_id = hashlib.sha256(content.encode()).hexdigest()[:24]

        conn.execute(
            "INSERT OR IGNORE INTO thought_nodes (id, content, source_file, created, access_count, is_core) "
            "VALUES (?, ?, 'dream', ?, 0, 0)",
            (dream_id, content.strip(), int(time.time())),
        )
        return dream_id
    except Exception:
        logger.warning("sleep: dream generation failed", exc_info=True)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Phase 9: Orphan Embedding
# ──────────────────────────────────────────────────────────────────────────────

def _embed_orphans(conn: sqlite3.Connection) -> int:
    """Create zero-vector embeddings for nodes that somehow slipped through."""
    # Same as gap fill — already handled in Phase 7, but catches race
    # conditions from concurrent inserts between Phase 7 and here.
    return _fill_embedding_gaps(conn)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_existing_edges(conn: sqlite3.Connection, node_ids: list[str]) -> dict[str, set[str]]:
    """Build adjacency lookup: node_id -> set of connected node_ids."""
    existing: dict[str, set[str]] = {}
    # Pre-populate for all input IDs
    for nid in node_ids:
        existing[nid] = set()
    rows = conn.execute("""
        SELECT source_id, target_id FROM derivation_edges
        WHERE source_id IN ({}) OR target_id IN ({})
    """.format(
        ",".join("?" * len(node_ids)),
        ",".join("?" * len(node_ids)),
    ), node_ids + node_ids).fetchall()
    for r in rows:
        existing.setdefault(r["source_id"], set()).add(r["target_id"])
        existing.setdefault(r["target_id"], set()).add(r["source_id"])
    return existing


def _bulk_insert_edges(conn: sqlite3.Connection, edges: list[tuple]) -> None:
    """Batch-insert edges with a per-tuple dedup check."""
    for i in range(0, len(edges), BATCH_WRITE_SIZE):
        batch = edges[i:i + BATCH_WRITE_SIZE]
        for src, tgt, sim in batch:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO derivation_edges (source_id, target_id, weight, relation_type) "
                    "VALUES (?, ?, ?, 'semantic_similarity')",
                    (src, tgt, sim),
                )
            except Exception:
                pass


def _merge_node_into(conn: sqlite3.Connection, discard_id: str, keep_id: str) -> None:
    """Redirect all edges from discard_id to keep_id, then mark discard as inactive."""
    conn.execute(
        "UPDATE derivation_edges SET source_id = ? WHERE source_id = ?",
        (keep_id, discard_id),
    )
    conn.execute(
        "UPDATE derivation_edges SET target_id = ? WHERE target_id = ?",
        (keep_id, discard_id),
    )
    # Merge access count
    discard_acc = conn.execute("SELECT access_count FROM thought_nodes WHERE id = ?", (discard_id,)).fetchone()
    keep_acc = conn.execute("SELECT access_count FROM thought_nodes WHERE id = ?", (keep_id,)).fetchone()
    combined = (discard_acc["access_count"] or 0) + (keep_acc["access_count"] or 0) if discard_acc and keep_acc else 0
    conn.execute("UPDATE thought_nodes SET access_count = ? WHERE id = ?", (combined, keep_id))
    conn.execute("UPDATE thought_nodes SET is_active = 0 WHERE id = ?", (discard_id,))


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize base64 → zlib → float16 → float32 embedding."""
    import base64
    import zlib
    raw = zlib.decompress(base64.b64decode(blob))
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


def _insert_zeros_embedding(conn: sqlite3.Connection, node_id: str) -> None:
    """Create a zero-vector embedding for a node that lacks one."""
    try:
        conn.execute(
            "INSERT OR IGNORE INTO vec_embeddings (node_id, embedding) VALUES (?, ?)",
            (node_id, _serialize_embedding(np.zeros(384, dtype=np.float32))),
        )
    except Exception:
        pass


def _serialize_embedding(arr: np.ndarray) -> bytes:
    """Serialize float32 array → float16 → zlib → base64."""
    import base64
    import zlib
    return base64.b64encode(zlib.compress(arr.astype(np.float16).tobytes()))
