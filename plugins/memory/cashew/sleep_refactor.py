# plugins/memory/cashew/sleep_refactor.py
"""Refactored Cashew sleep cycle — batch-scalable memory consolidation.

|Replaces the upstream O(N²) sleep cycle with vectorized numpy similarity
|search, batched DB writes, and a bounded work cap.  Designed to run at
|session end via lifecycle hooks (under 5 seconds for 7K nodes) rather
|than as a standalone cron-scheduled heavyweight.
|
|The ``background_dream`` parameter (added in v0.10.0) moves the LLM-powered
|dream generation (Phase 8) and orphan embedding (Phase 9) into a daemon
|thread so the lifecycle hook returns promptly — the cross-linking, dedup, GC,
|and core memory phases still run synchronously in ~20s, but the ~60s LLM
|call no longer blocks the session boundary.

|Usage (from CashewMemoryProvider):
|    from .sleep_refactor import run_sleep_cycle

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

import fcntl
import logging
import math
import random
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── thresholds ──────────────────────────────────────────────────────────────
CROSS_LINK_THRESHOLD = 0.78   # cosine similarity above which nodes get cross-linked
DEDUP_THRESHOLD = 0.82        # cosine similarity above which nodes are considered duplicates
MAX_NODES_PER_CYCLE = 2000    # work cap per cycle
MAX_EDGES_PER_CYCLE = 100_000 # edge cap per cycle
EDGES_PER_BATCH = 500         # commit after this many edge inserts
GC_K_NODES = 50               # random sample size for garbage collection
GC_THRESHOLD = 0.0            # fitness threshold for GC (0 = decay isolated nodes)
GC_GRACE_DAYS = 7             # min node age (days) before GC can decay it


# ── helpers ─────────────────────────────────────────────────────────────────

def _set_wal(conn: sqlite3.Connection) -> None:
    """Enable WAL mode if not already active."""
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if mode.lower() != "wal":
        logger.info("sleep: switching journal_mode %s → wal", mode)
        conn.execute("PRAGMA journal_mode=WAL")


def _load_embedding_matrix(
    conn: sqlite3.Connection, node_ids: list[str],
) -> tuple[list[str], np.ndarray]:
    """Load embeddings for *node_ids* from the `embeddings` table.

    Returns (valid_ids, matrix) where *matrix* has shape (N, embedding_dim).
    Filters NaN, inf, and zero vectors.
    """
    if not node_ids:
        return [], np.array([])

    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT e.node_id, e.vector FROM embeddings e "
        f"WHERE e.node_id IN ({placeholders})",
        node_ids,
    ).fetchall()

    vectors: list[np.ndarray] = []
    valid_ids: list[str] = []
    bad = 0
    for nid, blob in rows:
        try:
            vec = np.frombuffer(blob, dtype=np.float32)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                bad += 1
                continue
            if np.allclose(vec, 0):
                bad += 1
                continue
            valid_ids.append(nid)
            vectors.append(vec)
        except Exception:
            bad += 1

    if bad:
        logger.warning("sleep: skipped %d bad embeddings", bad)
    if not vectors:
        return [], np.array([])
    return valid_ids, np.array(vectors)


# ── Phase 1: candidate discovery (vectorized) ───────────────────────────────

def _find_candidates(
    ids: list[str], matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cross_link_pairs, dedup_pairs, similarity_matrix).

    Each pair array has shape (K, 2) of indices into *ids*.
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

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

    logger.info(
        "sleep: %d cross-link + %d dedup candidates "
        "(%d total / %d pairs)",
        len(cross_pairs), len(dedup_pairs),
        len(cross_pairs) + len(dedup_pairs),
        len(ids) * (len(ids) - 1) // 2,
    )
    return cross_pairs, dedup_pairs, sim


# ── Phase 2: batched cross-linking ──────────────────────────────────────────

def _batch_cross_links(
    conn: sqlite3.Connection,
    ids: list[str],
    cross_pairs: np.ndarray,
    sim: np.ndarray,
    source_files: dict[str, str] | None = None,
    max_edges: int | None = None,
) -> dict:
    """Insert cross-link edges in batches. Returns stats dict.

    When *source_files* is provided, pairs whose nodes share the same
    source_file are skipped (counted in ``same_source_skipped``).
    When *max_edges* is set, stops creating edges after reaching the cap.
    """
    stats = {
        "candidates": len(cross_pairs),
        "created": 0,
        "skipped": 0,
        "same_source_skipped": 0,
        "capped": False,
    }
    pending: list[tuple[str, str, float]] = []
    t0 = time.perf_counter()

    for batch_start in range(0, len(cross_pairs), EDGES_PER_BATCH):
        batch = cross_pairs[batch_start:batch_start + EDGES_PER_BATCH]
        for i, j in batch:
            if max_edges is not None and stats["created"] >= max_edges:
                stats["capped"] = True
                break
            n1 = ids[int(i)]
            n2 = ids[int(j)]
            # Same-source check
            if source_files is not None:
                sf1 = source_files.get(n1, "")
                sf2 = source_files.get(n2, "")
                if sf1 and sf2 and sf1 == sf2:
                    stats["same_source_skipped"] += 1
                    continue
            row = conn.execute(
                "SELECT COUNT(*) FROM derivation_edges "
                "WHERE (parent_id=? AND child_id=?) OR (parent_id=? AND child_id=?)",
                (n1, n2, n2, n1),
            ).fetchone()
            if row[0] > 0:
                stats["skipped"] += 1
                continue
            sim_val = float(sim[int(i), int(j)])
            pending.append((n1, n2, sim_val))
            pending.append((n2, n1, sim_val))
            stats["created"] += 1

        if max_edges is not None and stats["created"] >= max_edges:
            stats["capped"] = True
            break

        if pending:
            conn.executemany(
                "INSERT OR IGNORE INTO derivation_edges "
                "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
                [
                    (p, c, w, f"cross_link - similarity={w:.3f}")
                    for p, c, w in pending
                ],
            )
            conn.commit()
        pending.clear()

    elapsed = time.perf_counter() - t0
    logger.info(
        "sleep: cross-links %d created, %d skipped in %.1fs",
        stats["created"], stats["skipped"], elapsed,
    )
    return stats


# ── Phase 3: dedup via connected components ─────────────────────────────────

def _merge_cluster(
    conn: sqlite3.Connection,
    cluster_ids: list[str],
) -> Optional[str]:
    """Merge a cluster of near-duplicate nodes into the keeper.

    Keeper: highest access_count (tiebreak oldest timestamp).
    Rewires edges via read→delete→reinsert, decays losers.
    """
    if len(cluster_ids) < 2:
        return None

    placeholders = ",".join("?" * len(cluster_ids))
    keeper = conn.execute(
        f"SELECT id FROM thought_nodes WHERE id IN ({placeholders}) "
        "ORDER BY COALESCE(access_count, 0) DESC, "
        "COALESCE(timestamp, '9999') ASC LIMIT 1",
        cluster_ids,
    ).fetchone()
    if not keeper:
        return None

    keeper_id = keeper[0]
    losers = [n for n in cluster_ids if n != keeper_id]
    cluster_set = set(cluster_ids)
    all_p = ",".join("?" * len(cluster_ids))

    # Read all edges touching any cluster member
    edges = conn.execute(
        f"SELECT parent_id, child_id, weight, reasoning "
        f"FROM derivation_edges "
        f"WHERE parent_id IN ({all_p}) OR child_id IN ({all_p})",
        cluster_ids + cluster_ids,
    ).fetchall()

    # Delete all edges touching cluster members
    conn.execute(
        f"DELETE FROM derivation_edges "
        f"WHERE parent_id IN ({all_p}) OR child_id IN ({all_p})",
        cluster_ids + cluster_ids,
    )

    # Re-insert rewired (skip self-loops)
    for parent_id, child_id, weight, reasoning in edges:
        new_parent = keeper_id if parent_id in cluster_set else parent_id
        new_child = keeper_id if child_id in cluster_set else child_id
        if new_parent == new_child:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
            (new_parent, new_child, weight, reasoning),
        )

    # Decay losers (soft-delete)
    loser_placeholders = ",".join("?" * len(losers))
    conn.execute(
        f"UPDATE thought_nodes SET decayed=1 WHERE id IN ({loser_placeholders})",
        losers,
    )
    return keeper_id


def _run_dedup(
    conn: sqlite3.Connection,
    ids: list[str],
    dedup_pairs: np.ndarray,
) -> dict:
    """Build dedup graph, extract connected components, merge each."""
    stats = {"components": 0, "nodes_merged": 0}
    if len(dedup_pairs) == 0:
        return stats

    # Build adjacency
    adj: dict[str, set[str]] = defaultdict(set)
    for i, j in dedup_pairs:
        adj[ids[int(i)]].add(ids[int(j)])
        adj[ids[int(j)]].add(ids[int(i)])

    # BFS connected components
    visited: set[str] = set()
    components: list[list[str]] = []
    for node in adj:
        if node not in visited:
            queue = [node]
            visited.add(node)
            component = [node]
            while queue:
                cur = queue.pop(0)
                for neighbor in adj[cur]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)
            if len(component) >= 2:
                components.append(component)

    logger.info("sleep: %d dedup components to merge", len(components))

    for comp in components:
        result = _merge_cluster(conn, comp)
        if result:
            stats["components"] += 1
            stats["nodes_merged"] += len(comp) - 1

    if stats["components"] > 0:
        conn.commit()

    logger.info(
        "sleep: dedup %d components merged, %d nodes decayed",
        stats["components"], stats["nodes_merged"],
    )
    return stats


# ── Phase 4: node metrics ───────────────────────────────────────────────────

def _compute_metrics(conn: sqlite3.Connection) -> dict[str, dict]:
    """Compute branching factor + cross-link count for all active nodes."""
    t0 = time.perf_counter()
    rows = conn.execute("""
        SELECT
            tn.id,
            (SELECT COUNT(*) FROM derivation_edges
             WHERE parent_id = tn.id) AS branching,
            (SELECT COUNT(*) FROM derivation_edges
             WHERE (parent_id = tn.id OR child_id = tn.id)
               AND reasoning LIKE '%cross_link%') AS cross_links
        FROM thought_nodes tn
        WHERE (tn.decayed IS NULL OR tn.decayed = 0)
    """).fetchall()

    metrics = {}
    for nid, branching, cross_links in rows:
        metrics[nid] = {
            "branching_factor": branching or 0,
            "cross_links": cross_links or 0,
            "fitness": float((branching or 0) + (cross_links or 0) * 0.5),
        }

    logger.debug(
        "sleep: metrics computed for %d nodes in %.1fs",
        len(metrics), time.perf_counter() - t0,
    )
    return metrics


# ── Phase 5: garbage collection ─────────────────────────────────────────────

def _garbage_collect(
    conn: sqlite3.Connection,
    metrics: dict[str, dict],
) -> int:
    """Randomly sample K non-permanent nodes, decay those below threshold."""
    if not metrics:
        return 0

    perm_ids = {
        r[0] for r in conn.execute(
            "SELECT id FROM thought_nodes "
            "WHERE permanent=1 AND (decayed IS NULL OR decayed=0)"
        ).fetchall()
    }

    candidates = [
        nid for nid, m in metrics.items()
        if nid not in perm_ids and m["fitness"] <= GC_THRESHOLD
    ]

    # Age gate: exclude nodes created within the grace period
    if candidates and GC_GRACE_DAYS > 0:
        import datetime
        cutoff = (
            datetime.datetime.utcnow() - datetime.timedelta(days=GC_GRACE_DAYS)
        ).isoformat()
        ph = ",".join("?" * len(candidates))
        young = {
            r[0] for r in conn.execute(
                f"SELECT id FROM thought_nodes WHERE id IN ({ph}) AND timestamp >= ?",
                [*candidates, cutoff],
            ).fetchall()
        }
        candidates = [nid for nid in candidates if nid not in young]

    sample = (
        candidates
        if len(candidates) <= GC_K_NODES
        else random.sample(candidates, GC_K_NODES)
    )
    if not sample:
        return 0

    placeholders = ",".join("?" * len(sample))
    conn.execute(
        f"UPDATE thought_nodes SET decayed=1 WHERE id IN ({placeholders})",
        sample,
    )
    conn.commit()

    logger.info("sleep: GC decayed %d low-fitness nodes", len(sample))
    return len(sample)


# ── Phase 6: permanence evaluation ──────────────────────────────────────────

def _evaluate_permanence(conn: sqlite3.Connection) -> dict:
    """Promote nodes with access_count >= 10 to permanent status."""
    try:
        from core.permanence import promote_permanent_nodes
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        if not db_path:
            # :memory: or temp database — use direct SQL
            raise ImportError("in-memory DB")
        stats = promote_permanent_nodes(db_path, access_threshold=10)
        logger.info(
            "sleep: permanence promoted %d nodes (threshold=10)",
            stats.get("nodes_promoted", 0),
        )
        return stats
    except ImportError:
        cur = conn.execute(
            "UPDATE thought_nodes SET permanent=1 "
            "WHERE access_count >= 10 "
            "AND (permanent IS NULL OR permanent = 0) "
            "AND (decayed IS NULL OR decayed = 0)",
        )
        count = cur.rowcount
        conn.commit()
        logger.info("sleep: permanence promoted %d nodes", count)
        return {"nodes_promoted": count, "threshold": 10}


# ── Phase 7: core memory promotion ──────────────────────────────────────────

def _promote_core_memories(
    conn: sqlite3.Connection,
    metrics: dict[str, dict],
) -> dict:
    """Top √N nodes by fitness become core_memory + permanent."""
    if not metrics:
        return {"promoted": 0, "demoted": 0}

    curr = {
        r[0] for r in conn.execute(
            "SELECT id FROM thought_nodes WHERE node_type='core_memory'"
        ).fetchall()
    }

    ranked = sorted(metrics.items(), key=lambda x: x[1]["fitness"], reverse=True)
    target = int(math.sqrt(len(metrics)))
    should_be = {nid for nid, _ in ranked[:target]}
    promoted = should_be - curr
    demoted = curr - should_be

    if promoted:
        pp = ",".join("?" * len(promoted))
        conn.execute(
            f"UPDATE thought_nodes SET node_type='core_memory', permanent=1 "
            f"WHERE id IN ({pp})",
            list(promoted),
        )

    conn.execute(
        "UPDATE thought_nodes SET permanent=1 "
        "WHERE node_type='core_memory' AND (permanent IS NULL OR permanent = 0)"
    )

    if demoted:
        dp = ",".join("?" * len(demoted))
        conn.execute(
            f"UPDATE thought_nodes SET node_type='derived' "
            f"WHERE id IN ({dp}) AND node_type != 'seed'",
            list(demoted),
        )

    conn.commit()
    logger.info(
        "sleep: core memory %d promoted, %d demoted (target=%d)",
        len(promoted), len(demoted), target,
    )
    return {"promoted": len(promoted), "demoted": len(demoted), "target": target}


# ── Phase 8: dream generation ───────────────────────────────────────────────

def _generate_dream(
    conn: sqlite3.Connection,
    cross_link_tuples: list[tuple[str, str, float]],
    model_fn=None,
) -> Optional[str]:
    """LLM-powered dream node bridging the strongest cross-source pair."""
    if not cross_link_tuples or model_fn is None:
        return None

    # Find cross-links bridging different source files
    bridge_candidates = []
    for n1, n2, sim in cross_link_tuples:
        sources = conn.execute(
            "SELECT source_file FROM thought_nodes WHERE id IN (?, ?)",
            (n1, n2),
        ).fetchall()
        srcs = [r[0] for r in sources]
        if len({s for s in srcs if s}) > 1:
            bridge_candidates.append((n1, n2, sim))

    if not bridge_candidates:
        return None

    best = max(bridge_candidates, key=lambda x: x[2])
    n1, n2, sim = best

    nodes = conn.execute(
        "SELECT content, node_type FROM thought_nodes WHERE id IN (?, ?)",
        (n1, n2),
    ).fetchall()
    if len(nodes) != 2:
        return None

    content1, type1 = nodes[0]
    content2, type2 = nodes[1]

    prompt = (
        "Two thought-snippets surfaced from the same body of work. They were "
        "embedded close in vector space, suggesting they share something. Read "
        "them and find what they JOINTLY point at: a shared assumption, a hidden "
        "invariant, a recurring failure mode, a deeper principle, or a contradiction. "
        "Output ONE statement, in plain prose, that captures the synthesis. "
        "Be specific. Name the concrete thing they share. If they don't share "
        "anything meaningful, output a one-line note about WHY the embedding "
        "linked them anyway (lexical overlap, structural similarity, etc.).\n\n"
        "Rules: no preamble, no headers, no markdown. Output only the synthesis "
        "statement, on a single line.\n\n"
        f"SNIPPET A ({type1}):\n{content1}\n\n"
        f"SNIPPET B ({type2}):\n{content2}\n"
    )

    try:
        response = model_fn(prompt)
        if not response:
            return None
        dream_content = response.strip().splitlines()[0].strip()
        if len(dream_content) < 20:
            return None
    except Exception as e:
        logger.warning("sleep: dream LLM synthesis failed: %s", e)
        return None

    import hashlib
    dream_id = hashlib.sha256(dream_content.encode()).hexdigest()[:12]

    conn.execute(
        "INSERT OR REPLACE INTO thought_nodes "
        "(id, content, node_type, timestamp, mood_state, metadata, source_file) "
        "VALUES (?, ?, 'dream', datetime('now'), 'dreamy', '{}', 'sleep_protocol')",
        (dream_id, dream_content),
    )
    conn.execute(
        "INSERT OR IGNORE INTO derivation_edges "
        "(parent_id, child_id, weight, reasoning) "
        "VALUES (?, ?, ?, 'derived_from - Dream synthesis')",
        (n1, dream_id, sim),
    )
    conn.execute(
        "INSERT OR IGNORE INTO derivation_edges "
        "(parent_id, child_id, weight, reasoning) "
        "VALUES (?, ?, ?, 'derived_from - Dream synthesis')",
        (n2, dream_id, sim),
    )
    conn.commit()

    logger.info(
        "sleep: dream node %s bridging %s... ↔ %s...",
        dream_id, n1[:8], n2[:8],
    )
    return dream_id


# ── Phase 9: embedding gap closure ──────────────────────────────────────────

def _embed_orphans(conn: sqlite3.Connection, embedding_model: str = "thenlper/gte-large") -> int:
    """Embed any active nodes lacking an embedding row. Returns count."""
    rows = conn.execute(
        "SELECT tn.id, tn.content FROM thought_nodes tn "
        "LEFT JOIN embeddings e ON tn.id = e.node_id "
        "WHERE e.node_id IS NULL "
        "AND (tn.decayed IS NULL OR tn.decayed = 0) "
        "AND tn.content IS NOT NULL AND TRIM(tn.content) != ''"
    ).fetchall()

    if not rows:
        return 0

    logger.info("sleep: embedding %d orphaned nodes with %s...", len(rows), embedding_model)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model)

    embedded = 0
    for nid, content in rows:
        try:
            vec = model.encode(content, normalize_embeddings=True)
            blob = vec.astype(np.float32).tobytes()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(node_id, vector, model, updated_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (nid, blob, embedding_model),
                )
            except sqlite3.OperationalError:
                # Fallback for legacy schemas without model/updated_at columns
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?, ?)",
                    (nid, blob),
                )
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings (node_id, embedding) VALUES (?, ?)",
                    (nid, vec.astype(np.float32).tolist()),
                )
            except sqlite3.OperationalError:
                pass
            embedded += 1
        except Exception as e:
            logger.warning("sleep: failed to embed node %s: %s", nid[:8], e)

    conn.commit()
    logger.info("sleep: embedded %d orphaned nodes", embedded)
    return embedded


# ── main entry point ────────────────────────────────────────────────────────

def _run_dream_async(
    db_path: str,
    cross_link_tuples: list[tuple[str, str, float]],
    model_fn,
    embedding_model: str = "thenlper/gte-large",
) -> None:
    """Run Phase 8 (dream) + Phase 9 (orphan embedding) in a daemon thread.

    Opens its own SQLite connection — WAL mode handles concurrency with
    the new session's sync worker writes.
    """
    def _task():
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout=5000")
            _set_wal(conn)
            dream_id = _generate_dream(conn, cross_link_tuples, model_fn=model_fn)
            orphans = _embed_orphans(conn, embedding_model=embedding_model)
            conn.close()
            logger.info(
                "sleep: background dream complete (id=%s, orphans=%d)",
                dream_id or "none", orphans,
            )
        except Exception:
            logger.warning("sleep: background dream failed", exc_info=True)

    t = threading.Thread(target=_task, daemon=True)
    t.start()
    logger.debug("sleep: background dream thread spawned")


def run_sleep_cycle(
    db_path: str,
    limit: int = MAX_NODES_PER_CYCLE,
    model_fn=None,
    background_dream: bool = False,
    embedding_model: str = "thenlper/gte-large",
) -> dict:
    """Run one complete refactored sleep cycle.

    Args:
        db_path: Path to the Cashew SQLite database.
        limit: Maximum number of nodes to cross-link this cycle.
        model_fn: Optional callable(str) -> str for LLM-powered dream generation.
        background_dream: When True, Phase 8 (dream) and Phase 9 (orphan
            embedding) run in a daemon thread instead of blocking the caller.
            The LLM call is the dominant latency (~60s); this lets the lifecycle
            hook return promptly after the ~20s synchronous path.

    Returns:
        Dict with statistics for each phase. When *background_dream* is True,
        the dict includes ``dream_pending=True`` and the ``dream_id`` field
        will be None (it may or may not complete before the caller reads it).
    """
    # Acquire advisory lock to prevent concurrent sleep cycles across sessions.
    # Non-blocking: if another process holds the lock, skip this cycle.
    lock_path = db_path + ".sleep.lock"
    try:
        lock_fd = open(lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info("sleep: another cycle is already running — skipping")
        return {}

    t_start = time.perf_counter()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    _set_wal(conn)

    # Select nodes for this cycle (lowest-degree-first heuristic)
    rows = conn.execute(
        "SELECT e.node_id FROM embeddings e "
        "JOIN thought_nodes tn ON e.node_id = tn.id "
        "WHERE (tn.decayed IS NULL OR tn.decayed = 0) "
        "ORDER BY ("
        "  SELECT COUNT(*) FROM derivation_edges "
        "  WHERE parent_id = e.node_id OR child_id = e.node_id"
        ") ASC, tn.timestamp ASC "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    ids = [r[0] for r in rows]
    logger.info("sleep: selected %d nodes (limit=%d)", len(ids), limit)

    valid_ids, matrix = _load_embedding_matrix(conn, ids)
    if len(valid_ids) < 2:
        logger.warning("sleep: too few valid embeddings — aborting")
        conn.close()
        return {"error": "too few nodes", "nodes_selected": len(ids)}

    # Phase 1: candidate discovery
    cross_pairs, dedup_pairs, sim = _find_candidates(valid_ids, matrix)

    # Phase 2: cross-linking
    cross_stats = {"created": 0, "skipped": 0}
    cross_link_tuples: list[tuple[str, str, float]] = []
    if len(cross_pairs) > 0:
        # Resolve source_file for each valid node so _batch_cross_links can
        # skip same-document pairs (which carry no graph-discovery value).
        source_files: dict[str, str] = {}
        for row in conn.execute(
            "SELECT id, source_file FROM thought_nodes "
            "WHERE id IN ({})".format(
                ",".join(["?"] * len(valid_ids))
            ),
            valid_ids,
        ).fetchall():
            nid, sf = row
            if sf:
                source_files[nid] = sf
        cross_stats = _batch_cross_links(
            conn, valid_ids, cross_pairs, sim,
            source_files=source_files or None,
        )
        if model_fn is not None:
            for i, j in cross_pairs:
                cross_link_tuples.append((
                    valid_ids[int(i)], valid_ids[int(j)],
                    float(sim[int(i), int(j)]),
                ))

    # Phase 3: dedup
    dedup_stats = {"components": 0, "nodes_merged": 0}
    if len(dedup_pairs) > 0:
        dedup_stats = _run_dedup(conn, valid_ids, dedup_pairs)

    # Phase 4: metrics
    metrics = _compute_metrics(conn)

    # Phase 5: garbage collection
    gc_count = _garbage_collect(conn, metrics)

    # Phase 6: permanence
    perm_stats = _evaluate_permanence(conn)

    # Phase 7: core memory
    core_stats = _promote_core_memories(conn, metrics)

    # Phase 8: dream generation
    dream_id = None
    dream_pending = False
    dream_status = "skipped"  # default when model_fn is None or no cross-source pairs
    if model_fn is not None and cross_link_tuples:
        if background_dream:
            _run_dream_async(
                db_path=db_path,
                cross_link_tuples=cross_link_tuples,
                model_fn=model_fn,
                embedding_model=embedding_model,
            )
            dream_pending = True
            dream_status = "pending"
        else:
            dream_id = _generate_dream(conn, cross_link_tuples, model_fn=model_fn)
            dream_status = "ran" if dream_id else "failed"
    elif model_fn is not None and not cross_link_tuples:
        dream_status = "skipped"  # model available but no pairs

    # Phase 9: embed orphans
    if background_dream:
        orphans = 0  # handled by background dream thread
    else:
        orphans = _embed_orphans(conn, embedding_model=embedding_model)

    conn.close()
    elapsed = round(time.perf_counter() - t_start, 1)

    summary = {
        "nodes_selected": len(ids),
        "nodes_with_embeddings": len(valid_ids),
        "cross_link_candidates": len(cross_pairs),
        "dedup_candidates": len(dedup_pairs),
        "cross_links_created": cross_stats["created"],
        "cross_links_skipped": cross_stats["skipped"],
        "cross_link_same_source_skipped": cross_stats.get("same_source_skipped", 0),
        "cross_link_capped": cross_stats.get("capped", False),
        "dedup_components": dedup_stats["components"],
        "dedup_nodes_merged": dedup_stats["nodes_merged"],
        "nodes_gc_decayed": gc_count,
        "nodes_made_permanent": perm_stats.get("nodes_promoted", 0),
        "core_promoted": core_stats.get("promoted", 0),
        "core_demoted": core_stats.get("demoted", 0),
        "dream_id": dream_id,
        "dream_pending": dream_pending,
        "dream_generation": dream_status,
        "orphans_embedded": orphans,
        "total_nodes": len(metrics),
        "elapsed_s": elapsed,
    }

    if dream_pending:
        logger.info(
            "sleep: sync phases complete in %.1fs — %d nodes, %d cross-links, %d dedups, "
            "%d GC, %d permanent, %d core (dream pending in background)",
            elapsed,
            summary["total_nodes"],
            summary["cross_links_created"],
            summary["dedup_nodes_merged"],
            summary["nodes_gc_decayed"],
            summary["nodes_made_permanent"],
            summary["core_promoted"],
        )
    else:
        logger.info(
            "sleep: cycle complete in %.1fs — %d nodes, %d cross-links, %d dedups, "
            "%d GC, %d permanent, %d core, dream=%s, %d embedded",
            elapsed,
            summary["total_nodes"],
            summary["cross_links_created"],
            summary["dedup_nodes_merged"],
            summary["nodes_gc_decayed"],
            summary["nodes_made_permanent"],
            summary["core_promoted"],
            dream_status,
            summary["orphans_embedded"],
        )
    return summary
