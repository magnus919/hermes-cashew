"""Replacement-safe consolidation contracts for issues #192 and #193.

These tests exercise the adapter's scheduled ``run_sleep_cycle`` boundary
against temporary, production-shaped SQLite databases.  The current engine's
known failures are strict xfails: only an exact observed defect raises
``KnownConsolidationDebt``.  Setup, schema, import, and unrelated assertion
failures therefore remain ordinary test failures.
"""

from __future__ import annotations

import importlib.machinery
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlite_vec

import plugins.memory.cashew.sleep_refactor as sleep

# The fast suite intentionally supports running without cashew-brain. These
# contracts require its real schema and calibrated profiles. Guard only the
# confirmed-absent case; a present but broken installation must fail import.
if importlib.machinery.PathFinder.find_spec("core") is None:
    pytest.skip(
        "cashew-brain is required for consolidation safety contracts",
        allow_module_level=True,
    )

from core.db import ensure_schema
from core.model_profiles import get_profile

GTE_LARGE = "thenlper/gte-large"
_REAL_CONNECT = sqlite3.connect
_PAIR_BATCH = getattr(sleep, "EDGES_PER_BATCH", 500)


class KnownConsolidationDebt(Exception):  # noqa: N818 - domain marker required by #192
    """An exact, verified #193 defect; unrelated failures must not match it."""


def _new_brain(tmp_path: Path, name: str, *, vec_dim: int | None = None) -> Path:
    """Create a temporary brain with upstream schema and optional real vec0."""
    db_path = tmp_path / f"{name}.db"
    ensure_schema(str(db_path))
    conn = _REAL_CONNECT(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decay_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                content_summary TEXT,
                decay_reason TEXT NOT NULL,
                confidence_at_decay REAL,
                access_count_at_decay INTEGER,
                last_access_date TEXT,
                related_nodes TEXT,
                source_file TEXT,
                domain TEXT,
                node_type TEXT,
                decay_timestamp TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        if vec_dim is not None:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0("
                "node_id TEXT PRIMARY KEY, "
                f"embedding float[{vec_dim}] distance_metric=cosine)"
            )
            schema = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'vec_embeddings'"
            ).fetchone()
            assert schema is not None and schema[0] is not None
            canonical_schema = re.sub(r'[\s"`]+', "", schema[0].lower())
            assert "createvirtualtablevec_embeddingsusingvec0(" in canonical_schema
            assert "node_idtextprimarykey" in canonical_schema
            assert (
                f"embeddingfloat[{vec_dim}]distance_metric=cosine" in canonical_schema
            )
            assert [
                row[1] for row in conn.execute("PRAGMA table_info(vec_embeddings)")
            ] == ["node_id", "embedding"]
        conn.commit()
    finally:
        conn.close()
    return db_path


def _insert_node(
    conn: sqlite3.Connection,
    node_id: str,
    content: str,
    *,
    timestamp: str = "2099-01-01T00:00:00+00:00",
    last_accessed: str | None = None,
    access_count: int = 0,
    permanent: int = 0,
    tags: str | None = None,
    source_file: str = "contract",
) -> None:
    conn.execute(
        """
        INSERT INTO thought_nodes (
            id, content, node_type, domain, timestamp, access_count,
            last_accessed, source_file, decayed, metadata, permanent, tags
        ) VALUES (?, ?, 'observation', 'test', ?, ?, ?, ?, 0, '{}', ?, ?)
        """,
        (
            node_id,
            content,
            timestamp,
            access_count,
            last_accessed,
            source_file,
            permanent,
            tags,
        ),
    )


def _insert_embedding(
    conn: sqlite3.Connection,
    node_id: str,
    vector: np.ndarray,
    *,
    index: bool = False,
) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    blob = vector.tobytes()
    conn.execute(
        "INSERT INTO embeddings (node_id, vector, model, updated_at) "
        "VALUES (?, ?, ?, '2026-09-12T00:00:00+00:00')",
        (node_id, blob, GTE_LARGE),
    )
    if index:
        conn.execute(
            "INSERT INTO vec_embeddings (node_id, embedding) VALUES (?, ?)",
            (node_id, blob),
        )


def _two_vectors(similarity: float) -> tuple[np.ndarray, np.ndarray]:
    assert 0.0 <= similarity <= 1.0
    complement = math.sqrt(1.0 - similarity**2)
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([similarity, complement], dtype=np.float32)
    assert float(np.dot(left, right)) == pytest.approx(similarity, abs=1e-6)
    return left, right


def _run(
    db_path: Path,
    *,
    limit: int,
    max_edges: int,
    embedding_client: Any = None,
) -> dict[str, Any]:
    result = sleep.run_sleep_cycle(
        str(db_path),
        limit=limit,
        max_edges=max_edges,
        model_fn=None,
        background_dream=False,
        embedding_model=GTE_LARGE,
        embedding_device="cpu",
        embedding_client=embedding_client,
    )
    assert isinstance(result, dict)
    assert "error" not in result
    return result


def _node_states(
    conn: sqlite3.Connection,
) -> dict[str, tuple[int, int, str, str | None]]:
    return {
        node_id: (decayed or 0, permanent or 0, content, tags)
        for node_id, decayed, permanent, content, tags in conn.execute(
            "SELECT id, decayed, permanent, content, tags FROM thought_nodes"
        )
    }


@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must protect permanence 1/2 across GC and post-GC promotion",
)
def test_scheduled_cycle_preserves_permanence_and_gc_ordering(tmp_path: Path) -> None:
    """GC does real work without decaying pinned, accessed, or promoted nodes."""
    db_path = _new_brain(tmp_path, "permanence-gc")
    conn = _REAL_CONNECT(db_path)
    dim = 64
    entries: list[tuple[str, dict[str, Any]]] = [
        ("eligible", {}),
        ("permanent-one", {"permanent": 1}),
        ("permanent-two", {"permanent": 2}),
        ("promote-after-gc", {"access_count": 10}),
        ("accessed-isolated", {"access_count": 8}),
        (
            "temporal-tagged",
            {
                "content": "Review the decision on September 12, 2026.",
                "last_accessed": "2099-01-01T00:00:00+00:00",
                "tags": "calendar,decision",
            },
        ),
    ]
    entries.extend((f"support-{i:02d}", {"permanent": 1}) for i in range(46))
    assert len(entries) > 50
    for index, (node_id, values) in enumerate(entries):
        content = values.pop("content", f"contract node {node_id}")
        _insert_node(
            conn,
            node_id,
            content,
            timestamp=values.pop("timestamp", "2020-01-01T00:00:00+00:00"),
            **values,
        )
        vector = np.zeros(dim, dtype=np.float32)
        vector[index] = 1.0
        _insert_embedding(conn, node_id, vector)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone() == (52,)
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone() == (52,)
    conn.close()

    result = _run(db_path, limit=52, max_edges=10)
    assert result["nodes_selected"] == 52
    assert result["nodes_with_embeddings"] == 52
    assert result["cross_link_candidates"] == 0
    assert result["dedup_candidates"] == 0
    assert result["nodes_gc_decayed"] > 0, "fixture must force real GC work"

    conn = _REAL_CONNECT(db_path)
    states = _node_states(conn)
    audit = conn.execute(
        "SELECT node_id, decay_reason FROM decay_audit ORDER BY node_id"
    ).fetchall()
    conn.close()

    # Exact local-engine signature: permanent=2 and load-bearing accessed rows
    # are collected; stale pre-GC metrics then promote decayed rows to permanent.
    if (
        states["permanent-one"][:2] == (0, 1)
        and states["permanent-two"][:2] == (1, 1)
        and states["promote-after-gc"][:2] == (1, 1)
        and states["accessed-isolated"][:2] == (1, 1)
        and states["eligible"][:2] == (1, 1)
        and states["temporal-tagged"][:2] == (1, 1)
        and audit == []
    ):
        raise KnownConsolidationDebt(
            "local GC decayed permanence=2/accessed rows and promoted decayed rows"
        )

    assert states["permanent-one"][:2] == (0, 1)
    assert states["permanent-two"][0] == 0
    assert states["permanent-two"][1] >= 1
    assert states["promote-after-gc"][:2] == (0, 1)
    assert states["accessed-isolated"][0] == 0
    assert states["temporal-tagged"][0] == 0
    assert states["temporal-tagged"][3] == "calendar,decision"
    assert states["eligible"][:2] == (1, 0)
    assert ("eligible", "gc_fitness") in audit
    assert all(
        not (decayed and permanent >= 1) for decayed, permanent, _, _ in states.values()
    )


@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must preserve one permanent canonical when pinned duplicates merge",
)
def test_scheduled_cycle_preserves_a_permanent_duplicate_canonical(
    tmp_path: Path,
) -> None:
    """Pinned duplicates may merge, but a live permanent canonical must remain."""
    db_path = _new_brain(tmp_path, "permanent-duplicates")
    conn = _REAL_CONNECT(db_path)
    left, right = _two_vectors(0.96)
    _insert_node(
        conn,
        "pinned-one",
        "September 12, 2026 decision record, primary copy.",
        permanent=1,
        access_count=100,
        tags="calendar,primary",
    )
    _insert_node(
        conn,
        "pinned-two",
        "September 12, 2026 decision record, pinned copy.",
        permanent=2,
        access_count=1,
        tags="calendar,pinned",
    )
    _insert_embedding(conn, "pinned-one", left)
    _insert_embedding(conn, "pinned-two", right)
    conn.commit()
    conn.close()

    result = _run(db_path, limit=2, max_edges=10)
    assert result["dedup_candidates"] == 1, "fixture must force dedup work"
    assert result["dedup_nodes_merged"] == 1

    conn = _REAL_CONNECT(db_path)
    states = _node_states(conn)
    audit = conn.execute(
        "SELECT node_id, decay_reason, related_nodes, metadata FROM decay_audit"
    ).fetchall()
    conn.close()
    pair = {node_id: states[node_id] for node_id in ("pinned-one", "pinned-two")}
    active = {node_id: row for node_id, row in pair.items() if row[0] == 0}
    decayed = {node_id: row for node_id, row in pair.items() if row[0] == 1}

    if (
        len(active) == 1
        and next(iter(active.values()))[1] >= 1
        and len(decayed) == 1
        and next(iter(decayed.values()))[1] >= 1
        and audit == []
    ):
        raise KnownConsolidationDebt(
            "local dedup leaves an absorbed pinned row both permanent and decayed"
        )

    assert len(active) == 1
    assert next(iter(active.values()))[1] >= 1
    assert "september 12, 2026" in next(iter(active.values()))[2].lower()
    assert "calendar" in (next(iter(active.values()))[3] or "").split(",")
    assert len(decayed) == 1
    assert next(iter(decayed.values()))[1] == 0
    assert len(audit) == 1
    assert audit[0][0] in decayed
    assert audit[0][1] == "dedup_loser"


@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must use nontransitive dedup groups at the scheduled boundary",
)
def test_scheduled_cycle_does_not_merge_a_nontransitive_duplicate_chain(
    tmp_path: Path,
) -> None:
    """A~B and B~C with A!~C leaves two canonical memories, regardless of keeper."""
    db_path = _new_brain(tmp_path, "nontransitive")
    angle = math.acos(0.95)
    vectors = (
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([math.cos(angle), math.sin(angle)], dtype=np.float32),
        np.array([math.cos(2 * angle), math.sin(2 * angle)], dtype=np.float32),
    )
    similarities = np.array(vectors) @ np.array(vectors).T
    assert similarities[0, 1] == pytest.approx(0.95, abs=1e-6)
    assert similarities[1, 2] == pytest.approx(0.95, abs=1e-6)
    assert similarities[0, 2] < 0.90

    conn = _REAL_CONNECT(db_path)
    for node_id, vector in zip(("a", "b", "c"), vectors, strict=True):
        _insert_node(conn, node_id, f"canonical {node_id}", permanent=1)
        _insert_embedding(conn, node_id, vector)
    conn.commit()
    conn.close()

    result = _run(db_path, limit=3, max_edges=10)
    assert result["dedup_candidates"] == 2, "fixture must force both eligible links"

    conn = _REAL_CONNECT(db_path)
    active_ids = {
        row[0]
        for row in conn.execute(
            "SELECT id FROM thought_nodes WHERE decayed IS NULL OR decayed=0"
        )
    }
    conn.close()

    if (
        result["cross_link_candidates"] == 1
        and result["dedup_nodes_merged"] == 2
        and len(active_ids) == 1
    ):
        raise KnownConsolidationDebt(
            "local connected-component dedup transitively merged all three nodes"
        )

    assert result["cross_link_candidates"] == 0
    assert result["dedup_nodes_merged"] == 1
    assert len(active_ids) == 2
    assert active_ids <= {"a", "b", "c"}


@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must resolve the configured gte-large 0.90/0.94 thresholds",
)
def test_scheduled_cycle_uses_configured_gte_large_thresholds(tmp_path: Path) -> None:
    """gte-large classifies 0.89 as unrelated, 0.92 as link, and 0.95 as dedup."""
    profile = get_profile(GTE_LARGE)
    assert profile.cross_link_threshold == pytest.approx(0.90)
    assert profile.dedup_threshold == pytest.approx(0.94)

    observed: dict[float, tuple[int, int, int, int]] = {}
    for similarity in (0.89, 0.92, 0.95):
        db_path = _new_brain(tmp_path, f"threshold-{similarity}")
        conn = _REAL_CONNECT(db_path)
        left, right = _two_vectors(similarity)
        for node_id, vector in (("left", left), ("right", right)):
            _insert_node(conn, node_id, node_id, permanent=1)
            _insert_embedding(conn, node_id, vector)
        conn.commit()
        conn.close()

        result = _run(db_path, limit=2, max_edges=10)
        conn = _REAL_CONNECT(db_path)
        directed_rows = conn.execute(
            "SELECT COUNT(*) FROM derivation_edges WHERE reasoning LIKE 'cross_link%'"
        ).fetchone()[0]
        active_rows = conn.execute(
            "SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed=0"
        ).fetchone()[0]
        conn.close()
        observed[similarity] = (
            result["cross_link_candidates"],
            result["dedup_candidates"],
            directed_rows,
            active_rows,
        )

    assert sum(cross + dedup for cross, dedup, _, _ in observed.values()) > 0
    legacy_fixed_thresholds = {
        0.89: (0, 1, 0, 1),
        0.92: (0, 1, 0, 1),
        0.95: (0, 1, 0, 1),
    }
    if observed == legacy_fixed_thresholds:
        raise KnownConsolidationDebt(
            "local 0.78/0.82 constants classify gte-large 0.89 and 0.92 as duplicates"
        )

    assert observed == {
        0.89: (0, 0, 0, 2),
        0.92: (1, 0, 2, 2),
        0.95: (0, 1, 0, 1),
    }


def _insert_disjoint_crosslink_pairs(
    conn: sqlite3.Connection,
    pair_count: int,
) -> None:
    dimension = pair_count * 2
    for pair_index in range(pair_count):
        left = np.zeros(dimension, dtype=np.float32)
        right = np.zeros(dimension, dtype=np.float32)
        left[pair_index] = 1.0
        right[pair_index] = 0.92
        right[pair_count + pair_index] = math.sqrt(1.0 - 0.92**2)
        for side, vector in (("a", left), ("b", right)):
            node_id = f"pair-{pair_index:04d}-{side}"
            _insert_node(conn, node_id, node_id, permanent=1)
            _insert_embedding(conn, node_id, vector)


_CAPPED_FLUSH_XFAIL = pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 capped private batch must flush every reported pair",
)


@pytest.mark.parametrize(
    "cap",
    [
        0,
        pytest.param(1, marks=_CAPPED_FLUSH_XFAIL),
        pytest.param(_PAIR_BATCH, marks=_CAPPED_FLUSH_XFAIL),
        pytest.param(_PAIR_BATCH + 1, marks=_CAPPED_FLUSH_XFAIL),
    ],
)
def test_component_crosslink_caps_match_stored_directed_rows(
    tmp_path: Path,
    cap: int,
) -> None:
    """Supplemental private probe isolates cap flushing from threshold policy."""
    pair_count = cap + 1
    db_path = _new_brain(tmp_path, f"component-cap-{cap}")
    conn = _REAL_CONNECT(db_path)
    ids: list[str] = []
    pairs: list[tuple[int, int]] = []
    similarity = np.eye(pair_count * 2, dtype=np.float32)
    for pair_index in range(pair_count):
        left_id = f"pair-{pair_index:04d}-a"
        right_id = f"pair-{pair_index:04d}-b"
        _insert_node(conn, left_id, left_id)
        _insert_node(conn, right_id, right_id)
        left_index = len(ids)
        ids.extend((left_id, right_id))
        pairs.append((left_index, left_index + 1))
        similarity[left_index, left_index + 1] = 0.92
        similarity[left_index + 1, left_index] = 0.92
    conn.commit()

    stats = sleep._batch_cross_links(  # noqa: SLF001 - intentional component probe
        conn,
        ids,
        np.asarray(pairs, dtype=int),
        similarity,
        max_edges=cap,
    )
    directed_rows = conn.execute(
        "SELECT COUNT(*) FROM derivation_edges WHERE reasoning LIKE 'cross_link%'"
    ).fetchone()[0]
    conn.close()

    assert stats["candidates"] == pair_count, "fixture must force eligible work"
    expected_rows = cap * 2
    if (
        stats["created"] == cap
        and stats["capped"] is True
        and directed_rows < expected_rows
    ):
        raise KnownConsolidationDebt(
            f"cap={cap} reported {cap} pairs but stored {directed_rows} directed rows"
        )

    assert stats["created"] == cap
    assert stats["skipped"] == 0
    assert stats["capped"] is True
    assert directed_rows == expected_rows


def test_component_crosslink_summary_accounts_for_existing_pair(tmp_path: Path) -> None:
    """Supplemental private probe proves pair counts and existing-edge idempotence."""
    db_path = _new_brain(tmp_path, "component-existing")
    conn = _REAL_CONNECT(db_path)
    ids = ["existing-a", "existing-b", "novel-a", "novel-b"]
    for node_id in ids:
        _insert_node(conn, node_id, node_id)
    for parent, child in (("existing-a", "existing-b"), ("existing-b", "existing-a")):
        conn.execute(
            "INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning) "
            "VALUES (?, ?, 0.92, 'cross_link - existing contract edge')",
            (parent, child),
        )
    conn.commit()
    similarity = np.eye(4, dtype=np.float32)
    similarity[0, 1] = similarity[1, 0] = 0.92
    similarity[2, 3] = similarity[3, 2] = 0.92

    stats = sleep._batch_cross_links(  # noqa: SLF001 - intentional component probe
        conn,
        ids,
        np.array([[0, 1], [2, 3]], dtype=int),
        similarity,
        max_edges=10,
    )
    directed_rows = conn.execute(
        "SELECT COUNT(*) FROM derivation_edges WHERE reasoning LIKE 'cross_link%'"
    ).fetchone()[0]
    conn.close()

    assert stats["candidates"] == 2
    assert stats["created"] == 1
    assert stats["skipped"] == 1
    assert stats["capped"] is False
    assert directed_rows == 4


@pytest.mark.parametrize("cap", [0, 1, _PAIR_BATCH, _PAIR_BATCH + 1])
@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must persist every reported capped pair as two directed rows",
)
def test_scheduled_cycle_crosslink_caps_match_stored_directed_rows(
    tmp_path: Path,
    cap: int,
) -> None:
    """Caps count pairs at 0/1/batch/batch+1; each created pair stores two rows."""
    pair_count = cap + 1
    db_path = _new_brain(tmp_path, f"cap-{cap}")
    conn = _REAL_CONNECT(db_path)
    _insert_disjoint_crosslink_pairs(conn, pair_count)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone() == (
        pair_count * 2,
    )
    conn.close()

    result = _run(db_path, limit=pair_count * 2, max_edges=cap)
    conn = _REAL_CONNECT(db_path)
    directed_rows = conn.execute(
        "SELECT COUNT(*) FROM derivation_edges WHERE reasoning LIKE 'cross_link%'"
    ).fetchone()[0]
    active_rows = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed=0"
    ).fetchone()[0]
    conn.close()

    assert result["cross_link_candidates"] + result["dedup_candidates"] > 0
    if (
        result["cross_link_candidates"] == 0
        and result["dedup_candidates"] == pair_count
        and result["dedup_nodes_merged"] == pair_count
        and directed_rows == 0
        and active_rows == pair_count
    ):
        raise KnownConsolidationDebt(
            f"local fixed thresholds routed all {pair_count} eligible links to dedup"
        )

    expected_pairs = cap
    expected_rows = expected_pairs * 2
    if (
        result["cross_link_candidates"] == pair_count
        and result["dedup_candidates"] == 0
        and result["cross_links_created"] == expected_pairs
        and result["cross_link_capped"] is True
        and directed_rows < expected_rows
    ):
        raise KnownConsolidationDebt(
            f"cap={cap} reported {expected_pairs} pairs but stored {directed_rows} rows"
        )

    assert result["cross_link_candidates"] == pair_count
    assert result["dedup_candidates"] == 0
    assert result["cross_links_created"] == expected_pairs
    assert result["cross_links_skipped"] == 0
    assert result["cross_link_capped"] is True
    assert directed_rows == expected_rows
    assert active_rows == pair_count * 2


@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must count existing crosslinks without duplicating directed rows",
)
def test_scheduled_cycle_crosslink_summary_accounts_for_existing_pair(
    tmp_path: Path,
) -> None:
    """An existing pair is skipped; one novel pair creates exactly two rows."""
    db_path = _new_brain(tmp_path, "existing-crosslink")
    conn = _REAL_CONNECT(db_path)
    _insert_disjoint_crosslink_pairs(conn, 2)
    for parent, child in (
        ("pair-0000-a", "pair-0000-b"),
        ("pair-0000-b", "pair-0000-a"),
    ):
        conn.execute(
            "INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning) "
            "VALUES (?, ?, 0.92, 'cross_link - existing contract edge')",
            (parent, child),
        )
    conn.commit()
    conn.close()

    result = _run(db_path, limit=4, max_edges=10)
    conn = _REAL_CONNECT(db_path)
    rows = conn.execute(
        "SELECT parent_id, child_id FROM derivation_edges "
        "WHERE reasoning LIKE 'cross_link%' ORDER BY parent_id, child_id"
    ).fetchall()
    conn.close()

    if (
        result["cross_link_candidates"] == 0
        and result["dedup_candidates"] == 2
        and result["dedup_nodes_merged"] == 2
        and rows == []
    ):
        raise KnownConsolidationDebt(
            "local fixed thresholds deduplicated both 0.92 pairs and removed existing edges"
        )

    assert result["cross_link_candidates"] == 2
    assert result["dedup_candidates"] == 0
    assert result["cross_links_created"] == 1
    assert result["cross_links_skipped"] == 1
    assert result["cross_link_capped"] is False
    assert len(rows) == 4
    assert {
        ("pair-0000-a", "pair-0000-b"),
        ("pair-0000-b", "pair-0000-a"),
    } <= set(rows)


def _patch_vec_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sqlite-vec available on each real connection opened by the cycle."""

    def connect_with_vec(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = _REAL_CONNECT(*args, **kwargs)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn

    monkeypatch.setattr(sleep.sqlite3, "connect", connect_with_vec)


@pytest.mark.xfail(
    raises=KnownConsolidationDebt,
    strict=True,
    reason="#193 must atomically dual-write valid orphan vectors and reject invalid vectors",
)
def test_scheduled_cycle_orphan_embeddings_keep_real_vec_index_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid float32 vectors reach both stores; empty/NaN vectors change neither."""
    db_path = _new_brain(tmp_path, "orphan-vectors", vec_dim=4)
    conn = _REAL_CONNECT(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    anchor_blobs: dict[str, bytes] = {}
    for index, node_id in enumerate(("anchor-a", "anchor-b")):
        vector = np.zeros(4, dtype=np.float32)
        vector[index] = 1.0
        anchor_blobs[node_id] = vector.tobytes()
        _insert_node(conn, node_id, node_id, permanent=1)
        _insert_embedding(conn, node_id, vector, index=True)
    for node_id, content in (
        ("valid-orphan", "valid orphan vector"),
        ("empty-orphan", "empty orphan vector"),
        ("invalid-orphan", "invalid orphan vector"),
    ):
        _insert_node(conn, node_id, content, permanent=1)
    conn.commit()
    assert dict(conn.execute("SELECT node_id, vector FROM embeddings")) == anchor_blobs
    assert (
        dict(conn.execute("SELECT node_id, embedding FROM vec_embeddings"))
        == anchor_blobs
    )
    conn.close()

    orphan_vectors = {
        "valid orphan vector": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "empty orphan vector": np.array([], dtype=np.float32),
        "invalid orphan vector": np.array([np.nan, 0.0, 0.0, 1.0], dtype=np.float32),
    }

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def encode(self, texts: list[str]) -> np.ndarray:
            assert len(texts) == 1
            content = texts[0]
            assert content in orphan_vectors, f"unexpected model input: {content!r}"
            self.calls.append(content)
            return orphan_vectors[content].reshape(1, -1).copy()

    client = FakeClient()
    _patch_vec_connections(monkeypatch)

    result = _run(db_path, limit=2, max_edges=10, embedding_client=client)
    assert client.calls == list(orphan_vectors)

    conn = _REAL_CONNECT(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    ordinary = dict(
        conn.execute("SELECT node_id, vector FROM embeddings ORDER BY node_id")
    )
    indexed = dict(
        conn.execute("SELECT node_id, embedding FROM vec_embeddings ORDER BY node_id")
    )
    conn.close()

    allowed_ids = {*anchor_blobs, "valid-orphan", "empty-orphan", "invalid-orphan"}
    assert set(ordinary) <= allowed_ids
    assert set(indexed) <= allowed_ids
    for node_id, blob in anchor_blobs.items():
        assert ordinary.get(node_id) == blob
        assert indexed.get(node_id) == blob

    expected_orphan_blobs = {
        "valid-orphan": orphan_vectors["valid orphan vector"].tobytes(),
        "empty-orphan": orphan_vectors["empty orphan vector"].tobytes(),
        "invalid-orphan": orphan_vectors["invalid orphan vector"].tobytes(),
    }
    for node_id, expected_blob in expected_orphan_blobs.items():
        if node_id in ordinary:
            assert ordinary[node_id] == expected_blob
        if node_id in indexed:
            assert indexed[node_id] == expected_blob

    current_list_binding_signature = (
        result["orphans_embedded"] == 0
        and set(ordinary) == {*anchor_blobs, "valid-orphan"}
        and set(indexed) == set(anchor_blobs)
    )
    if current_list_binding_signature:
        raise KnownConsolidationDebt(
            "local list binding left ordinary orphan rows without vec0 rows"
        )

    invalid_vector_signature = (
        ordinary.get("valid-orphan") == expected_orphan_blobs["valid-orphan"]
        and indexed.get("valid-orphan") == expected_orphan_blobs["valid-orphan"]
        and "empty-orphan" not in ordinary
        and "empty-orphan" not in indexed
        and "invalid-orphan" in ordinary
    )
    if invalid_vector_signature:
        raise KnownConsolidationDebt(
            "orphan embedding accepted a NaN vector instead of preserving both stores"
        )

    assert result["orphans_embedded"] == 1
    expected_final = {
        **anchor_blobs,
        "valid-orphan": expected_orphan_blobs["valid-orphan"],
    }
    assert ordinary == expected_final
    assert indexed == expected_final
