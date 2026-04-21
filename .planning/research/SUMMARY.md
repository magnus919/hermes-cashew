# Project Research Summary

**Project:** hermes-cashew v0.2.0 "Retrieval Core"
**Domain:** Hermes Agent memory provider plugin (local SQLite knowledge graph with embeddings)
**Researched:** 2026-04-21
**Confidence:** HIGH

## Executive Summary

hermes-cashew v0.2.0 upgrades a lightweight Hermes memory plugin from keyword-based retrieval to semantic search via sqlite-vec, backed by recursive BFS graph traversal. Research confirms that the upstream `cashew-brain` library (>=1.0.0,<2.0.0) already implements all three target capabilities—sqlite-vec retrieval, schema migration, and expanded config—making this milestone primarily an integration exercise rather than a reinvention. The plugin must remain importable when `cashew-brain` is absent, silently degrade on any failure, and never trigger embedding model downloads in CI.

The recommended approach is: (1) harden additive schema migration so v0.1.0 DBs upgrade safely, (2) wire `retrieve_recursive_bfs()` into `prefetch()` and `cashew_query` with a keyword-based fallback, and (3) expand the flat JSON config from 4 to ~30 keys with sane defaults and env var overrides. Key risks include macOS system Python blocking sqlite-vec extension loading, immutable `vec0` virtual tables making vector schema changes destructive, and BFS graph traversal hitting cycles or recursion limits. All are mitigatable with defensive coding, iterative BFS with caps, and strict additive migration discipline.

## Key Findings

### Recommended Stack

The stack remains lightweight: Python >=3.10, `cashew-brain>=1.0.0,<2.0.0`, and `sqlite-vec>=0.1.9,<0.2.0`. No migration framework (Alembic/SQLAlchemy), no numpy, no pydantic, and no YAML parser at runtime. sqlite-vec is a ~160 KB C-extension wheel providing O(log N) ANN search inside SQLite; it does not download models, so `HF_HUB_OFFLINE=1` in CI remains safe. Migrations use raw SQL (`PRAGMA table_info` → `ALTER TABLE ADD COLUMN`) matching upstream's `_ensure_schema()` pattern. Config stays flat JSON to honor the Hermes `memory setup` contract; upstream's nested YAML keys are flattened with underscore separators (e.g., `performance_token_budget` → `token_budget`).

**Core technologies:**
- **sqlite-vec >=0.1.9,<0.2.0**: SQLite vector-search extension — upstream already depends on it; small wheels, no model downloads.
- **sqlite3 (stdlib)**: SQLite driver — `enable_load_extension(True)` required before `sqlite_vec.load(conn)`.
- **dataclasses + json (stdlib)**: Typed config object and serialization — keeps plugin lightweight; no pydantic needed.
- **cashew-brain >=1.0.0,<2.0.0**: Backing store — contains `core/retrieval.py`, `core/session.py`, and all BFS/embedding logic.

### Expected Features

**Must have (table stakes):**
- sqlite-vec O(log N) embedding search with graceful fallback to brute-force BLOB scan — users expect semantic retrieval; macOS/restricted envs must not crash.
- Recursive BFS graph walk from seed nodes — Cashew's value is graph traversal, not just vector search.
- Automatic additive schema migration — v0.1.0 DBs upgrade transparently; never DROP columns.
- Expanded config (~30 keys, flat JSON) with sane defaults and env var overrides — zero mandatory editing; backward-compatible with existing 4-key `cashew.json`.
- Recency weighting via `referent_time` — biographical clock for imported historical data.

**Should have (competitive):**
- Hybrid scoring (embedding + graph proximity + recency) — better context injection than pure vector search.
- Access metadata updates on retrieval (`access_count++`, `last_accessed`) — signals importance for GC and ranking.
- `permanent=1` immunity in retrieval — important beliefs must survive GC.

**Defer (v2+):**
- Streaming BFS — dashboard animation feature, no LLM consumer needs it.
- Bulk embedding backfill — upstream CLI (`cashew embed`) handles this; plugin should not orchestrate.
- Full node_type taxonomy customization — upstream supports it, but hardcoded defaults suffice for now.

### Architecture Approach

The plugin's job is integration, not reimplementation. `cashew-brain` already provides `retrieve_recursive_bfs()`, `format_context()`, additive schema migration, and config defaults. The plugin must: (a) ensure the DB schema is complete enough for upstream retrieval functions, (b) expose enough config keys for upstream retrieval params, and (c) wire BFS retrieval into `prefetch()` and `handle_tool_call(cashew_query)` with a fallback to the existing `ContextRetriever.retrieve()`. All `cashew-brain` imports must remain lazy (inside methods) to keep the module loadable when the dependency is absent. A new `_try_bfs_retrieve()` method wraps the lazy import, attempts BFS retrieval, and returns `None` on any failure so the caller falls back to the v0.1.0 keyword path.

**Major components:**
1. **`CashewMemoryProvider`** — Hermes ABC implementation; modified to add `_try_bfs_retrieve()`, config-driven retriever init, and expanded schema migration.
2. **`config.py`** — Pure config helpers; expanded from 4 to ~30 flat JSON keys with env overrides and backward compatibility.
3. **`_ensure_db_schema()`** — Schema creation + additive migration; adds `_migrate_columns()` and `_ensure_vec_embeddings_table()`.
4. **`_try_bfs_retrieve()`** — New semantic search wrapper with fallback; lazy-imports `core.retrieval` and silently degrades.

### Critical Pitfalls

1. **sqlite-vec fails on macOS system Python** — `enable_load_extension` is disabled in Apple's bundled SQLite. Wrap `sqlite_vec.load()` in `try/except (AttributeError, RuntimeError)`, set a `_sqlite_vec_available` flag, and fall back to keyword retrieval. Document Homebrew Python workaround.
2. **vec0 virtual tables are immutable** — `ALTER TABLE ADD COLUMN` does not apply to virtual tables. Any schema change requires `DROP TABLE` + recreate, destroying all embeddings. Design the `vec0` table once (`node_id TEXT PRIMARY KEY, embedding FLOAT[384] DISTANCE_METRIC=cosine`), store mutable metadata in regular SQLite tables, and never auto-rebuild.
3. **v0.1.0→v0.2.0 migration must be strictly additive** — Data loss is unacceptable. Use `ALTER TABLE ... ADD COLUMN` only, with `DEFAULT` values where needed. Never rebuild `thought_nodes` or `derivation_edges` automatically. Wrap in a transaction; on failure, rollback and silent-degrade.
4. **Config expansion breaks existing tests** — Tests assert `len(schema) == 4` and exact key sets. Update assertions to check presence of original 4 keys and sensible defaults for new keys, not exact counts. Preserve v0.1.0 key names and defaults.
5. **BFS traversal is a recursion/cycle bomb** — Naive recursive BFS hits `RecursionError` on cyclic graphs or dense clusters. Use iterative BFS (`deque`), cap `max_depth` (≤3) and `max_total_nodes` (≤50), track visited `node_id`s, and issue edges as a single parameterized query per wave. Add a 2s timeout guard.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Schema Migration Foundation
**Rationale:** Retrieval depends on the correct schema (especially `vec_embeddings` virtual table and new `thought_nodes` columns). Config expansion is independent but retrieval reads config params. Schema must come first.
**Delivers:** `_ensure_db_schema()` can upgrade a v0.1.0 DB to the full schema expected by `cashew-brain` v1.0.0. Includes additive column migration, `vec_embeddings` virtual table creation (guarded), and preservation of existing `_migrate_edges_timestamp_not_null` logic.
**Addresses:** Issue #17 (complete schema + automatic migration)
**Avoids:** Pitfall 2 (vec0 immutability), Pitfall 3 (data loss in migration), Pitfall 6 (migration vs worker lock)

### Phase 2: sqlite-vec + Recursive BFS Retrieval
**Rationale:** Depends on Phase 1 schema (retrieval needs `vec_embeddings` and all `thought_nodes` columns). This is the highest-complexity, highest-value feature.
**Delivers:** `prefetch()` and `handle_tool_call(cashew_query)` use `retrieve_recursive_bfs()` when available, falling back to keyword retrieval. Includes lazy import of `core.retrieval`, handling of `format_context()` shape differences, and graceful fallback on all failures.
**Addresses:** Issue #7 (sqlite-vec + recursive BFS retrieval)
**Avoids:** Pitfall 1 (macOS load failure), Pitfall 7 (BFS recursion/cycles), Pitfall 8 (`is_available` contract violation), Pitfall 9 (dimension mismatch), Pitfall 12 (`fake_embedder` gaps)

### Phase 3: Expanded Config Alignment
**Rationale:** Self-contained; can be done in parallel with Phase 2 if retrieval uses hardcoded defaults as fallback. Doing it after Phase 2 allows retrieval to immediately consume new config keys.
**Delivers:** ~30-key flat JSON config with sane defaults, env var overrides (`CASHEW_*`), backward compatibility with v0.1.0 4-key files, and updated test assertions.
**Addresses:** Issue #16 (expanded config alignment)
**Avoids:** Pitfall 4 (test breakage), Pitfall 5 (env var persistence), Pitfall 10 (`recall_k` overload), Pitfall 11 (test assertions), Pitfall 14 (dataclass bloat)

### Phase 4: Integration Tests & Verification
**Rationale:** End-to-end validation that all three features work together and that v0.1.0→v0.2.0 upgrades are safe.
**Delivers:** E2E tests for migration + retrieval, config round-trip, fallback paths, and CI verification (no embedding downloads, no `database is locked` errors).
**Addresses:** Cross-cutting verification of Issues #7, #16, #17
**Avoids:** All critical pitfalls surface here; catch them before release.

### Phase Ordering Rationale

- **Schema first:** `vec_embeddings` must exist before retrieval can use it; new columns must exist before BFS queries reference them.
- **Retrieval second:** Highest risk and highest value; depends on schema but not on config expansion (can use hardcoded defaults).
- **Config third:** Straightforward expansion; benefits from being tested against the working retrieval path.
- **Integration last:** Validates the full v0.1.0→v0.2.0 upgrade story and catches interaction bugs.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2 (Retrieval):** Ambiguity around `retrieve()` vs `retrieve_recursive_bfs()` — Issue #7 names the latter but demands recency weighting, which only the former provides. Decision needed before implementation.
- **Phase 2 (Retrieval):** `ContextRetriever` vs standalone functions — type mismatch between `RelevantNode` and `RetrievalResult` requires design decision on how `prefetch()` formats output.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Schema Migration):** Raw SQL `PRAGMA` + `ALTER TABLE ADD COLUMN` is well-documented and already used upstream.
- **Phase 3 (Config Alignment):** Flat JSON + dataclasses + env overrides is a standard Python pattern; upstream config provides the key list and defaults.
- **Phase 4 (Integration Tests):** pytest + mocking + temp_path fixtures are standard; no novel research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | sqlite-vec 0.1.9 is stable with wheels for all target platforms. Upstream cashew-brain 1.0.0 source was inspected directly. No risky dependencies added. |
| Features | HIGH | Upstream source code inspected for `retrieve_recursive_bfs()`, `retrieve()`, `_recency_weight()`, config schema, and migration patterns. Feature set is a subset of upstream capabilities. |
| Architecture | HIGH | Dual-layout plugin constraints (lazy imports, silent degrade, non-daemon worker) are well-understood from v0.1.0. Integration pattern is thin wrapper + fallback. |
| Pitfalls | HIGH | All critical pitfalls were identified from direct code inspection, sqlite-vec docs, and SQLite documentation. Prevention strategies are concrete and testable. |

**Overall confidence:** HIGH

### Gaps to Address

1. **`retrieve()` vs `retrieve_recursive_bfs()` ambiguity:** Issue #7 names `retrieve_recursive_bfs()` but also demands recency weighting, which that function lacks. Decision needed during Phase 2 planning: enhance `retrieve_recursive_bfs()` post-call with recency weighting, or switch to `retrieve()` which already has hybrid scoring + recency but a simpler walk.
2. **Type mismatch between `RelevantNode` and `RetrievalResult`:** `ContextRetriever.retrieve()` returns `List[RelevantNode]`; `core.retrieval.retrieve()` returns `List[RetrievalResult]`. `format_context()` signatures differ. Plugin formatting code may need adjustment; exact delta needs inspection during Phase 2.
3. **Embedding model dimension pinning:** v0.2.0 pins `vec0` to 384 dimensions (`all-MiniLM-L6-v2`). If users change models later, the table must be rebuilt. A decision on whether to store model metadata and auto-detect dimension changes should be made in Phase 2.
4. **macOS CI coverage:** The graceful fallback for `enable_load_extension` failure needs a dedicated test, ideally on a macOS runner with stock Python. If CI lacks macOS runners, this must be manually validated.

## Sources

### Primary (HIGH confidence)
- `cashew-brain` 1.0.0 installed source code — `core/retrieval.py`, `core/embeddings.py`, `core/context.py`, `core/config.py`, `core/db.py`, `core/session.py` inspected directly.
- sqlite-vec PyPI (version 0.1.9) and Python docs (`sqlite_vec.load()`, `serialize_float32()`, `vec_distance_cosine`) — https://alexgarcia.xyz/sqlite-vec/python.html
- sqlite-vec API reference (`vec0` virtual tables) — https://alexgarcia.xyz/sqlite-vec/api-reference.html
- hermes-cashew v0.1.0 source — `plugins/memory/cashew/__init__.py`, `config.py`, `tools.py`, `tests/`, `AGENTS.md`

### Secondary (MEDIUM confidence)
- SQLite ALTER TABLE documentation — https://www.sqlite.org/lang_altertable.html
- sqlite-vec GitHub — virtual table behavior, no ALTER support
- Alembic batch migrations for SQLite — copy-rebuild pattern reference

### Tertiary (LOW confidence)
- Software Engineering Stack Exchange — additive-only migration philosophy
- Code Without Rules — Python queue/threading gotchas
- SQLite locking documentation — https://sqlite.org/lockingv3.html

---
*Research completed: 2026-04-21*
*Ready for roadmap: yes*
