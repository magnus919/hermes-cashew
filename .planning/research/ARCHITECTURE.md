# Architecture Patterns

**Domain:** Hermes Agent memory provider plugin (hermes-cashew) — v0.2.0 Retrieval Core milestone
**Researched:** 2026-04-21
**Confidence:** HIGH (source code inspected directly from installed cashew-brain 1.0.0)

## Executive Summary

The v0.2.0 milestone adds three capabilities to the existing dual-layout plugin: **sqlite-vec semantic retrieval**, **additive schema migration**, and **expanded config (~30 keys)**. All three must integrate with the existing constraints: lazy imports (module loadable without cashew-brain), silent degrade on failure, no network/FS in `is_available()`, and the bounded non-daemon sync queue.

The key architectural insight is that **cashew-brain already implements all three capabilities internally** — the plugin's job is not to reimplement sqlite-vec or BFS traversal, but to:
1. Ensure the DB schema is complete enough for cashew-brain's retrieval functions
2. Expose enough config keys for cashew-brain's retrieval params
3. Wire `retrieve_recursive_bfs()` into `prefetch()` and `cashew_query` tool calls with graceful fallback to the existing keyword-based `ContextRetriever.retrieve()`

## Recommended Architecture

### Component Diagram (v0.2.0)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hermes Agent (loads plugin at runtime)               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              plugins/memory/cashew/__init__.py                    │   │
│  │           CashewMemoryProvider (existing, MODIFIED)               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │  prefetch() │  │handle_tool_ │  │    sync_turn()          │  │   │
│  │  │   (MOD)     │  │   call()    │  │   (existing)            │  │   │
│  │  │  ┌───────┐  │  │   (MOD)     │  │                         │  │   │
│  │  │  │ NEW:  │  │  │             │  │                         │  │   │
│  │  │  │_try_  │  │  │  ┌───────┐  │  │                         │  │   │
│  │  │  │bfs_   │  │  │  │ NEW:  │  │  │                         │  │   │
│  │  │  │retrieve│  │  │  │_try_  │  │  │                         │  │   │
│  │  │  │()     │  │  │  │bfs_   │  │  │                         │  │   │
│  │  │  └───┬───┘  │  │  │retrieve│  │  │                         │  │   │
│  │  │      │      │  │  │()     │  │  │                         │  │   │
│  │  │  ┌───▼───┐  │  │  └───┬───┘  │  │                         │  │   │
│  │  │  │FALLBACK│  │  │  ┌───▼───┐  │  │                         │  │   │
│  │  │  │keyword │  │  │  │FALLBACK│  │  │                         │  │   │
│  │  │  │retrieve│  │  │  │keyword │  │  │                         │  │   │
│  │  │  └───────┘  │  │  │retrieve│  │  │                         │  │   │
│  │  └─────────────┘  │  └───────┘  │  └─────────────────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │initialize() │  │_ensure_db_  │  │   _start_sync_worker()  │  │   │
│  │  │   (MOD)     │  │  schema()   │  │      (existing)         │  │   │
│  │  │             │  │   (MOD)     │  │                         │  │   │
│  │  │  NEW:       │  │  ┌───────┐  │  │                         │  │   │
│  │  │  config-    │  │  │ NEW:  │  │  │                         │  │   │
│  │  │  driven     │  │  │_migrate│  │  │                         │  │   │
│  │  │  _retriever │  │  │_columns│  │  │                         │  │   │
│  │  │  init       │  │  │()     │  │  │                         │  │   │
│  │  │             │  │  └───────┘  │  │                         │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│           ┌──────────────────┼──────────────────┐                      │
│           │                  │                  │                       │
│  ┌────────▼────────┐ ┌───────▼───────┐ ┌───────▼────────┐             │
│  │ plugins/memory/ │ │ plugins/memory/│ │  core.retrieval │             │
│  │ cashew/config.py│ │ cashew/tools.py│ │  (LAZY IMPORT)  │             │
│  │   (MODIFIED)    │ │   (existing)   │ │                 │             │
│  │  ~30-key schema │ │  envelope fns  │ │retrieve_recursive│            │
│  │  env overrides  │ │                │ │    _bfs()       │             │
│  │  backward compat│ │                │ │                 │             │
│  └─────────────────┘ └────────────────┘ └─────────────────┘             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              root __init__.py (flat-entry shim)                 │   │
│  │              NO CHANGES — still re-exports register             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | v0.1.0 State | v0.2.0 Change | Communicates With |
|-----------|---------------|--------------|---------------|-------------------|
| `CashewMemoryProvider` | Hermes ABC implementation, lifecycle | Exists | MODIFIED — new `_try_bfs_retrieve()`, config-driven `_retriever` init | Config, Tools, cashew-brain (lazy) |
| `config.py` | Pure config helpers, zero coupling | 4-key flat JSON | MODIFIED — ~30 keys, nested sections, env overrides, backward compat | Provider only |
| `tools.py` | JSON envelope builders | Exists | UNCHANGED — same envelope shapes | Provider only |
| `_ensure_db_schema()` | Schema creation + migration | CREATE TABLE only | MODIFIED — additive column migration, vec table creation | SQLite directly |
| `_try_bfs_retrieve()` | Semantic search with fallback | Does not exist | NEW — wraps `retrieve_recursive_bfs()`, falls back to `ContextRetriever.retrieve()` | `core.retrieval` (lazy), `core.context` (lazy) |
| `retrieve_recursive_bfs()` | sqlite-vec + BFS traversal | Does not exist in plugin | NEW — imported lazily from cashew-brain | SQLite + sqlite-vec |

## Data Flow Changes

### Recall Path (v0.1.0 → v0.2.0)

**v0.1.0:**
```
prefetch(query) ──► ContextRetriever.retrieve(query, max_nodes=k)
                      └──► naive keyword extraction (Python regex)
                           └──► format_context(nodes)
```

**v0.2.0:**
```
prefetch(query) ──► _try_bfs_retrieve(query, top_k=k)
                      ├──► [attempt] core.retrieval.retrieve_recursive_bfs(db_path, query, top_k=k)
                      │       ├──► sqlite-vec embedding search (O(log N))
                      │       ├──► BFS graph walk from seed nodes
                      │       └──► format_context(results) [new shape]
                      └──► [fallback] ContextRetriever.retrieve(query, max_nodes=k)
                               └──► naive keyword extraction
```

### Schema Initialization Flow (v0.2.0)

```
initialize() ──► _ensure_db_schema(db_path)
                    ├──► CREATE TABLE IF NOT EXISTS thought_nodes (full schema)
                    ├──► CREATE TABLE IF NOT EXISTS derivation_edges (full schema)
                    ├──► CREATE TABLE IF NOT EXISTS embeddings (full schema)
                    ├──► _migrate_columns(conn) — additive ALTER TABLE for missing columns
                    │       ├──► reasoning, mood_state, metadata, last_updated, etc.
                    │       └──► PRAGMA-driven: only adds what doesn't exist
                    ├──► _migrate_edges_timestamp_not_null(conn) — existing complex migration
                    └──► _ensure_vec_embeddings_table(conn) — NEW
                            ├──► try: import sqlite_vec
                            ├──► CREATE VIRTUAL TABLE vec_embeddings USING vec0(...)
                            └──► silent skip on ImportError / failure
```

### Config Loading Flow (v0.2.0)

```
load_config(hermes_home)
    ├──► read $HERMES_HOME/cashew.json (if exists)
    ├──► deep-merge file over DEFAULTS
    │       ├──► v0.1.0 files: 4 keys merge cleanly into expanded defaults
    │       └──► unknown keys: preserved (not dropped) for forward compat
    ├──► apply env var overrides
    │       ├──► CASHEW_DB_PATH, CASHEW_EMBEDDING_MODEL (existing)
    │       └──► CASHEW_TOKEN_BUDGET, CASHEW_TOP_K, CASHEW_WALK_DEPTH (new)
    ├──► flatten nested sections to CashewConfig dataclass fields
    └──► return CashewConfig(...)
```

## Patterns to Follow

### Pattern 1: Lazy Import Gate
**What:** Every cashew-brain import happens inside a method, never at module level. The defensive `try/except ImportError` at module top sets `None` as fallback.
**When:** Any code path that touches cashew-brain runtime.
**Example:**
```python
# Module level (existing pattern)
try:
    from core.context import ContextRetriever
except ImportError:
    ContextRetriever = None

# Inside method (new BFS path)
def _try_bfs_retrieve(self, query: str, top_k: int) -> str:
    try:
        from core.retrieval import retrieve_recursive_bfs, format_context
    except Exception:
        logger.debug("retrieve_recursive_bfs unavailable")
        return None  # signals fallback
    try:
        results = retrieve_recursive_bfs(
            db_path=str(self._db_path),
            query=query,
            top_k=top_k,
            n_seeds=self._config.n_seeds,
            picks_per_hop=self._config.picks_per_hop,
            max_depth=self._config.walk_depth,
        )
        return format_context(results)
    except Exception:
        logger.warning("BFS retrieval failed", exc_info=True)
        return None  # signals fallback
```

### Pattern 2: Silent Degrade with Fallback Stack
**What:** New retrieval tries the best path, returns `None` on any failure, caller falls back to the v0.1.0 path. No exception propagates to Hermes.
**When:** Any feature that depends on optional/new Cashew capabilities.
**Example:**
```python
def prefetch(self, query: str) -> str:
    if self._retriever is None or self._config is None:
        return ""
    # v0.2.0: try BFS first
    if self._config.use_bfs_retrieval:  # or auto-detect sqlite-vec availability
        bfs_result = self._try_bfs_retrieve(query, top_k=self._config.recall_k)
        if bfs_result is not None:
            return bfs_result
    # v0.1.0 fallback (preserved verbatim)
    try:
        nodes = self._retriever.retrieve(query, max_nodes=self._config.recall_k)
        return self._retriever.format_context(nodes)
    except Exception:
        logger.warning("cashew recall failed", exc_info=True)
        return ""
```

### Pattern 3: Additive Schema Migration
**What:** Inspect existing schema via `PRAGMA table_info`, add missing columns with `ALTER TABLE ADD COLUMN`. Never drop tables or recreate with data loss.
**When:** `initialize()` on every provider startup.
**Example:**
```python
def _migrate_columns(self, conn: sqlite3.Connection) -> None:
    """Add missing columns without dropping data."""
    cursor = conn.execute("PRAGMA table_info(thought_nodes)")
    existing = {row[1] for row in cursor.fetchall()}
    
    additions = [
        ("reasoning", "TEXT"),
        ("mood_state", "TEXT"),
        ("metadata", "TEXT DEFAULT '{}'"),
        ("last_updated", "TEXT"),
        ("permanent", "INTEGER DEFAULT 0"),
        ("tags", "TEXT"),
        ("referent_time", "TEXT"),
    ]
    for col, dtype in additions:
        if col not in existing:
            conn.execute(f"ALTER TABLE thought_nodes ADD COLUMN {col} {dtype}")
            logger.info("Migrated thought_nodes: added %s", col)
```

### Pattern 4: Config Backward Compatibility
**What:** v0.1.0 flat JSON (4 keys) must load into expanded config without errors. Missing keys get defaults. Unknown keys are preserved in the raw dict (not dropped) so future versions can read them.
**When:** `load_config()` and `save_config()`.
**Example:**
```python
# DEFAULTS expanded from 4 keys to ~30, but same 4 keys keep same values
DEFAULTS: dict[str, Any] = {
    # v0.1.0 keys (unchanged values for compat)
    "cashew_db_path": "cashew/brain.db",
    "embedding_model": "all-MiniLM-L6-v2",
    "recall_k": 5,
    "sync_queue_timeout": 30.0,
    # v0.2.0 new keys (sane defaults)
    "use_bfs_retrieval": True,
    "n_seeds": 5,
    "picks_per_hop": 3,
    "walk_depth": 2,
    "token_budget": 2000,
    # ... etc
}

# Loading: merge file over defaults, filter to known dataclass fields
merged = {**DEFAULTS, **raw}
known = {f.name for f in dataclasses.fields(CashewConfig)}
# v0.2.0: preserve unknown keys in a `_extras` dict for forward compat
extras = {k: v for k, v in merged.items() if k not in known}
filtered = {k: v for k, v in merged.items() if k in known}
config = CashewConfig(**filtered, _extras=extras)
```

### Pattern 5: Optional Dependency Probe
**What:** sqlite-vec is an optional dependency. Probe at runtime (during schema init), cache the result. Retrieval uses it if available, falls back to brute-force automatically (cashew-brain handles this internally).
**When:** `initialize()` and `_ensure_db_schema()`.
**Example:**
```python
def _ensure_vec_embeddings_table(self, conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        logger.debug("sqlite-vec not available; skipping vec_embeddings table")
        return
    
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
        USING vec0(node_id text primary key, embedding float[384] distance_metric=cosine)
    """)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Importing core.retrieval at Module Level
**What:** Adding `from core.retrieval import retrieve_recursive_bfs` at the top of `__init__.py`.
**Why bad:** Breaks module load when cashew-brain is not installed (wheel-smoke, discovery). Violates Phase 1 constraint.
**Instead:** Lazy import inside `_try_bfs_retrieve()` with `try/except Exception`.

### Anti-Pattern 2: Recreating Tables with Data Loss
**What:** Dropping `thought_nodes` and recreating with full schema.
**Why bad:** Destroys user's thought graph on plugin upgrade.
**Instead:** Additive `ALTER TABLE ADD COLUMN` only. The only exception is `derivation_edges.timestamp` which already has a data-preserving migration (`_migrate_edges_timestamp_not_null`).

### Anti-Pattern 3: Synchronous Embedding in Hot Path
**What:** Calling `embed_text()` inside `prefetch()` or `sync_turn()`.
**Why bad:** `embed_text()` triggers model load (~500MB) on first call. `sync_turn()` contract is <10ms. `prefetch()` is called synchronously before every LLM response.
**Instead:** Cashew-brain's `retrieve_recursive_bfs()` embeds the query internally, but it does so via the embedding service which may load the model. The plugin must ensure `HF_HUB_OFFLINE=1` in CI and mock embeddings in tests. In production, first query will be slow — document this. Do NOT add embedding calls to `sync_turn()`.

### Anti-Pattern 4: Nested Config Without Flat Fallback
**What:** Switching `cashew.json` from flat JSON to nested YAML structure.
**Why bad:** Breaks existing v0.1.0 installs. Hermes's `save_config()` passes flat dicts.
**Instead:** Keep JSON flat. Nest logically in the dataclass (e.g., `gc_mode` flat key) or use prefixed keys (`gc_threshold`, `sleep_enabled`). Deep-merge from flat file.

### Anti-Pattern 5: Reimplementing sqlite-vec Logic in Plugin
**What:** Writing custom `vec0` virtual table creation or embedding search in the plugin.
**Why bad:** Cashew-brain already has `_ensure_embeddings_table()`, `search()`, `backfill_vec_index()`. Reimplementing creates drift and maintenance burden.
**Instead:** Delegate to `core.embeddings.search()` and `core.retrieval.retrieve_recursive_bfs()`. The plugin only ensures the `vec_embeddings` virtual table exists (as a migration step) so cashew-brain can use it.

## Scalability Considerations

| Concern | At 100 nodes | At 10K nodes | At 100K nodes |
|---------|--------------|--------------|---------------|
| **Retrieval** | Keyword scan is fine (~10ms) | sqlite-vec O(log N) required | sqlite-vec essential; BFS depth should stay ≤3 |
| **Schema migration** | ALTER TABLE ADD COLUMN is instant | Same — metadata operation | Same — SQLite schema changes are O(1) |
| **Config size** | ~30 keys, negligible | Same | Same |
| **Sync queue** | Bounded queue (16) handles burst | Same — worker drains asynchronously | Same; dropped turns logged |
| **Embedding storage** | ~1.5MB (384-dim float32) | ~150MB | ~1.5GB; consider embedding model dimensionality |

## Integration Points (New Code Touches Existing)

| Integration Point | Existing Code | New Code | Risk Level |
|-------------------|---------------|----------|------------|
| `prefetch()` | Calls `ContextRetriever.retrieve()` | Adds `_try_bfs_retrieve()` then falls back | **MEDIUM** — hot path, must not regress latency |
| `handle_tool_call(cashew_query)` | Same as prefetch | Same BFS path | **MEDIUM** — LLM-visible, must not break envelope |
| `initialize()` | Creates `_retriever = ContextRetriever(...)` | No change to retriever init; schema migration expanded | **LOW** — additive only |
| `_ensure_db_schema()` | CREATE TABLE + `_migrate_edges_timestamp_not_null()` | Add `_migrate_columns()` + `_ensure_vec_embeddings_table()` | **LOW** — additive, idempotent |
| `config.py` | 4-key flat JSON | ~30-key flat JSON with env overrides | **MEDIUM** — must preserve v0.1.0 compat |
| `CashewConfig` dataclass | 4 fields | ~30 fields | **LOW** — Python dataclass expansion is safe |
| `system_prompt_block()` | Shows graph node/edge count | May show retrieval mode (keyword vs BFS) | **LOW** — cosmetic |

## Suggested Build Order

The build order respects dependencies: schema must be complete before retrieval can use it; config must be loadable before retrieval can read its params.

### Phase A: Schema Migration Foundation
**Goal:** `_ensure_db_schema()` can upgrade a v0.1.0 DB to the full schema expected by cashew-brain v1.0.0.

**Tasks:**
1. Add `_migrate_columns()` — additive ALTER TABLE for all missing `thought_nodes` columns (`reasoning`, `mood_state`, `metadata`, `last_updated`, `permanent`, `tags`, `referent_time`)
2. Add `_ensure_vec_embeddings_table()` — create `vec_embeddings` virtual table if `sqlite-vec` is importable
3. Update `verify.py` schema to match (it currently creates tables without `referent_time`)
4. Tests: verify migration on v0.1.0-schema DB, verify idempotency (running twice is safe)

**Depends on:** Nothing — pure SQLite operations.
**Blocks:** Phase B (retrieval needs correct schema), Phase C (config doesn't strictly block but retrieval reads config).

### Phase B: sqlite-vec + Recursive BFS Retrieval
**Goal:** `prefetch()` and `cashew_query` use `retrieve_recursive_bfs()` when available, falling back to keyword retrieval.

**Tasks:**
1. Add `_try_bfs_retrieve()` method with lazy import of `core.retrieval`
2. Modify `prefetch()` to try BFS first, then fall back to `ContextRetriever.retrieve()`
3. Modify `handle_tool_call(cashew_query)` to use same path
4. Handle `format_context()` shape difference: `core.retrieval.format_context()` returns different output than `ContextRetriever.format_context()` — ensure both are acceptable to Hermes
5. Tests: mock `retrieve_recursive_bfs` to avoid embedding download; test fallback path; test envelope shapes

**Depends on:** Phase A (schema must have all columns for `retrieve_recursive_bfs` to work).
**Blocks:** Nothing — standalone feature.

### Phase C: Expanded Config with Env Overrides
**Goal:** Config supports ~30 keys, loads v0.1.0 JSON seamlessly, supports env var overrides.

**Tasks:**
1. Expand `DEFAULTS` to ~30 keys (flattened from upstream YAML sections)
2. Expand `CashewConfig` dataclass fields
3. Add env var override logic in `load_config()` (pattern: `CASHEW_<UPPER_SNAKE_KEY>`)
4. Update `get_config_schema()` to return new field descriptors
5. Update `save_config()` to preserve unknown keys (forward compat)
6. Tests: round-trip v0.1.0 JSON; verify env overrides; verify unknown-key preservation

**Depends on:** Nothing — config is self-contained.
**Blocks:** Phase B retrieval reads config params (`n_seeds`, `picks_per_hop`, `walk_depth`, `use_bfs_retrieval`). Can be done in parallel with Phase B if retrieval uses hardcoded defaults as fallback.

### Phase D: Integration Tests & Verification
**Goal:** End-to-end verification of all three features together.

**Tasks:**
1. E2E test: initialize with v0.1.0 DB → migration runs → BFS retrieval works
2. E2E test: config with all 30 keys → save → load → values preserved
3. E2E test: sqlite-vec unavailable → falls back to keyword retrieval
4. Update `verify.py` to exercise BFS path if sqlite-vec is available
5. Update CI if needed (no changes expected — `HF_HUB_OFFLINE=1` still blocks downloads)

**Depends on:** Phase A, B, C.
**Blocks:** Release.

## Detailed Design Decisions

### Decision: Keep Config Flat JSON (Not Nested YAML)
**Rationale:** Hermes's `save_config()` passes flat dicts. Existing v0.1.0 `cashew.json` files are flat. Switching to YAML or nested JSON would break backward compatibility and require Hermes-side changes.
**Mapping:** Upstream YAML sections (database, models, performance, gc, sleep, domains) are flattened with prefixes: `db_path` → `cashew_db_path`, `token_budget` → `token_budget`, `gc_mode` → `gc_mode`, etc.

### Decision: `retrieve_recursive_bfs()` Returns `RetrievalResult`, Not `RelevantNode`
**Rationale:** `core.retrieval.RetrievalResult` and `core.context.RelevantNode` are different dataclasses with different fields. The plugin must not try to unify them. Use `core.retrieval.format_context()` for BFS results, `ContextRetriever.format_context()` for keyword results.
**Impact:** `prefetch()` returns a string in both cases, so Hermes is agnostic to the internal representation.

### Decision: Auto-Enable BFS When sqlite-vec Available
**Rationale:** The user shouldn't need to manually toggle `use_bfs_retrieval`. If `sqlite-vec` is installed and `vec_embeddings` table exists, use BFS. If not, fall back silently.
**Config key:** `use_bfs_retrieval` (default `True`) acts as a kill-switch if the user wants to force keyword mode.
**Detection:** Probe `sqlite_vec` import and `vec_embeddings` table existence during `initialize()`. Cache result in `self._bfs_available`.

### Decision: Schema Migration Is Idempotent and Silent
**Rationale:** `initialize()` runs on every Hermes session start. Migration must be fast and quiet on already-migrated databases.
**Implementation:** Every migration step checks `PRAGMA table_info` before running `ALTER TABLE`. No-op if column already exists.

### Decision: No Plugin-Side Embedding Management
**Rationale:** Cashew-brain's `core.embeddings.embed_nodes()` handles embedding generation and dual-write to `embeddings` + `vec_embeddings`. The plugin stays "dumb" about embeddings — it only ensures the virtual table exists.
**Exception:** If the user upgrades from v0.1.0 (no embeddings) to v0.2.0, the first `retrieve_recursive_bfs()` call will find no embeddings and fall back. Cashew-brain's internal think cycle or the user running `cashew embed` will backfill. The plugin does NOT trigger embedding generation.

## Sources

- **HIGH confidence:** `cashew-brain` 1.0.0 installed source code inspected directly:
  - `/Users/magnus/Library/Python/3.13/lib/python/site-packages/core/embeddings.py` — sqlite-vec integration, `_ensure_embeddings_table()`, `search()`
  - `/Users/magnus/Library/Python/3.13/lib/python/site-packages/core/retrieval.py` — `retrieve_recursive_bfs()`, `format_context()`, `RetrievalResult`
  - `/Users/magnus/Library/Python/3.13/lib/python/site-packages/core/traversal.py` — `TraversalEngine`, graph walk primitives
  - `/Users/magnus/Library/Python/3.13/lib/python/site-packages/core/config.py` — upstream config schema (~30 keys, YAML, env overrides)
  - `/Users/magnus/Library/Python/3.13/lib/python/site-packages/core/db.py` — `NODE_COLUMNS`, schema constants
- **HIGH confidence:** `hermes-cashew` v0.1.0 source:
  - `plugins/memory/cashew/__init__.py` — existing provider, `_ensure_db_schema()`, `prefetch()`
  - `plugins/memory/cashew/config.py` — existing 4-key config
  - `AGENTS.md` — constraints (lazy imports, silent degrade, dual layout)
  - `.planning/PROJECT.md` — milestone scope (Issues #7, #16, #17)
- **MEDIUM confidence:** sqlite-vec documentation (alexgarcia.xyz/sqlite-vec/python.html) — Python API, `serialize_float32()`, extension loading
