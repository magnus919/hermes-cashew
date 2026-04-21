# Feature Landscape

**Domain:** Hermes Agent memory provider plugin (local SQLite thought-graph with embeddings)
**Researched:** 2026-04-21

## Scope

This research covers three feature areas for the v0.2.0 "Retrieval Core" milestone:
1. **sqlite-vec + recursive BFS retrieval** (GitHub Issue #7)
2. **Complete schema + automatic migration** (GitHub Issue #17)
3. **Expanded config alignment** (GitHub Issue #16)

---

## 1. sqlite-vec + Recursive BFS Retrieval

### 1.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| O(log N) embedding search via sqlite-vec | Current keyword scan is O(N) and degrades as graph grows. sqlite-vec provides native vector indexing in SQLite. | **High** | C extension loaded via `conn.enable_load_extension(True); sqlite_vec.load(conn)`. macOS system Python may block extensions (use Homebrew Python). Must handle `AttributeError` on `enable_load_extension`. |
| Recursive BFS graph walk from seed nodes | Cashew's value is graph traversal, not just vector search. Seeds → neighbors → best picks per hop. | **Medium** | Upstream `retrieve_recursive_bfs()` loads full adjacency list into memory (`defaultdict(set)`). Memory footprint scales with edge count, not node count. |
| Cosine similarity ranking | Standard for sentence-transformer embeddings (384-dim float32). sqlite-vec uses `distance_metric=cosine`. | **Low** | Query vector must be `.astype(np.float32).tobytes()`. sqlite-vec returns cosine *distance* (0 = identical); convert to similarity with `1.0 - distance`. |
| Graceful fallback if sqlite-vec unavailable | Plugin must not break on systems where the C extension can't load (CI, restricted envs). | **Medium** | Brute-force fallback scans `embeddings` table BLOBs. Already implemented upstream in `core.embeddings.search()`. Must be wired into plugin's retrieval path. |
| Recency weighting via `referent_time` | Biographical clock: a 2019 memory imported today should rank as old, not fresh. | **Medium** | Upstream `retrieve()` applies `_recency_weight()` with half-life ~365 days. `retrieve_recursive_bfs()` does **not** apply recency — plugin must add it if Issue #7 requires it. |

### 1.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Hybrid scoring (embedding + graph + recency) | Better context injection than pure vector search. Graph proximity captures relational relevance. | **Medium** | Upstream `retrieve()` (not `retrieve_recursive_bfs()`) implements this: `hybrid_score = (embedding * 0.5 + graph_proximity * 0.5) * recency_factor`. Ambiguity: Issue #7 names `retrieve_recursive_bfs()` but also demands recency weighting. Decision needed: use `retrieve()` hybrid, or enhance `retrieve_recursive_bfs()` with recency. |
| `referent_time` biographical clock vs `timestamp` operational clock | Separates user-facing event time from ingestion time. Critical for imported historical data. | **Low** | Operational paths (decay/GC) must use `timestamp`, not `referent_time`. Plugin retrieval is user-facing — should use `COALESCE(referent_time, timestamp)`. |

### 1.3 Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full table scan on every query | Defeats the purpose of sqlite-vec. Current keyword-based `ContextRetriever.retrieve()` does this. | Replace with sqlite-vec seeded retrieval. Keep keyword fallback only for zero-embedding edge cases. |
| Requiring sqlite-vec without fallback | Breaks CI, restricted environments, and Windows users where extension loading is tricky. | Always provide brute-force fallback. Log `WARNING` with `exc_info=True` when fallback is active. |
| Streaming BFS (`retrieve_bfs_streaming`) in plugin | Designed for dashboard animation (live hop-by-hop reveal). No LLM or Hermes consumer needs this. | Keep streaming in upstream dashboard only. Plugin uses synchronous retrieval. |
| Backfill all embeddings on plugin load | Blocking, slow, and unnecessary. Embedding happens lazily or during `end_session` sync. | Let `core.session.end_session` handle extraction + embedding. Plugin does not orchestrate bulk backfill. |

---

## 2. Complete Schema + Automatic Migration

### 2.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `thought_nodes` with all upstream columns | v0.1.0 schema is mostly complete but may lag behind upstream additions. | **Low** | Current schema already has: `id`, `content`, `node_type`, `domain`, `timestamp`, `access_count`, `last_accessed`, `confidence`, `source_file`, `decayed`, `metadata`, `last_updated`, `mood_state`, `permanent`, `tags`, `referent_time`. Ensure `reasoning` is present if upstream nodes use it (currently on `derivation_edges`). |
| `derivation_edges` with correct constraints | `_create_edge` in upstream does not provide `timestamp` on INSERT. If column is nullable without default, IntegrityError occurs. | **Medium** | Already handled in v0.1.1 via `_migrate_edges_timestamp_not_null()`. Must preserve this migration path. |
| `embeddings` BLOB table | Stores raw float32 vectors for brute-force fallback and embedding cache. | **Low** | Columns: `node_id`, `vector`, `model`, `updated_at`. Already in v0.1.0 schema. |
| `vec_embeddings` virtual table | sqlite-vec's `vec0` virtual table for O(log N) search. | **Medium** | `CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id TEXT PRIMARY KEY, embedding FLOAT[384] DISTANCE_METRIC=cosine)`. Must be created only after `sqlite_vec.load(conn)` succeeds. |
| Automatic additive migration on plugin load | Users must not run manual scripts when upgrading from v0.1.0. | **Medium** | Pattern: `PRAGMA table_info(table)` → `ALTER TABLE ADD COLUMN` for each missing column. Idempotent, safe. Never `DROP COLUMN`. |
| Never DROP columns on existing DBs | Data loss is unacceptable for a memory store. | **Low** | If a column type is wrong, add a new column and migrate data; do not drop the old one until a major version boundary. |

### 2.2 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| `permanent=1` immunity to decay | Important beliefs/decisions should not be garbage-collected. | **Low** | Upstream GC skips nodes with `permanent=1`. Plugin retrieval should also respect this. |
| `referent_time` biographical clock | Imported historical data ranks correctly by age. | **Low** | Already in schema. Plugin retrieval uses it; plugin sync/write uses `timestamp`. |
| Access metadata auto-updated on retrieval | `access_count++`, `last_accessed` updated. Signals node importance for GC and ranking. | **Low** | Should happen in `prefetch()` and `handle_tool_call(cashew_query)`. Silent-degrade if update fails. |
| Dual-write embeddings → BLOB + vec table | sqlite-vec searches virtual table; brute-force fallback reads BLOB table. Both stay in sync. | **Medium** | Upstream `embeddings.py` does this in `embed_nodes()`. Plugin relies on `core.session.end_session` to handle dual-write. |

### 2.3 Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| DROP COLUMN or table recreation | Destroys user data. SQLite `ALTER TABLE DROP COLUMN` is only available in 3.35.0+ and is risky. | Additive migration only. If a column is obsolete, ignore it in code but leave it in the schema. |
| Manual migration scripts | Breaks "install and forget" UX. Users will skip steps and file bugs. | Automatic migration inside `initialize()` or `_ensure_db_schema()`. |
| Schema version table | Adds unnecessary state. SQLite already has `PRAGMA table_info` and `sqlite_master`. | Inspect `PRAGMA table_info(table_name)` at runtime. No version bookkeeping needed. |
| Creating `vec_embeddings` when sqlite-vec unavailable | `CREATE VIRTUAL TABLE` will fail if the extension isn't loaded, poisoning `initialize()`. | Guard virtual table creation with `_vec_available` check. Fallback to BLOB-only search. |

---

## 3. Expanded Config Alignment

### 3.1 Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Expand from 4 keys to ~30+ upstream keys | Upstream Cashew expects tuning for retrieval, domains, GC, features. Plugin currently only exposes DB path, model, recall_k, timeout. | **Medium** | Flat JSON is Hermes convention. Upstream uses nested YAML. Mapping layer required. |
| Backward-compatible with existing `cashew.json` | v0.1.0 users have 4-key files. Must not break on load. | **Low** | Deep merge: file values overlay defaults. Unknown keys dropped (existing behavior). |
| Sane defaults for every key | Plugin must work out-of-the-box with zero config editing. | **Low** | Use upstream defaults from `core.config.CashewConfig._get_default_config()`. |
| Env var overrides | Power users need to tweak without editing JSON. | **Low** | Pattern: `os.environ.get('CASHEW_WALK_DEPTH', default)`. Upstream already does this for many keys. |

### 3.2 Recommended Config Keys (Plugin-Relevant Subset)

| Key | Upstream Path | Default | Purpose |
|-----|---------------|---------|---------|
| `cashew_db_path` | `database.path` | `cashew/brain.db` | Already exists (v0.1.0) |
| `embedding_model` | `models.embedding.name` | `all-MiniLM-L6-v2` | Already exists (v0.1.0) |
| `recall_k` | `performance.top_k_results` | `5` | Already exists as `recall_k` (v0.1.0) |
| `sync_queue_timeout` | — | `30.0` | Plugin-specific; already exists |
| `walk_depth` | `performance.walk_depth` | `2` | BFS hop limit |
| `similarity_threshold` | `performance.similarity_threshold` | `0.3` | Minimum relevance score |
| `access_weight` | `performance.access_weight` | `0.2` | Hybrid ranking: access frequency |
| `temporal_weight` | `performance.temporal_weight` | `0.1` | Hybrid ranking: recency |
| `token_budget` | `performance.token_budget` | `2000` | Context injection budget |
| `confidence_threshold` | `performance.confidence_threshold` | `0.7` | Minimum node confidence |
| `default_domain` | `domains.default` | `general` | Domain for unclassified nodes |
| `user_domain` | `domains.user` | `user` | User domain name |
| `ai_domain` | `domains.ai` | `ai` | AI domain name |
| `auto_classify` | `domains.auto_classify` | `true` | Auto-assign domain on extraction |
| `gc_mode` | `gc.mode` | `soft` | Garbage collection mode |
| `gc_threshold` | `gc.threshold` | `0.05` | Decay eligibility threshold |
| `gc_grace_days` | `gc.grace_days` | `7` | Days before decay eligible |
| `features_decay_pruning` | `features.decay_pruning` | `true` | Enable decay/GC |
| `features_pattern_detection` | `features.pattern_detection` | `true` | Enable pattern detection |

### 3.3 Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Flat JSON instead of nested YAML | Matches Hermes `memory setup` UI and plugin conventions. | **Low** | Mapping layer translates flat keys to nested upstream expectations. |
| Only plugin-relevant keys exposed | Reduces cognitive load. Users don't see cron schedules or OpenClaw paths. | **Low** | Filter upstream keys. Omit `think.schedule`, `sleep.schedule`, `integration.openclaw`, etc. |
| Deep merge with v0.1.0 4-key config | Existing users upgrade transparently. | **Low** | `load_config()` merges `DEFAULTS` dict with on-disk JSON. Add new keys to `DEFAULTS`. |

### 3.4 Anti-Features

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Exposing upstream scheduling keys | `think.schedule`, `sleep.schedule`, `extract.schedule` — plugin does not run cron jobs. Cashew CLI handles these. | Omit from plugin schema. If needed later, add selectively. |
| Nested config file | Hermes `memory setup` generates flat JSON. Nested structures break the UI. | Keep flat JSON. Map to nested upstream concepts internally if needed. |
| Requiring YAML parser | Adds dependency (`PyYAML`) for no benefit. Hermes ecosystem uses JSON. | Stay with `json` stdlib. |
| Exposing `integration.openclaw` paths | Tight coupling to a specific orchestrator. Plugin should be orchestrator-agnostic. | Omit entirely. |
| Removing existing keys | Breaks backward compatibility with v0.1.0 configs. | Keep all v0.1.0 keys. Add new ones. Never rename without deprecation. |

---

## Feature Dependencies

```
Schema Migration (Issue #17)
    └── Required by: sqlite-vec retrieval (Issue #7) — vec_embeddings table must exist
    └── Required by: Config alignment (Issue #16) — new keys may reference schema features

Config Alignment (Issue #16)
    └── Required by: sqlite-vec retrieval (Issue #7) — walk_depth, similarity_threshold, etc.

sqlite-vec Retrieval (Issue #7)
    └── Depends on: Schema migration — vec_embeddings virtual table
    └── Depends on: Config alignment — retrieval tuning parameters
    └── Replaces: Current keyword-based ContextRetriever.retrieve()
```

---

## MVP Recommendation

### Must Ship (v0.2.0)

1. **Automatic schema migration** (Issue #17 baseline)
   - Additive `ALTER TABLE ADD COLUMN` for any missing columns
   - Create `vec_embeddings` virtual table if sqlite-vec is available
   - Preserve existing `_migrate_edges_timestamp_not_null` logic
   - Complexity: **Medium** | Risk: **Low**

2. **sqlite-vec retrieval with BFS** (Issue #7 core)
   - Replace `ContextRetriever.retrieve()` keyword scan with `core.retrieval.retrieve()` or `retrieve_recursive_bfs()`
   - Graceful fallback to brute-force BLOB scan
   - Recency weighting via `referent_time`
   - Complexity: **High** | Risk: **Medium** (macOS extension loading, type mismatches between `RelevantNode` and `RetrievalResult`)

3. **Expanded config with sane defaults** (Issue #16)
   - Add retrieval-relevant keys to `DEFAULTS` and `CashewConfig`
   - Flat JSON, backward-compatible with v0.1.0
   - Env var overrides for key values
   - Complexity: **Medium** | Risk: **Low**

### Should Ship (v0.2.x)

4. **Access metadata updates on retrieval**
   - Increment `access_count`, update `last_accessed` in `prefetch()` and `cashew_query`
   - Complexity: **Low** | Risk: **Low**

5. **`permanent=1` respect in retrieval**
   - Ensure GC and retrieval paths never exclude permanent nodes
   - Complexity: **Low** | Risk: **Low**

### Defer (v0.3.0 or later)

6. **Streaming BFS** — Dashboard feature, not plugin concern.
7. **Full node_type taxonomy customization** — Upstream supports custom types, but plugin can use hardcoded defaults for now.
8. **Bulk embedding backfill** — Let upstream CLI (`cashew embed`) handle this.
9. **Hybrid scoring graph proximity tuning** — If using `retrieve_recursive_bfs`, graph proximity is implicit in the BFS. If using `retrieve()`, it's already implemented.

---

## Open Questions / Ambiguities

1. **`retrieve()` vs `retrieve_recursive_bfs()`**: Issue #7 names `retrieve_recursive_bfs()`, but that function ranks purely by cosine similarity without recency weighting. Upstream `retrieve()` has hybrid scoring + recency but uses a simpler BFS walk. **Decision needed**: which function does the plugin call? If `retrieve_recursive_bfs()`, recency weighting must be added post-call.

2. **`ContextRetriever` dependency**: The plugin currently instantiates `ContextRetriever` and calls `.retrieve()` + `.format_context()`. `core.retrieval` exposes standalone functions (`retrieve()`, `retrieve_recursive_bfs()`, `format_context()`). **Decision needed**: switch to standalone functions, or extend `ContextRetriever` upstream?

3. **Type mismatch**: `ContextRetriever.retrieve()` returns `List[RelevantNode]`; `core.retrieval.retrieve()` returns `List[RetrievalResult]`. `format_context()` signatures differ. Plugin formatting code may need adjustment.

4. **macOS extension loading**: `sqlite_vec.load()` requires `enable_load_extension`, which fails on macOS system Python. CI uses Homebrew Python, but end users may not. **Mitigation**: catch `AttributeError` and fall back to brute-force with a clear log message.

---

## Sources

- [Cashew upstream `core/retrieval.py`](https://github.com/rajkripal/cashew/blob/main/core/retrieval.py) — `retrieve_recursive_bfs()`, `retrieve()`, `_graph_walk()`, `_recency_weight()` — **HIGH confidence**
- [Cashew upstream `core/embeddings.py`](https://github.com/rajkripal/cashew/blob/main/core/embeddings.py) — sqlite-vec loading, dual-write, brute-force fallback — **HIGH confidence**
- [Cashew upstream `core/context.py`](https://github.com/rajkripal/cashew/blob/main/core/context.py) — Current keyword-based `ContextRetriever` — **HIGH confidence**
- [Cashew upstream `core/config.py`](https://github.com/rajkripal/cashew/blob/main/core/config.py) — ~30-key nested YAML config with defaults and env overrides — **HIGH confidence**
- [Cashew upstream `config.example.yaml`](https://github.com/rajkripal/cashew/blob/main/config.example.yaml) — Canonical flat example of all keys — **HIGH confidence**
- [Cashew upstream `DESIGN.md`](https://github.com/rajkripal/cashew/blob/main/DESIGN.md) — Schema specification, BFS algorithm description, sleep protocol — **HIGH confidence**
- [sqlite-vec Python docs](https://alexgarcia.xyz/sqlite-vec/python.html) — Extension loading, `serialize_float32`, macOS caveats — **HIGH confidence**
- [sqlite-vec GitHub](https://github.com/asg017/sqlite-vec) — Virtual table syntax, `vec0` parameters — **HIGH confidence**
- [hermes-cashew `plugins/memory/cashew/__init__.py`](https://github.com/rajkripal/hermes-cashew/blob/main/plugins/memory/cashew/__init__.py) — Current `_ensure_db_schema()`, `prefetch()`, `handle_tool_call()` — **HIGH confidence**
- [hermes-cashew `plugins/memory/cashew/config.py`](https://github.com/rajkripal/hermes-cashew/blob/main/plugins/memory/cashew/config.py) — Current 4-key config — **HIGH confidence**
