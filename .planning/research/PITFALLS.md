# Domain Pitfalls

**Domain:** hermes-cashew v0.2.0 — sqlite-vec retrieval, schema migration, config expansion
**Researched:** 2026-04-21
**Confidence:** HIGH (codebase fully read; sqlite-vec docs scraped; SQLite/alembic patterns verified)

---

## Critical Pitfalls

Mistakes that cause rewrites, data loss, or broken v0.1.0 users.

### Pitfall 1: sqlite-vec `load()` fails silently on macOS system Python
**What goes wrong:** `sqlite_vec.load(db)` raises `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'` on macOS because Apple's bundled SQLite omits extension support. If unhandled, `initialize()` crashes and the provider silent-degrades permanently. Users on stock macOS Python cannot use vector search.
**Why it happens:** The sqlite-vec Python package is a thin wrapper around a C extension that must be loaded via `conn.enable_load_extension(True)`. macOS system Python disables this for security.
**Consequences:** All macOS users with unmodified Python lose semantic search; the plugin falls back to a degraded state with no vector index.
**Prevention:**
1. Wrap `sqlite_vec.load()` in a `try/except (AttributeError, RuntimeError)`.
2. Set a `_sqlite_vec_available` flag on the provider.
3. In retrieval path, branch: if sqlite-vec unavailable, fall back to the existing keyword/`LIKE`-based retrieve path (or return empty with a log line).
4. Document the Homebrew Python workaround (`brew install python`) in README.
**Detection:** CI should test on macOS runner with stock Python; look for `AttributeError` on `enable_load_extension`. A dedicated test can assert graceful fallback.
**Phase to prevent:** Phase 1 (Retrieval Core) — the retrieval implementation must include the fallback branch.

### Pitfall 2: vec0 virtual table schema is immutable — additive migration requires full rebuild
**What goes wrong:** sqlite-vec stores vectors in a `vec0` virtual table. Unlike regular SQLite tables, virtual tables do **not** support `ALTER TABLE ADD COLUMN`. If v0.2.0 needs to change the vector schema (dimensions, metadata columns, distance metric), the only path is `DROP TABLE` + recreate, which destroys all embeddings.
**Why it happens:** Virtual tables are implemented via callbacks; SQLite's ALTER machinery doesn't apply. sqlite-vec's `vec0` table shape is fixed at `CREATE VIRTUAL TABLE` time.
**Consequences:** Any schema change to the vector table requires regenerating every embedding — a multi-minute operation on large graphs and a data-loss event if the old vectors aren't recomputable.
**Prevention:**
1. Design the `vec0` table once and freeze its schema for the whole v0.2.x series.
2. Store any mutable metadata in a regular SQLite table (joinable to the vec0 table by `node_id`), not in the vec0 table itself.
3. If a vec0 rebuild is ever unavoidable, implement it as an explicit offline migration tool, not an automatic on-load step.
**Detection:** Search for `ALTER TABLE` in any migration code touching a `vec0` table; it's a guaranteed bug.
**Phase to prevent:** Phase 1 (Retrieval Core) — table design happens here.

### Pitfall 3: v0.1.0 DB migration drops user data because `ALTER TABLE` is used for NOT NULL/DEFAULT changes
**What goes wrong:** The existing `_migrate_edges_timestamp_not_null` already does a table rebuild (copy → drop → rename) because SQLite can't ALTER a column constraint. v0.2.0 adds columns like `reasoning`, `mood_state`, `metadata` to `thought_nodes`. If a migration script naively tries `ALTER TABLE thought_nodes ADD COLUMN reasoning TEXT` on a v0.1.0 DB that already has data, it works; but if it then tries to set `NOT NULL DEFAULT` on an existing nullable column, SQLite rejects it. The "fix" of recreating the table can lose rows if the copy logic is buggy.
**Why it happens:** SQLite supports `ALTER TABLE ADD COLUMN` only for simple cases. Changing constraints, defaults, or column types requires the copy-rebuild dance. Each rebuild is a rewrite opportunity for bugs.
**Consequences:** Corrupted or truncated `thought_nodes` / `derivation_edges` tables; user loses their thought graph.
**Prevention:**
1. Keep all v0.2.0 changes **strictly additive** via `ALTER TABLE ... ADD COLUMN` with no `NOT NULL` without `DEFAULT`.
2. Never rebuild `thought_nodes` or `derivation_edges` automatically on plugin load. If a constraint change is needed, defer it to a manual maintenance command.
3. Before any migration runs, `PRAGMA integrity_check` and `PRAGMA foreign_key_check`.
4. Wrap the entire migration in a transaction; on any exception, `ROLLBACK` and silent-degrade (log WARNING, set `_retriever = None`).
**Detection:** Unit test that creates a v0.1.0-schema DB, seeds it with rows, runs migration, and asserts row counts + cell contents are identical.
**Phase to prevent:** Phase 2 (Schema Migration) — the migration engine must be hardened here.

### Pitfall 4: Config expansion breaks existing tests and `hermes memory setup`
**What goes wrong:** `test_e2e_install_lifecycle_schema_returns_field_descriptors` asserts `len(schema) == 4` and `keys == {'cashew_db_path', 'embedding_model', 'recall_k', 'sync_queue_timeout'}`. `test_config_roundtrip.py` asserts `set(DEFAULTS.keys()) == {those 4}`. Expanding to ~30 keys without updating tests causes immediate test failures. Worse, Hermes's `memory_setup.py` iterates the schema list to build an interactive prompt; 30 fields make for a poor UX but won't crash.
**Why it happens:** The config schema is the contract between the plugin and both Hermes and the test suite. The current tests pin the exact shape.
**Consequences:** CI goes red; `hermes memory setup` shows 30 fields (some possibly confusing); users with existing 4-key `cashew.json` files may see dropped keys if `save_config` still filters against `dataclasses.fields(CashewConfig)`.
**Prevention:**
1. Update all config tests to assert `len(schema) >= 4` (not `== 4`) and that the original 4 keys are still present, rather than pinning the exact set.
2. Keep the original 4 keys' defaults identical to v0.1.0 values.
3. In `load_config`, continue the existing pattern: `merged = {**DEFAULTS, **raw}` then filter to known keys. Existing 4-key files load fine; new keys get defaults.
4. In `save_config`, decide whether to write all ~30 keys (verbose but complete) or only non-default keys (minimal diff). Recommendation: write all keys with defaults so the file is self-documenting, but sort them so diffs are stable.
**Detection:** `pytest tests/test_config_roundtrip.py tests/test_e2e_install_lifecycle.py` will fail immediately.
**Phase to prevent:** Phase 3 (Config Alignment) — update tests as part of the config expansion PR.

### Pitfall 5: Env var overrides bypass `save_config` and leak secrets into the JSON file
**What goes wrong:** If env var overrides (e.g., `CASHEW_RECALL_K=10`) are merged at `load_config` time, a later `save_config` may write the overridden value back to `cashew.json`, permanently mutating the user's config. If the override was a secret (e.g., API key — though cashew has none), it leaks to disk.
**Why it happens:** The current `save_config` takes the `values` dict and writes it merged over DEFAULTS. If env vars are injected into `values` during the load→save roundtrip, they become persistent.
**Consequences:** Ephemeral env overrides become permanent config; CI/test overrides pollute user files; `hermes memory setup` may show values the user never intended to save.
**Prevention:**
1. Separate "file config" from "effective config". `CashewConfig` (effective) is a superset; `load_config` returns file values only.
2. Env var overrides are applied in a separate layer **after** `load_config`, inside the provider or a `resolve_effective_config()` helper.
3. `save_config` only ever writes values that came from the file or from Hermes's `memory setup` interaction — never env-derived values.
4. Do not add any secret fields to the cashew schema (it doesn't need them), but if future phases do, mark them `secret=True` and exclude from `save_config`.
**Detection:** Test: set env var, `load_config`, `save_config`, read file back — assert env value is NOT in file.
**Phase to prevent:** Phase 3 (Config Alignment) — env var logic lives here.

### Pitfall 6: Migration runs while the sync worker holds the DB open
**What goes wrong:** `initialize()` currently creates the queue, then calls `_ensure_db_schema`, then starts the worker. If v0.2.0 makes migration more complex (e.g., rebuilding a table), the migration holds an exclusive lock. The worker thread may start and attempt to `end_session` (which opens its own connection) before migration commits, causing `sqlite3.OperationalError: database is locked` or a deadlock.
**Why it happens:** SQLite uses file-level locking. A long-running transaction in `initialize()` blocks writers. The worker thread starts as soon as `_start_sync_worker()` is called.
**Consequences:** Worker crashes on first turn, logs WARNING, and continues; turns pile up in queue; user sees repeated "cashew sync worker: turn failed" warnings.
**Prevention:**
1. Run all migration code **before** `_start_sync_worker()` is called. The current ordering in `initialize()` is already correct (schema → worker), but verify it stays that way.
2. Keep migration transactions short. If a table rebuild is unavoidable, do it in `initialize()` before worker start, commit, then close the connection before starting the worker.
3. Never start the worker inside a `with sqlite3.connect(...)` context manager that spans migration + worker startup.
**Detection:** Test with a synthetic slow migration (`time.sleep(0.5)` inside the migration transaction) and a worker that immediately tries to write; assert no `database is locked` warning.
**Phase to prevent:** Phase 2 (Schema Migration) — ordering is part of the migration design.

### Pitfall 7: BFS traversal triggers RecursionError or unbounded query cost
**What goes wrong:** `retrieve_recursive_bfs()` walks the graph from seed nodes via `derivation_edges`. If the graph contains cycles (A→B→C→A) or a very dense cluster, naive recursive BFS either blows the Python call stack or issues N+1 queries, each round-tripping to SQLite.
**Why it happens:** Graphs built from conversational extraction naturally form cycles (revisiting topics). Python's default recursion limit is ~1000. SQLite without `PRAGMA recursive_triggers` won't detect cycles in SQL.
**Consequences:** `RecursionError` crashes the retrieval call (caught by the existing `except Exception` in `prefetch`, so it silent-degrades to empty — but every query fails, making memory useless). Or, slow retrieval (>10s) blocks the Hermes turn.
**Prevention:**
1. Implement BFS iteratively (deque), not recursively.
2. Hard cap `max_depth` (e.g., 3–5 hops) and `max_total_nodes` (e.g., 50).
3. Track visited `node_id`s in a `set()` to avoid cycles.
4. Issue edges as a single parameterized query per BFS wave (`SELECT * FROM derivation_edges WHERE parent_id IN (...)`), not one query per node.
5. Add a timeout guard around the entire BFS (e.g., 2s) and return partial results if exceeded.
**Detection:** Unit test with a cyclic graph (A→B→C→A) and a deep chain (100 nodes); assert no `RecursionError`, retrieval completes in <100ms.
**Phase to prevent:** Phase 1 (Retrieval Core) — BFS implementation lives here.

### Pitfall 8: `is_available()` probes the new vector table and breaks the zero-I/O contract
**What goes wrong:** v0.2.0 might be tempted to check `vec0` table existence in `is_available()` to report "vector search ready". The current contract is: `is_available()` does **one** `Path.exists()` call on `cashew.json`. Adding any DB probe (table existence, PRAGMA query, etc.) violates this and may trigger `database is locked` or embedding model load.
**Why it happens:** Hermes calls `is_available()` frequently and from threads. If it opens the DB, it competes with the sync worker.
**Consequences:** Hermes UI hangs; `database is locked` errors in logs; CI tests that assert exactly one `Path.exists` call fail.
**Prevention:**
1. `is_available()` must remain a config-file probe only. Vector readiness is a separate internal flag (`_vec_search_available`) checked only at retrieval time.
2. If the UI needs to show "vector search: on/off", expose it in `system_prompt_block()` (which already does a fast `COUNT(*)` query, but only when called, not on every `is_available` poll).
**Detection:** `test_is_available_calls_path_exists_exactly_once` will fail immediately.
**Phase to prevent:** Phase 1 (Retrieval Core) — do not change `is_available()`.

---

## Moderate Pitfalls

### Pitfall 9: Embedding dimension mismatch between old and new models
**What goes wrong:** The config's `embedding_model` may change (e.g., user switches from `all-MiniLM-L6-v2` (384-dim) to `BAAI/bge-small-en` (384-dim, different tokenizer) or a 768-dim model). The `vec0` table was created with a fixed dimension. Inserting a differently-sized vector raises a sqlite-vec error.
**Why it happens:** sqlite-vec `vec0` tables declare the dimension at creation (`embedding float[384]`). Changing the model without rebuilding the table is a type error.
**Prevention:**
1. Store the model name and dimension in a `meta` table at initialization.
2. On `initialize()`, if the configured model differs from the stored model, log a WARNING and either (a) rebuild the vec0 table, or (b) disable vector search until a manual migration is run.
3. For v0.2.0, pin the dimension to 384 (MiniLM's size) in the `vec0` schema and document that changing models requires a rebuild.
**Phase to prevent:** Phase 1 (Retrieval Core).

### Pitfall 10: `recall_k` config field is overloaded between semantic and graph recall
**What goes wrong:** `recall_k` currently means "max nodes from Cashew's retrieve". v0.2.0 has two retrieval paths: sqlite-vec semantic search (seeds) + BFS graph walk (expansion). If both paths respect the same `recall_k`, a user setting `recall_k=5` might get 2 seed nodes + 3 BFS nodes, or 5 seeds + 5 BFS nodes, depending on implementation. The behavior is unpredictable.
**Why it happens:** Two distinct operations (vector top-k, graph traversal depth/breadth) are compressed into one integer.
**Prevention:**
1. Split into `recall_k` (semantic seeds, default 5) and `graph_expand_k` (max additional nodes from BFS, default 10) or `bfs_max_depth` / `bfs_max_nodes`.
2. Keep `recall_k` for backward compatibility; add new keys with defaults. The total context is bounded by `recall_k + graph_expand_k`.
**Phase to prevent:** Phase 3 (Config Alignment).

### Pitfall 11: Tests asserting exact schema keys break when new config keys are added
**What goes wrong:** `test_config_roundtrip.py` has `test_defaults_contain_exactly_four_keys_with_documented_values` and `test_cashew_config_dataclass_field_set_matches_defaults`. Adding 26 new keys without updating these tests causes failures.
**Why it happens:** Tests were written when 4 keys was the whole universe.
**Prevention:**
1. Update assertions to check that the **original 4 keys** are present with correct defaults, and that new keys have sensible defaults — not that the total is exactly 4.
2. Add a separate test for "no key lacks a default" to prevent mandatory-user-input regressions.
**Phase to prevent:** Phase 3 (Config Alignment) — tests updated in same PR.

### Pitfall 12: `fake_embedder` fixture doesn't block sqlite-vec native loads
**What goes wrong:** `conftest.py`'s `fake_embedder` patches `core.embeddings.*` functions. sqlite-vec does its own vector computation in C; it doesn't use `core.embeddings`. However, if the retrieval path calls any sentence-transformers code (e.g., to embed the query), that will try to download/load the model in CI.
**Why it happens:** sqlite-vec itself is just a storage/query engine. Embedding the query still requires a model. If v0.2.0 uses the same `core.embeddings` path as v0.1.0, the fixture still works. But if it switches to a different embedder (e.g., `sqlite_vec` helper or direct `sentence_transformers` call), the fixture misses it.
**Prevention:**
1. Ensure query embedding goes through the same `core.embeddings.embed_text` path that `fake_embedder` blocks, OR add a new fixture that patches any new embedding entry point.
2. In CI, keep the `HF_HUB_OFFLINE=1` guard and the log-scan for `Downloading.*MiniLM`.
3. For unit tests of retrieval, mock the embedding step entirely (return a fixed float vector).
**Phase to prevent:** Phase 1 (Retrieval Core) — test infrastructure updated alongside retrieval.

---

## Minor Pitfalls

### Pitfall 13: `system_prompt_block()` COUNT query slows down on large graphs
**What goes wrong:** `system_prompt_block()` does `SELECT COUNT(*), (SELECT COUNT(*) FROM derivation_edges) FROM thought_nodes`. On a 100K-node graph, this can take 100ms+ and is called every time Hermes assembles a system prompt (potentially every turn).
**Why it happens:** `COUNT(*)` without `WHERE` is a full table scan in SQLite unless an index exists.
**Prevention:**
1. Add `sqlite3` execution around the query with a short timeout, or cache the counts and invalidate on sync_turn completion.
2. For v0.2.0, an index on `thought_nodes(id)` already exists (PRIMARY KEY), but not on `derivation_edges`. Consider if an index on `derivation_edges(parent_id, child_id)` is worthwhile — it's already the PRIMARY KEY, so it's indexed.
3. The real fix: `COUNT(*)` is fast enough for <10K rows; if users grow beyond that, add a `node_count` / `edge_count` cached counter maintained by triggers or by the sync worker.
**Phase to prevent:** Phase 1 (Retrieval Core) — or defer to a performance milestone.

### Pitfall 14: ` CashewConfig` dataclass becomes unwieldy at 30 fields
**What goes wrong:** A 30-field `@dataclass(frozen=True)` creates a 30-argument `__init__`. `load_config` uses `CashewConfig(**filtered)`, which is fine, but reading/configuring becomes error-prone.
**Why it happens:** Flat config scales poorly.
**Prevention:**
1. Group related fields into nested dataclasses (e.g., `RetrievalConfig`, `SyncConfig`, `GCConfig`) while keeping the **on-disk JSON flat** for backward compatibility. The loader flattens JSON into nested dataclass kwargs via `__post_init__` or a custom factory.
2. Keep `get_config_schema()` flat (Hermes expects a flat list), but the internal typed representation can be nested.
**Phase to prevent:** Phase 3 (Config Alignment).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| **Phase 1: sqlite-vec + BFS retrieval** | Pitfall 1 (macOS load failure), Pitfall 7 (BFS recursion/cycles), Pitfall 8 (is_available contract), Pitfall 9 (dimension mismatch), Pitfall 12 (fake_embedder gaps) | Wrap sqlite-vec load in try/except; iterative BFS with depth+node caps; keep `is_available()` unchanged; pin dimension or auto-detect; extend fixture coverage |
| **Phase 2: Schema migration** | Pitfall 2 (vec0 immutability), Pitfall 3 (data loss in migration), Pitfall 6 (migration vs worker lock) | Never ALTER vec0; only additive ALTER on regular tables; migration before worker start; integrity_check before/after |
| **Phase 3: Config expansion** | Pitfall 4 (test breakage), Pitfall 5 (env var persistence), Pitfall 10 (recall_k overload), Pitfall 11 (test assertions), Pitfall 14 (dataclass bloat) | Update tests to `>=4` keys; separate env layer from save layer; split recall_k semantic; assertions on original keys + defaults only; nested internal config |
| **Phase 4: Integration & QA** | All critical pitfalls surface here during E2E testing | Run full lifecycle test with a v0.1.0 DB artifact; test on macOS runner; test cyclic graph; test env var roundtrip |

---

## Sources

- sqlite-vec Python docs: https://alexgarcia.xyz/sqlite-vec/python.html (scraped 2026-04-21) — macOS `enable_load_extension` limitation, `serialize_float32()`, version requirements
- sqlite-vec GitHub / PyPI — vec0 virtual table behavior, no ALTER support
- SQLite ALTER TABLE docs: https://www.sqlite.org/lang_altertable.html — limited ALTER support
- Alembic batch migrations for SQLite: https://alembic.sqlalchemy.org/en/latest/batch.html — copy-rebuild pattern
- Software Engineering Stack Exchange: "How to avoid breaking backward compatibility because of database changes" — additive-only migration philosophy
- Code Without Rules: "The tragic tale of the deadlocking Python queue" — queue/threading gotchas
- SQLite locking docs: https://sqlite.org/lockingv3.html — file-level locking behavior
- hermes-cashew codebase (v0.1.5): `plugins/memory/cashew/__init__.py`, `config.py`, `tools.py`, `tests/` — contract specifics, test assertions, threading model
