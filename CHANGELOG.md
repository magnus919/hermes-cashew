# Changelog

## v0.3.0 (2026-05-12) — Stop Vibe-Coding, Start Integrating

This release is an object lesson in the dangers of **vibe coding** — the
trap of building from scratch when you should be integrating. We caught
ourselves reimplementing large swaths of cashew-brain (our upstream
dependency) instead of wrapping it. The result was:

- A custom `_ensure_db_schema` that duplicated upstream table creation
  and migration logic
- A retrieval pipeline (`_retrieve_with_vec` → `_retrieve_bfs` →
  `_retrieve_keyword` → `_score_nodes`) that was 80% boilerplate already
  handled by `retrieve_recursive_bfs()`
- An embarrassing bug where `vec_embeddings` used `rowid` to look up
  `thought_nodes.id` (SHA hashes) — semantic search always returned 0
- SQL parameter ordering bugs in the hand-rolled keyword fallback

The fix wasn't more code. It was **less**. We gutted 150+ lines of
custom retrieval, delegated to upstream's `retrieve_recursive_bfs()`,
and became a thin Hermes → Cashew adapter — which is what this project
was always meant to be.

### Changed

- **Upstream integration**: Upgraded cashew-brain to v1.1.0 (git SHA
  pin). Now uses `core.db.ensure_schema()` for schema management,
  `retrieve_recursive_bfs()` for three-tier retrieval (vec → BFS →
  keyword), and `get_schema_version()` for downstream migration
  branching.
- **Retrieval**: Replaced 150+ lines of custom retrieval with thin
  adapter around upstream. Removed `_retrieve_with_vec`,
  `_retrieve_bfs`, `_retrieve_keyword`, `_score_nodes`,
  `_apply_filters`, `_vec_available`, `_get_query_embedding`.
- **Config**: Removed dead `confidence_threshold` key (upstream dropped
  the column). 30 keys (down from 31).

### Fixed

- **vec_embeddings schema mismatch**: `_create_vec_embeddings` omitted
  `node_id TEXT primary key` and `distance_metric=cosine`, causing
  `SELECT rowid` to return sequential integers that never matched
  SHA-based node IDs. All vector queries returned 0 results.
  Includes migration for existing databases with the broken schema.
- **Plugin loading**: Fixed namespace collision with Hermes' built-in
  `plugins` package (`sys.path.insert(0, ...)` in `main.py:107` blocks
  PEP 420 resolution). Entry point `:register` suffix fixed (was
  returning function instead of module). `register_memory_provider`
  guarded for `PluginContext` (entry-point path) vs `_ProviderCollector`
  (directory path).
- **Test isolation**: `fake_embedder` fixture now patches
  `core.session` references too (upstream `end_session` imports
  `embed_nodes` at module level). Home leak snapshot uses file-path
  comparison instead of mtime (stops false positives from concurrent
  MCP server logging).

### Added

- Phase 11 E2E tests: full lifecycle (save → init → prefetch → sync →
  shutdown) and 4-thread concurrent DB stress test.
- `_migrate_vec_embeddings`: auto-drops stale vec_embeddings tables
  missing the `node_id` column and recreates with correct schema.
- `_keyword_search`: SQL `LIKE` fallback with individual term `AND`
  for environments where upstream can't embed.
- Dev install docs in `AGENTS.md` (Hermes venv + symlink).
