# Changelog

## v0.4.0 (2026-05-12) — Brain Operations

LLM integration via `auxiliary.memory` convention. The plugin now
delegates LLM-powered operations (extraction, think cycles, sleep
synthesis) to a Hermes auxiliary model instead of hardcoded
`model_fn=None`. Open source conventions established, issue tracker
reconciled against thin-adapter architecture.

### Added

- **LLM integration**: `llm_aux_role` config key — set to `"memory"`
  in cashew.json to wire upstream LLM features. Plugin reads Hermes
  own config.yaml to find `auxiliary.<role>`, resolves credentials,
  and constructs an OpenAI-compatible callable for upstream. ~80 lines,
  zero Hermes core changes. See README for setup guide.
- **CONTRIBUTING.md** — full open source contribution guide with DCO
  sign-off, Conventional Commits, PR process, and development setup
- **Issue templates** — bug report and feature request templates
- **PR template** — structured pull request template

### Changed

- **Release workflow**: now gates publishing behind tests passing,
  added `workflow_dispatch` trigger for manual re-runs
- **Test isolation**: `CASHEW_*` env vars stripped in conftest to
  prevent Hermes session variables from leaking into tests
- **Config env var parsing**: list-typed env vars now handle both
  comma-separated (`a,b,c`) and Python repr() format (`['a', 'b']`)
- **PyPI dependency**: cashew-brain switched from git+SHA pin to
  `>=1.1.0,<2.0.0` (PyPI rejected direct dependency URLs)

### Fixed

- **Entry point test**: updated to match v0.3.0 module-load contract
  (entry point returns module, not callable)
- **macOS fallback test**: removed reference to deleted `_retrieve_with_vec`
  method; gracefully skips mock when sqlite3.Connection is immutable

### Issue Tracking

All 9 open issues reconciled against thin-adapter architecture:

- **Closed**: #8 (think cycles), #9 (warm daemon), #10 (dashboard),
  #12 (domain separation), #13 (extractors), #14 (novelty gate),
  #20 (schema fork)
- **Re-scoped**: #15 (privacy → exclude_tags)
- **Implemented & closed**: #11 (LLM integration)

1 open issue remains: #15 (privacy / exclude_tags passthrough).

## v0.3.0 (2026-05-12) — Stop Vibe-Coding, Start Integrating
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
