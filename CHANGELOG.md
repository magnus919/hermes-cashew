# Changelog

## v0.10.0 (2026-05-18) — Background Dream Dispatch

### Added

- **`background_dream` parameter on `run_sleep_cycle()`** — When `True`,
  Phase 8 (LLM-powered dream generation) and Phase 9 (orphan embedding) run
  in a daemon thread instead of blocking the caller. The cross-linking, dedup,
  GC, and core memory phases still run synchronously (~20s), but the ~60s
  LLM call no longer blocks the session lifecycle. ([#60](https://github.com/magnus919/hermes-cashew/issues/60))

- **`_run_dream_async()` helper** — Opens its own SQLite connection in the
  daemon thread (WAL mode handles concurrent readers/writers). Logs completion
  or failure. Error in the background thread is caught and logged — never
  propagates.

- **`dream_pending` key in sleep cycle summary** — When background mode is
  active, the summary dict includes `dream_pending=True` and `orphans_embedded=0`
  (handled by the background thread). The `dream_id` field is `None` until the
  background thread completes.

### Changed

- **`CashewMemoryProvider.on_session_end()` now passes `background_dream=True`**
  to `run_sleep_cycle()`. The session lifecycle hook returns after the
  synchronous phases (~20s) instead of waiting for the full cycle (~84s).

### Performance

| Metric | Before | After |
|--------|--------|-------|
| `/new` session latency | ~118s (31s drain + 87s sleep) | ~51s (31s drain + 20s sync phases) |
| Dream still completes | Always (blocking) | Best-effort (daemon thread) |

## v0.9.0 (2026-05-15) — First-Load Bootstrap & Forest-Level Insight Extraction

### Added

- **Auto-generate `cashew.json` on first load** — when `initialize()` finds no
  config file, it writes one with all 32 DEFAULTS so `is_available()` returns
  `True` immediately after install. Existing configs are never overwritten.
  ([#47](https://github.com/magnus919/hermes-cashew/issues/47))

- **`llm_aux_role` defaults to `"memory"`** — LLM-powered extraction is active
  out of the box instead of requiring manual configuration. The plugin
  auto-populates `auxiliary.memory` in Hermes `config.yaml` from the main
  model config if absent. ([#47](https://github.com/magnus919/hermes-cashew/issues/47))

- **`is_available()` returns `True` when deps are present** — a fresh install
  shows green in `hermes memory status` even without a `cashew.json` file,
  because `initialize()` will generate defaults on first run.
  ([#47](https://github.com/magnus919/hermes-cashew/issues/47))

- **`on_pre_compress` hook** — Implements the `on_pre_compress(messages)` ABC
  method on `CashewMemoryProvider`. Uses a dedicated LLM extraction prompt
  (separate from `end_session`'s per-turn prompt) to identify conversation-arc
  patterns — topic shifts, framing changes, implicit decisions, unstated
  subjects, and recurring interaction patterns. Creates `insight`/`observation`
  nodes in the Cashew graph via upstream `_create_node` + `embed_nodes`.
  Silent-degrades without LLM configuration.
  ([#36](https://github.com/magnus919/hermes-cashew/issues/36))

### Changed

- **Both `plugin.yaml` files** — Added `on_pre_compress` to the hooks list.

## v0.8.2 (2026-05-14) — Hermes `hermes_plugins` Namespace Fix

### Fixed

- **GH#43** — Root `__init__.py` flat-entry detection missed Hermes' `hermes_plugins`
  synthetic namespace (introduced in Hermes 0.8.8+), causing the plugin to silently
  fail registration. The `_is_flat` check now includes `_spec_parent == "hermes_plugins"`
  alongside the existing `_hermes*` prefix checks.
  ([#44](https://github.com/magnus919/hermes-cashew/pull/44))

## v0.8.1 (2026-05-14) — Embedding Gap Closure Fix

### Fixed

- **GH#41** — `_embed_orphans()` failed with `NOT NULL constraint failed:
  embeddings.model` on every orphaned node. The primary INSERT used
  `model_name` (wrong column name — schema column is `model`), and the
  fallback omitted both `model` and `updated_at` (both NOT NULL). Orphaned
  nodes were never embedded, so the gap never closed.
  ([#42](https://github.com/magnus919/hermes-cashew/pull/42))

### Changed

- **Test schema** — `_create_schema` in tests now matches the upstream
  `embeddings` table exactly (with `model TEXT NOT NULL` and
  `updated_at TEXT NOT NULL`), preventing future column-mismatch regressions.

## v0.8.0 (2026-05-14) — Refactored Sleep Cycle Re-enabled

### Added

- **`plugins/memory/cashew/sleep_refactor.py`** — ground-up refactored sleep
  cycle replacing the upstream O(N²) implementation. Nine-phase pipeline:
  vectorized cross-linking (numpy + batched DB writes), connected-component
  dedup, node metrics, garbage collection, permanence evaluation, core memory
  promotion, LLM-powered dream generation, and embedding gap closure.
  Processes 7,100 nodes in ~4 seconds (vs hours → timeout upstream).

- **Sleep cycle re-enabled in `on_session_end()`** — runs automatically at
  session end when `sleep_cycles: true` in `cashew.json` and an LLM is wired
  via `llm_aux_role`. Work-capped at 2,000 nodes per cycle, converging
  gradually over ~3-4 sessions.

### Changed

- **`sleep_cycles` config flag is no longer a no-op** — now gates the
  refactored sleep cycle in `on_session_end()`. Set to `true` to enable.

### Fixed

- **GH#39** — Sleep cycle removed in v0.7.4 due to upstream performance
  (100% CPU for hours at 7K nodes). Replaced with vectorized refactored
  implementation.

## v0.7.4 (2026-05-13) — Remove Sleep Cycle from Lifecycle Hooks

### Removed

- **Sleep cycle removed from `on_session_end()` and `shutdown()`**: The
  upstream `run_sleep_cycle()` is too heavyweight to run synchronously in
  lifecycle hooks. With 6K+ nodes and 59K+ edges, it computes a full N×N
  embedding similarity matrix (36M+ comparisons), performs Bron–Kerbosch
  clique detection, and runs per-node graph traversals via
  `calculate_node_metrics()` — taking hours and accumulating overlapping
  instances when multiple session ends fire. Sleep consolidation will be
  reimplemented in a dedicated background thread with a mutex guard and
  configurable interval (see GH#42).
- **`sleep_cycles` config flag is now a no-op** until the redesign is
  complete. Set it to `False` in `cashew.json` to avoid confusion.

## v0.7.3 (2026-05-12) — sqlite-vec Fix

### Fixed
- **sqlite-vec not loading on macOS**: Changed `conn.load_extension("vec0")`
  to prefer `sqlite_vec.load(conn)` (from the pip package), with fallback
  to bare name. The old code failed silently on macOS because vec0.dylib
  isn't in the standard library load path. Install sqlite-vec in the
  Hermes venv to enable vector search acceleration.

## v0.7.2 (2026-05-12) — Spec Compliance

Compliance fixes from the memory provider plugin audit.

### Added
- **README.md** in plugin directory (required by the plugin spec).

### Changed
- **Worker thread**: `daemon=False` → `daemon=True` to match the Hermes
  memory provider spec threading contract.

### Fixed
- **Sleep cycle shutdown fallback**: While upstream fix #11410 means
  `on_session_end()` fires on session expiry in current Hermes, adding a
  fallback in `shutdown()` protects against edge cases where the session
  end path is skipped (crash, process kill, long-running sleep cycle
  interrupted by restart). Uses a flag to prevent duplication when both
  paths fire in sequence.

## v0.7.1 (2026-05-12) — Sleep on Session End

Moved sleep cycle from `shutdown()` to `on_session_end()` and fixed a
silent `AttributeError` on the return value.

### Fixed

- **Sleep cycle now runs on `on_session_end()` instead of `shutdown()`**:
  `on_session_end()` is the correct lifecycle hook for "final extraction/flush"
  per the Hermes memory provider docs. `shutdown()` is for connection cleanup.
  This makes sleep cycles observable and testable — they fire after every
  conversation session, not just on process restart. (GH#33)

- **Return type bug in sleep cycle logging**: Upstream `run_sleep_cycle()`
  returns a plain `Dict`, but the code was accessing `.new_nodes` and
  `.new_edges` attributes — causing a silent `AttributeError` caught by the
  generic exception wrapper. Fixed to use `result.get(...)` with meaningful
  metric names (cross-links, dedups, dreams, decayed, promotions, demotions).

Sleep cycles on shutdown and think cycles on a periodic interval.
Together they complete the upstream brain operation pipeline —
extraction → thinking → sleeping → consolidation.

### Added

- **think_interval**: Config key (default 10) controls how many sync
  turns pass between upstream think_cycle() calls. Discovers cross-
  domain connections and generates insight nodes. Set to 0 to disable.
- **Sleep cycles on shutdown**: Calls upstream run_sleep_cycle() when
  the provider shuts down (gateway restart), if LLM is wired and
  sleep_cycles is True. Graph consolidation without blocking the hot
  path.
- **README**: LLM Integration section updated with think_interval docs.

## v0.6.0 (2026-05-12) — Think Cycles

*Release was published but changelog entry was deferred to v0.7.0.
All v0.6.0 changes are included in the v0.7.0 entry above.*

## v0.5.0 (2026-05-12) — Privacy & Visibility

Privacy controls via `exclude_tags` filtering on all retrieval paths.
The final open issue from the original milestone plan is resolved.

### Added

- **exclude_tags filtering**: `cashew_query` tool accepts optional
  `exclude_tags` array parameter to filter out tagged nodes from results.
  Works in both upstream retrieval (`retrieve_recursive_bfs()`) and
  keyword fallback (SQL `NOT LIKE` exclusion).
- **Privacy Controls section**: documented in README with use cases
  (vault:private tagging, domain isolation, declassification).

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
