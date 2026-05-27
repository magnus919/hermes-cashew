# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph with semantic search and automatic context recall. Get from zero to
a working install in under five minutes.

**v0.9.0** auto-generates `cashew.json` with defaults on first load, enables
LLM-powered extraction by default (no manual `auxiliary.memory` setup), and
adds forest-level insight extraction via `on_pre_compress`. **v0.8.0**
re-enabled the sleep cycle with a ground-up refactored implementation —
vectorized cross-linking, batched DB writes, ~4s at 7K nodes.

## Prerequisites

- [Hermes Agent](https://github.com/nousresearch/hermes-agent) installed
- `cashew-brain>=1.0.0` — installed automatically by `hermes plugins install`
- `sqlite-vec` — optional, enables semantic search (install below if wanted)

## Install

```bash
hermes plugins install magnus919/hermes-cashew
```

This clones the repository to `~/.hermes/plugins/cashew/` and registers the
plugin entry point. After install, restart the gateway:

```bash
hermes gateway restart
```

## Register with Hermes

After installing, set cashew as the active memory provider:

```bash
hermes config set memory.provider cashew
hermes gateway restart
```

Or use the interactive setup (v0.2.0 now includes cashew in the provider picker):

```bash
hermes memory setup
```

## Zero-Config Startup

hermes-cashew works out of the box — all 32 configuration keys have sane
defaults. On first agent startup, the plugin auto-generates `~/.hermes/cashew.json`
with the full default configuration and auto-populates `auxiliary.memory` in
Hermes `config.yaml` from the main model config, so LLM-powered extraction
is active without any manual setup.

Created `~/.hermes/cashew.json` only if you want to override specific defaults.
The file is never overwritten once it exists:

```bash
# Optional: override individual defaults
cat > ~/.hermes/cashew.json << 'EOF'
{
  "recall_k": 10,
  "think_interval": 15,
  "user_domain": "user"
}
EOF
```

### Full Config Reference

#### Core

| Key | Default | Description |
|-----|---------|-------------|
| `cashew_db_path` | `cashew/brain.db` | Path to SQLite DB, relative to `hermes_home` |
| `embedding_model` | `thenlper/gte-large` | Sentence-transformers model for embeddings (1024-dim) |
| `llm_aux_role` | `memory` | Hermes auxiliary role for LLM-powered extraction; requires `auxiliary.memory` in `config.yaml` |
| `auto_extraction` | `true` | Auto-extract knowledge from conversation turns |
| `sync_queue_timeout` | `30.0` | Seconds to wait for sync worker drain on shutdown |

#### Retrieval

| Key | Default | Description |
|-----|---------|-------------|
| `recall_k` | `5` | Context fragments returned per query |
| `similarity_threshold` | `0.3` | Minimum similarity for BFS graph walk |
| `walk_depth` | `2` | Graph BFS traversal depth |
| `token_budget` | `2000` | Max tokens per context injection |
| `prefetch_k` | `3` | Nodes to pre-warm into context on each turn |
| `prefetch_cues` | `3` | Cue phrases to send to LLM for prefetch generation |

#### Domains & Classification

| Key | Default | Description |
|-----|---------|-------------|
| `user_domain` | `user` | Domain label for user messages |
| `ai_domain` | `ai` | Domain label for AI messages |
| `default_domain` | `general` | Fallback domain for unclassified content |
| `auto_classify` | `true` | Auto-classify nodes into domains |
| `domain_classifications` | `["personal", "work", "projects", "learning", "system"]` | Available domain labels |
| `domain_separation_enabled` | `true` | Enforce domain boundaries in retrieval |

#### Sleep Cycle

| Key | Default | Description |
|-----|---------|-------------|
| `sleep_cycles` | `true` | Enable the refactored sleep cycle (cross-linking, dedup, GC, dreams) |
| `sleep_schedule` | `"every 12h"` | Cron schedule for sleep cycle |
| `sleep_max_nodes` | `2000` | Max nodes per sleep cycle tick |
| `think_cycles` | `true` | Enable periodic insight generation (think cycle) |
| `think_interval` | `10` | Turns between think cycle runs (0 = disable) |
| `think_cycle_nodes` | `5` | Node clusters per think cycle |
| `max_think_iterations` | `3` | Max iterative refinements per think cycle |
| `novelty_threshold` | `0.82` | Minimum novelty score to surface an insight |

#### Garbage Collection

| Key | Default | Description |
|-----|---------|-------------|
| `gc_mode` | `soft` | `"soft"` or `"hard"` decay |
| `gc_threshold` | `0.05` | Minimum importance score before decay |
| `gc_grace_days` | `7` | Days before a node can be decayed |
| `gc_protect_types` | `["seed", "core_memory"]` | Node types exempt from decay |
| `gc_think_cycle_penalty` | `1.5` | Importance penalty multiplier for think-cycle nodes |
| `decay_pruning` | `true` | Prune low-value nodes over time |
| `pattern_detection` | `true` | Detect recurring patterns in extracted knowledge |

#### Tuning

| Key | Default | Description |
|-----|---------|-------------|
| `access_weight` | `0.2` | Weight of access count in importance scoring |
| `temporal_weight` | `0.1` | Weight of recency in importance scoring |
| `clustering_eps` | `0.35` | DBSCAN epsilon for think-cycle clustering |
| `clustering_min_samples` | `3` | Minimum samples per cluster in think cycle |

Environment variables override config values: prefix any key with `CASHEW_`
(e.g. `CASHEW_RECALL_K=10`).

## Verify the Install

```bash
hermes gateway restart   # ensure gateway picks up the new plugin
hermes memory status
```

Expected output shows `Provider: cashew` with `Plugin: installed` and `Status: available`.

## How It Works

`hermes-cashew` provides two LLM-accessible tools:

- **`cashew_query`** — searches the local thought graph for context relevant to
  the current conversation. Uses sqlite-vec for semantic search when available,
  with keyword fallback on macOS or when the extension is unavailable.
- **`cashew_extract`** — explicitly persists a conversation turn into the graph.
  The agent can call this when it judges a turn contains worth-remembering knowledge.

Both tools are registered automatically when Hermes loads the plugin.
On each session start, `prefetch()` retrieves relevant context from the graph
and injects it into the system prompt.

## Privacy Controls (Optional)

Nodes in the thought graph can carry tags. The `cashew_query` tool accepts an
`exclude_tags` parameter to filter out nodes with specific tags from results:

```json
{"query": "prior decisions", "exclude_tags": ["vault:private"]}
```

This works in both the upstream retrieval path (sqlite-vec / BFS) and the
keyword fallback. Common use cases:

- **Privacy**: Tag sensitive nodes with `vault:private` to exclude them from
  group or shared contexts
- **Domain isolation**: Exclude nodes from specific domains during broad queries
- **Declassification**: Remove exclusion to reveal previously private nodes

## LLM Integration

hermes-cashew enables LLM-powered extraction by default — `llm_aux_role` is
set to `"memory"`, and the plugin auto-populates `auxiliary.memory` in Hermes
`config.yaml` from the main model config on first load.

No manual configuration is needed. To verify LLM extraction is active:

```bash
grep "using" ~/.hermes/logs/agent.log | grep "llm_aux_role"
# Expected: llm_aux_role='memory': using <provider> <model> via <base_url>
```

To disable LLM extraction (heuristic-only mode), set `llm_aux_role` to null in
`cashew.json`:

```json
{"llm_aux_role": null}
```

Or remove the section entirely — the default will regenerate it on next start.

### What the LLM enables upstream

- **LLM extraction** — structured knowledge extraction with typed nodes,
  confidence scores, tags, and domain assignment
- **Think cycles** — cross-domain synthesis, generates `insight` nodes
  from clusters of related knowledge. Runs every `think_interval` sync
  turns (default 10). Set `think_interval` to 0 to disable.
- **Sleep synthesis** — Graph consolidation pipeline: cross-linking, dedup,
  garbage collection, permanence evaluation, core memory promotion, and
  LLM-powered dream generation. Runs as a **Hermes cron job** on a configurable
  schedule (default: every 12 hours), not at session boundaries. The cron script
  reads `cashew.json` at runtime and operates without an LLM — if LLM-powered
  dream synthesis is desired, it requires additional configuration (see
  [Sleep Cycle Cron Scheduling](#sleep-cycle-cron-scheduling) below).
  Processes up to `sleep_max_nodes` per cycle (default 2,000).
- **Pre-compress insight extraction** — Before context compression discards
  old messages, extracts conversation-arc patterns (topic shifts, framing
  changes, implicit decisions) using a dedicated LLM prompt. Creates
  `insight`/`observation` nodes in the graph. Requires `llm_aux_role`
  configuration. Silent-degrades without LLM.

Without `llm_aux_role`, the plugin uses heuristic-only extraction — no
API calls, no LLM cost, zero-config.

**Design note:** The `auxiliary.memory` convention is provider-agnostic.
Any memory provider plugin can declare `llm_aux_role` and reference the
same `auxiliary.memory` section, making this a standard pattern across
the Hermes plugin ecosystem.

## Sleep Cycle Cron Scheduling

hermes-cashew runs its graph consolidation pipeline (cross-linking, dedup,
garbage collection, permanence evaluation, core memory promotion) as a
**Hermes ``no_agent`` cron job**, not at session boundaries. This means
``/new`` returns instantly — no synchronous sleep cycle work blocks the
start of a new session.

### When the cron job is registered

The cron job is created during plugin initialization (``initialize()``) only
when **all** of the following are true:

| Condition | Config Key | Default | Behavior if false |
|-----------|-----------|---------|-------------------|
| Sleep cycles enabled | ``sleep_cycles`` | ``true`` | Cron not registered |
| Schedule non-empty | ``sleep_schedule`` | ``\"every 12h\"`` | Cron not registered; set to ``\"\"`` to disable |
| Provider init succeeds | — | — | Exception caught, ``_config`` set to ``None``, cron never reached |
| Hermes cron module available | — | — | ``ImportError`` caught, WARNING logged |
| ``create_job()`` succeeds | — | — | Exception caught, WARNING logged |
| No job already registered for this provider instance | — | — | No-op dedup guard |

The cron job is **removed** on plugin shutdown (``shutdown()``). A dedup
helper scans for existing ``cashew-sleep-cycle`` jobs by name on each
registration to prevent N jobs accumulating across N restarts.

### When the cron job runs

On the configured schedule (default ``every 12h``), the Hermes scheduler
executes ``$HERMES_HOME/scripts/cashew-sleep-cycle.py`` with **no LLM** —
it is a ``no_agent`` script, meaning zero LLM overhead per tick. The script
reads ``cashew.json`` at runtime to discover its database path and
``sleep_max_nodes`` setting.

### What happens during a cron tick

1. Reads ``cashew.json`` to get ``cashew_db_path`` and ``sleep_max_nodes``
2. Selects up to ``sleep_max_nodes`` (default 2,000) oldest-unprocessed nodes
3. Computes pairwise cosine similarity (vectorized numpy)
4. Creates cross-links between similar node pairs (threshold: 0.78)
5. Deduplicates near-identical nodes (threshold: 0.82) via BFS clustering
6. Runs garbage collection on low-fitness isolated nodes
7. Promotes frequently-accessed nodes to permanent / core memory status
8. Prints a JSON summary (captured by the cron scheduler's output log)

**No LLM-powered dream generation** occurs in cron mode — the script passes
``model_fn=None``. Cross-linking, dedup, and GC are the 80% benefit without
the API key dependency in a subprocess.

### Config reference

| Key | Default | Description |
|-----|---------|-------------|
| ``sleep_schedule`` | ``\"every 12h\"`` | Cron expression or interval string. Set to ``\"\"`` to disable cron-based scheduling entirely. Examples: ``\"every 30m\"``, ``\"0 */2 * * *\"``, ``\"0 3 * * *\"`` (daily at 3am). |
| ``sleep_max_nodes`` | ``2000`` | Maximum number of nodes to cross-link in a single sleep cycle. Higher values converge faster but take longer per tick. |

## Semantic Search (Optional)

`sqlite-vec` is an optional SQLite extension that enables vector similarity search.
Without it, cashew falls back to keyword-based retrieval — still functional,
but less precise.

**Install:**
```bash
pip install sqlite-vec
```

You may also need to enable load extension support in your SQLite build:
```bash
sqlite3_config(SQLITE_ENABLE_LOAD_EXTENSION)
```

If sqlite-vec is not available at runtime, you'll see this INFO log on startup:
```
sqlite-vec not available; semantic search will use fallback
```

This is normal and expected on systems without sqlite-vec support.

## Uninstall

```bash
hermes plugins remove cashew
hermes config set memory.provider built-in   # revert to built-in memory
rm -rf ~/.hermes/cashew   # optional: remove the local graph data
```

## Troubleshooting

### `Plugin: NOT installed` in `hermes memory status`

1. **cashew-brain not installed in Hermes venv** — `hermes plugins install` does not
   automatically install Python package dependencies into Hermes's venv. Install it manually:
   ```bash
   ~/.hermes/hermes-agent/venv/bin/python3 -m ensurepip
   ~/.hermes/hermes-agent/venv/bin/python3 -m pip install cashew-brain
   ```

2. **Stale pycache or entry point not registered** — If cashew-brain is installed
   but the plugin still shows NOT installed:
   ```bash
   cd ~/.hermes/plugins/cashew && \
     ~/.hermes/hermes-agent/venv/bin/python3 -m pip install -e .
   hermes gateway restart
   ```

### `Status: not available`

The plugin is available when cashew-brain is importable. Check:
```bash
~/.hermes/hermes-agent/venv/bin/python3 -c "from core.context import ContextRetriever; print('ok')"
```
If this fails, cashew-brain is not installed in the Hermes venv (see above).

### Hermes-agent venv has no `pip`

Hermes-agent creates a minimal venv without pip. Bootstrap it first:

```bash
~/.hermes/hermes-agent/venv/bin/python3 -m ensurepip
~/.hermes/hermes-agent/venv/bin/python3 -m pip install <package>
```

Do **not** run `pip install` from outside the venv targeting the hermes python,
or the package will land in the wrong environment.

### Embedding model download on first use

`cashew-brain` bundles sentence-transformers. The first retrieval operation may
trigger a ~500 MB embedding model download. To avoid this in automated environments:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 hermes ...
```

## Development

```bash
git clone https://github.com/magnus919/hermes-cashew
cd hermes-cashew
pip install -e ".[dev]"   # macOS
python3 -m pip install -e ".[dev]"   # Linux
pytest                      # run the test suite
```

Tests require no network access and mock the embedding model automatically
(`HF_HUB_OFFLINE=1` is set by `tests/conftest.py`).

## Architecture Notes

The plugin uses a dual-path loading strategy to support both `pip install -e .`
(development) and `hermes plugins install` (flat-entry loader):

- **pip / test path**: Python's namespace package mechanism resolves
  `plugins.memory.cashew` to `plugins/memory/cashew/__init__.py` via `sys.path`
- **flat-entry path**: Hermes loads `~/.hermes/plugins/cashew/__init__.py` as
  `_hermes_user_memory.cashew`. The root `__init__.py` detects this context
  and exec's the nested implementation with `sys.modules` patched so relative
  imports resolve correctly

## License

See [LICENSE](./LICENSE).
