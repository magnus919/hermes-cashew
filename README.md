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

| Key | Default | Description |
|-----|---------|-------------|
| `cashew_db_path` | `cashew/brain.db` | Path to SQLite DB, relative to `hermes_home` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for retrieval |
| `recall_k` | `5` | Max nodes returned per recall query |
| `sync_queue_timeout` | `30` | Seconds to wait for sync worker drain on shutdown |
| `user_domain` | `cli/user` | Domain label for user messages |
| `ai_domain` | `cli/ai` | Domain label for AI messages |
| `vec_dimension` | `384` | Embedding dimension (fixed for v0.2.0) |
| `gc_interval_turns` | `100` | GC run frequency |
| `gc_delete_probability` | `0.01` | Node deletion probability per GC |
| `enable_query_decomposition` | `true` | Enable query decomposition |
| `max_tokens_per_node` | `512` | Token limit per context node |
| `feature_bfs_retrieval` | `true` | Enable BFS graph traversal |
| `feature_semantic_search` | `true` | Enable sqlite-vec semantic search |
| `feature_context_summarization` | `false` | Enable context summarization |
| `max_depth` | `3` | Max BFS traversal depth |
| `similarity_threshold` | `0.7` | Minimum similarity score |
| `max_nodes_per_query` | `20` | Maximum nodes per query |

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
- **Sleep synthesis** — Runs at session end via `on_session_end()` when
  `sleep_cycles: true`. Nine-phase consolidation pipeline: cross-linking,
  dedup, garbage collection, permanence evaluation, core memory promotion,
  and LLM-powered dream generation. Processes 7K nodes in ~4 seconds
  (vs hours in the upstream O(N²) implementation). Work-capped at 2,000
  nodes per cycle — converges gradually over multiple sessions.
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
