# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph with semantic search and automatic context recall. Get from zero to
a working install in under five minutes.

**v0.2.0** brings semantic search via sqlite-vec, recursive graph traversal,
expanded configuration (31 keys with sane defaults), and zero-config startup.

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

hermes-cashew works out of the box — all 31 configuration keys have sane
defaults. Create `~/.hermes/cashew.json` only if you want to override them:

```bash
cat > ~/.hermes/cashew.json << 'EOF'
{
  "cashew_db_path": "cashew/brain.db",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "recall_k": 5,
  "user_domain": "cli/user",
  "ai_domain": "cli/ai",
  "sync_queue_timeout": 30,
  "vec_dimension": 384,
  "gc_interval_turns": 100,
  "gc_delete_probability": 0.01,
  "enable_query_decomposition": true,
  "max_tokens_per_node": 512,
  "feature_bfs_retrieval": true,
  "feature_semantic_search": true,
  "feature_context_summarization": false,
  "max_depth": 3,
  "similarity_threshold": 0.7,
  "max_nodes_per_query": 20
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
