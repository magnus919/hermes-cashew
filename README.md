# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph. Get from zero to a working install in under five minutes.

## Prerequisites

- [Hermes Agent](https://github.com/nousresearch/hermes-agent) installed
- `cashew-brain` — installed automatically by `hermes plugins install` or manually:
  ```bash
  ~/.hermes/hermes-agent/venv/bin/python3 -m ensurepip  # bootstrap pip if missing
  ~/.hermes/hermes-agent/venv/bin/python3 -m pip install \
    "cashew-brain @ git+https://github.com/rajkripal/cashew.git@90d1c73"
  ```

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

> **Note:** `hermes memory setup` shows a hardcoded list of providers and does
> not yet include cashew in the interactive picker. Use `hermes config set`
> to activate it directly.

## Configure

Create `~/.hermes/cashew.json` with your settings:

```bash
cat > ~/.hermes/cashew.json << 'EOF'
{
  "cashew_db_path": "cashew/brain.db",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "recall_k": 5,
  "sync_queue_timeout": 30
}
EOF
```

| Key | Default | Description |
|-----|---------|-------------|
| `cashew_db_path` | `cashew/brain.db` | Path to the SQLite DB, relative to `hermes_home` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for retrieval |
| `recall_k` | `5` | Maximum nodes returned per recall query |
| `sync_queue_timeout` | `30` | Seconds to wait for the sync worker to drain on shutdown |

## Verify the Install

```bash
hermes gateway restart   # ensure gateway picks up the new plugin
hermes memory status
```

Expected output shows `Provider: cashew` with `Plugin: installed` and `Status: available`.

## How It Works

`hermes-cashew` provides two LLM-accessible tools:

- **`cashew_query`** — searches the local thought graph for context relevant to
  the current conversation. The agent calls this automatically during `prefetch()`.
- **`cashew_extract`** — explicitly persists a conversation turn into the graph.
  The agent can call this when it judges a turn contains worth-remembering knowledge.

Both tools are registered automatically when Hermes loads the plugin.

## Uninstall

```bash
hermes plugins remove cashew
hermes config set memory.provider built-in   # revert to built-in memory
rm -rf ~/.hermes/cashew   # optional: remove the local graph data
```

## Troubleshooting

### `Plugin: NOT installed` in `hermes memory status`

This has two common causes:

1. **`cashew-brain` not installed in Hermes venv** — `hermes plugins install` does not
   automatically install Python package dependencies into Hermes's venv. Install it manually:
   ```bash
   ~/.hermes/hermes-agent/venv/bin/python3 -m ensurepip
   ~/.hermes/hermes-agent/venv/bin/python3 -m pip install \
     "cashew-brain @ git+https://github.com/rajkripal/cashew.git@90d1c73"
   ```

2. **Stale pycache or entry point not registered** — If `cashew-brain` is installed
   but the plugin still shows NOT installed, the entry point may not be registered:
   ```bash
   cd ~/.hermes/plugins/cashew && \
     ~/.hermes/hermes-agent/venv/bin/python3 -m pip install -e .
   hermes gateway restart
   ```

### `Plugin: installed` but `Status: not available`

Ensure `~/.hermes/cashew.json` exists. The plugin's `is_available()` checks for this
file's presence. Create it with the default config shown above.

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
# Clone the repo first
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
