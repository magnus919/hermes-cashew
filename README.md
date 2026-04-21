# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph. Get from zero to a working install in under five minutes.

## Prerequisites

- [Hermes Agent](https://github.com/nousresearch/hermes-agent) installed
- `cashew-brain` (installed automatically as a dependency)

## Install

```bash
hermes plugins install magnus919/hermes-cashew
```

This clones the repository to `~/.hermes/plugins/cashew/` and installs
`cashew-brain` into Hermes's virtual environment.

## Register with Hermes

After installing, set cashew as the active memory provider:

```bash
hermes config set memory.provider cashew
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
hermes memory status
```

Expected output shows `Provider: cashew` with `Plugin: installed`. If it shows
`Plugin: NOT installed`, ensure `~/.hermes/cashew.json` exists.

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

Ensure `~/.hermes/cashew.json` exists. The plugin checks for this file in
`is_available()`. Create it with the default config shown above.

### Hermes-agent venv has no `pip`

Hermes-agent manages its own virtual environment. To install plugins manually
into that venv (e.g. if `hermes plugins install` fails):

```bash
uv pip install git+https://github.com/magnus919/hermes-cashew \
  --python ~/.hermes/hermes-agent/venv/bin/python
```

### Embedding model download on first use

`cashew-brain` bundles embeddings. The first retrieval operation may trigger a
~500 MB embedding model download. To avoid this in automated environments, set
`HF_HUB_OFFLINE=1` before running hermes.

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

## License

See [LICENSE](./LICENSE).
