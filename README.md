# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph. Get from zero to a working install in under five minutes.

## Prerequisites

- Python 3.10 or later
- [Hermes Agent](https://github.com/nousresearch/hermes-agent) installed
- pip

## Install

```bash
pip install hermes-cashew
```

## Register with Hermes

After installing, register the plugin so Hermes discovers it:

```bash
hermes memory setup
```

When prompted to choose a memory provider, select **cashew**. This writes a
`cashew.json` config file to your `hermes_home` directory (default:
`~/.hermes/cashew.json`).

## Verify the Install

Run the built-in smoke test to confirm everything is wired correctly:

```bash
python -m plugins.memory.cashew.verify
```

Expected output: a short report ending with `OK`. Exit code 0 means success.
Non-zero exit code (or output containing `[cashew]` prefixed error lines) indicates
a problem — see Troubleshooting below.

## How It Works

`hermes-cashew` provides two LLM-accessible tools:

- **`cashew_query`** — searches the local thought graph for context relevant to
  the current conversation. The agent calls this automatically during `prefetch()`.
- **`cashew_extract`** — explicitly persists a conversation turn into the graph.
  The agent can call this when it judges a turn contains worth记住了 knowledge.

Both tools are registered automatically when Hermes loads the plugin after
`hermes memory setup` selects `cashew`.

## Configuration

After first run, edit `~/.hermes/cashew.json` (or your configured `hermes_home`):

```json
{
  "cashew_db_path": "cashew/brain.db",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "recall_k": 5,
  "sync_queue_timeout": 30
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `cashew_db_path` | `cashew/brain.db` | Path to the SQLite DB, relative to `hermes_home` |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for retrieval |
| `recall_k` | `5` | Maximum nodes returned per recall query |
| `sync_queue_timeout` | `30` | Seconds to wait for the sync worker to drain on shutdown |

## Uninstall

```bash
pip uninstall hermes-cashew
hermes memory setup   # re-run to unregister, or just forget the cashew choice
rm -rf ~/.hermes/cashew   # optional: remove the local graph data
```

## Troubleshooting

### `python: module plugins.memory.cashew.verify not found`

Run from the repository root, or ensure `pip install -e .` was used during development.
The verify module is only available when the package is installed or the repo root
is on `PYTHONPATH`.

### Embedding model download attempts during install

`hermes-cashew` uses `cashew-brain` which bundles embeddings. The first retrieval
operation may trigger a ~500 MB embedding model download. To avoid this in
automated environments, set `HF_HUB_OFFLINE=1` and use the provided fake-embedder
fixture in tests (see `tests/conftest.py`).

### Hermes does not discover the plugin

Ensure `hermes memory setup` is run **after** `pip install hermes-cashew`.
The entry point is registered during `pip install`; Hermes scans it on setup.

For more on how Hermes discovers memory providers, see the
[Hermes memory-provider plugin developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin).

## Development

```bash
pip install -e ".[dev]"   # install with test dependencies
pytest                      # run the test suite
```

Tests require no network access and mock the embedding model automatically
(`HF_HUB_OFFLINE=1` is set by `tests/conftest.py`).

## License

See [LICENSE](./LICENSE).
