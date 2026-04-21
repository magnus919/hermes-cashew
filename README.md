# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph. Get from zero to a working install in under five minutes.

## Prerequisites

- Python 3.10 or later (`python3` on Linux)
- [Hermes Agent](https://github.com/nousresearch/hermes-agent) installed
- pip (`pip3` on Linux)

## Install

> **Note:** This package is not yet published to PyPI. Install it directly from
> the git repository.

```bash
# macOS
pip install git+https://github.com/magnus919/hermes-cashew

# Debian/Ubuntu (PEP 668 externally-managed-environment error)
# Recommended: use pipx to install as an application
sudo apt install pipx && pipx install git+https://github.com/magnus919/hermes-cashew

# Alternative: create a virtual environment first
python3 -m venv ~/.venv/hermes-cashew
~/.venv/hermes-cashew/bin/pip install git+https://github.com/magnus919/hermes-cashew
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
python -m plugins.memory.cashew.verify   # macOS
python3 -m plugins.memory.cashew.verify # Linux
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
pip uninstall hermes-cashew   # macOS
python3 -m pip uninstall hermes-cashew   # Linux
hermes memory setup   # re-run to unregister, or just forget the cashew choice
rm -rf ~/.hermes/cashew   # optional: remove the local graph data
```

If installed via pipx, use `pipx uninstall hermes-cashew` instead.

## Troubleshooting

### `error: externally-managed-environment` (PEP 668)

On Debian/Ubuntu and other modern Linux distros, pip is blocked from installing
packages system-wide. Use `pipx` (recommended for applications) or a virtual
environment:

```bash
# Recommended: pipx handles isolation for CLI applications
sudo apt install pipx && pipx install hermes-cashew

# Alternative: manual virtual environment
python3 -m venv ~/.venv/hermes-cashew
~/.venv/hermes-cashew/bin/pip install hermes-cashew
```

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

Ensure `hermes memory setup` is run **after** installing hermes-cashew.
The entry point is registered during `pip install`; Hermes scans it on setup.

For more on how Hermes discovers memory providers, see the
[Hermes memory-provider plugin developer guide](https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin).

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
