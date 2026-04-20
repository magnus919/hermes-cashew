# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin that uses [Cashew](https://github.com/rajkripal/cashew) as the backing store — persistent thought-graph memory with local embeddings, organic decay, and autonomous think cycles.

**Repo:** https://github.com/magnus919/hermes-cashew

**Status:** scaffolding stage — only `CLAUDE.md`, `LICENSE`, and `README.md` exist on disk. The layout, `pyproject.toml`, `plugin.yaml`, and test files described below are the *target* state; create them as work progresses.

---

## Project Purpose

This plugin bridges two systems:

- **Hermes Agent** — the agent runtime that calls into memory providers via a defined ABC
- **Cashew** — a SQLite-backed, locally-embedded knowledge graph (`cashew-brain` on PyPI)

When active, this provider feeds Cashew-retrieved context into Hermes' system prompt and syncs completed turns back into the knowledge graph.

---

## Repository Layout

Hermes imposes a specific plugin path — the non-obvious part:

```
plugins/memory/cashew/
├── __init__.py      # CashewMemoryProvider + register()
├── plugin.yaml      # Hermes plugin metadata
└── README.md        # End-user setup guide
```

Tests live in `tests/`; CI in `.github/workflows/tests.yml`.

---

## Installing Dependencies

> **macOS note:** The system Python is likely `python3` and the package manager `pip3`. Substitute accordingly.

```bash
# Standard
pip install -e ".[dev]"

# macOS / Homebrew Python
pip3 install -e ".[dev]"
```

Dependencies declared in `pyproject.toml`:

- `cashew-brain` — the Cashew library
- `pytest`, `pytest-asyncio` — test runner (dev)

No Hermes Agent package is installed — the plugin is loaded in-process by Hermes from the `plugins/memory/cashew/` directory. The `MemoryProvider` ABC and `MemoryManager` are imported from `agent.*` at Hermes Agent runtime.

---

## Plugin Interface (Hermes Agent ABC)

The plugin lives at `plugins/memory/cashew/__init__.py` and implements `MemoryProvider` from `agent.memory_provider`. Key contracts:

| Method | Notes |
|---|---|
| `name` (property) | Returns `"cashew"` |
| `is_available()` | Check env/config only — **no network or filesystem I/O** |
| `initialize(session_id, **kwargs)` | `kwargs["hermes_home"]` — use this for all storage paths, never `~/.hermes` directly |
| `get_config_schema()` | Declares `cashew_db_path` (non-secret, default `cashew.db`) |
| `save_config(values, hermes_home)` | Writes config JSON under `hermes_home` |
| `get_tool_schemas()` | Exposes `cashew_query` and `cashew_extract` tools |
| `handle_tool_call(name, args)` | Routes tool calls to Cashew's Python API |
| `prefetch(query)` | Returns recalled context string before each LLM call |
| `sync_turn(user, assistant)` | **Must be non-blocking** — run in a daemon thread |
| `on_session_end(messages)` | Flush remaining sync work |
| `shutdown()` | Join any outstanding threads |

### Threading Rule

`sync_turn()` is called on Hermes's hot path. It must return in under 10 ms. The plugin uses:

- A bounded `queue.Queue(maxsize=16)` for pending turns.
- A single **non-daemon** worker thread that drains the queue and calls Cashew's extraction API per turn.
- A sentinel value (not `None` — conflicts with `queue.Queue`'s own close semantics) pushed by `shutdown()` to signal the worker to exit.
- `shutdown()` joins the worker with a bounded timeout (default 30s, configurable via `sync_queue_timeout`); if the timeout expires, a WARNING is logged and the method returns — it never raises into Hermes.
- When Cashew raises during extraction, the worker logs `WARNING` with `exc_info=True` and continues draining. One bad turn does not poison the queue.

Queue overflow policy and exact sentinel pattern are resolved by `/gsd-research-phase 4` before Phase 4 implementation.

This supersedes the rolling-daemon-thread sketch previously documented here. Daemon threads drop work on shutdown and cannot apply backpressure; the queue-worker pattern ships on day one rather than being a breaking-change retrofit.

### Profile Isolation

All file paths must be scoped under `hermes_home`:

```python
# CORRECT
from pathlib import Path
db_path = Path(hermes_home) / "cashew" / "brain.db"

# WRONG — breaks multi-profile setups
db_path = Path("~/.hermes/cashew/brain.db").expanduser()
```

---

## Cashew Integration

The plugin uses Cashew's Python API directly:

```python
from core.context import ContextRetriever
from core.embeddings import load_embeddings
```

Cashew requires ~2 GB RAM and downloads the `all-MiniLM-L6-v2` embedding model (~500 MB) on first use. Tests must not trigger the embedding model download — use mocking or a pre-seeded test fixture.

---

## Testing

Tests live in `tests/` and use `pytest`. They follow the pattern from Hermes Agent's `tests/agent/test_memory_plugin_e2e.py`.

```bash
# All tests
pytest

# Single test
pytest tests/test_cashew_provider.py::test_full_lifecycle -xvs

# macOS
python3 -m pytest
```

### Acceptance Test Pattern

```python
from agent.memory_manager import MemoryManager
from plugins.memory.cashew import CashewMemoryProvider

def test_full_lifecycle(tmp_path):
    provider = CashewMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)
    mgr.initialize_all(session_id="test-1", platform="cli", hermes_home=str(tmp_path))

    result = mgr.handle_tool_call("cashew_query", {"hints": "test"})
    assert result is not None

    mgr.sync_all("user message", "assistant message")
    mgr.on_session_end([])
    mgr.shutdown_all()
```

Tests must:
- Use `tmp_path` (pytest fixture) for all file I/O — never write to `~`
- Mock or stub the Cashew embedding model download
- Test tool schema registration, tool routing, and the full lifecycle
- **Not** require a running Hermes Agent process

---

## GitHub Actions CI

`.github/workflows/tests.yml` runs on every push and pull request to `main`. It tests on Python 3.11 on `ubuntu-latest`. The embedding model download is mocked at the test layer — CI must not download 500 MB artifacts.

---

## Non-obvious ignores

Beyond standard Python ignores, `.gitignore` must also cover `cashew-config.json` — it is written at runtime by `save_config()`, not authored by hand.

See `plugins/memory/cashew/plugin.yaml` for the hook registration list.

---

## Conventions

- Python 3.10+ syntax
- Type hints on all public methods
- `logging` (not `print`) for diagnostics — use `logger = logging.getLogger(__name__)`
- Exceptions from Cashew should be caught and logged as warnings, not surfaced to Hermes as hard failures
- No hardcoded paths — everything through `hermes_home` or `tmp_path` in tests
