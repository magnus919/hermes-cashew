---
name: hermes-cashew
description: Guide for working on the hermes-cashew plugin — a Hermes Agent memory provider backed by Cashew thought-graph memory (SQLite + local embeddings). Use when making changes to the plugin, tests, config, or CI pipeline.
---

# hermes-cashew Development

## Architecture

Thin adapter around upstream [cashew-brain](https://github.com/rajkripal/cashew). The plugin delegates retrieval, schema management, and LLM operations to upstream. Custom code is limited to the Hermes integration surface.

### Key Files

| File | Purpose |
|------|---------|
| `plugins/memory/cashew/__init__.py` | CashewMemoryProvider class + register() entry point |
| `plugins/memory/cashew/config.py` | CashewConfig dataclass, defaults, load/save, schema |
| `plugins/memory/cashew/tools.py` | JSON envelope builders for tool call responses |
| `plugins/memory/cashew/sleep_refactor.py` | Refactored sleep cycle (vectorized, batch-scalable) |
| `plugins/memory/cashew/sleep_cron_script.py` | Cron script entry point for Hermes scheduler |
| `plugins/memory/cashew/verify.py` | Verification helpers |
| `tests/` | pytest suite (236 tests) |
| `__init__.py` (root) | Re-export shim — delegates to plugins.memory.cashew |

### Layout Note

Two load paths must work:
1. **pip install**: `import plugins.memory.cashew` (wheel exposes `plugins/`)
2. **hermes plugins install**: Root `__init__.py` re-exports from nested module

`plugins/` and `plugins/memory/` are PEP 420 namespace packages (no `__init__.py`).

## Dev Commands

```bash
pip install -e ".[dev]"    # one-command setup
pytest                      # run all tests (236)
ruff check .               # lint
mypy --explicit-package-bases plugins/memory/cashew/   # type check (strict)
vulture plugins/memory/cashew/ --min-confidence 80     # dead code
deptry .                   # unused deps
python scripts/check-duplicates.py   # duplicate detection
```

## Key Constraints

- **No ~/.hermes writes.** All paths scoped under `hermes_home`. Tests use `tmp_path`.
- **sync_turn < 10ms.** Bounded queue + non-daemon worker thread.
- **Cashew dependency**: `cashew-brain>=1.1.0,<2.0.0`
- **HF_HUB_OFFLINE=1** in tests. Embedding model must be mocked.
- **Silent degrade** on Cashew failures — log WARNING, return empty, never raise.

## Config Keys (31 keys)

Config is stored in `$HERMES_HOME/cashew.json`. See `plugins/memory/cashew/config.py` for the full CashewConfig dataclass and DEFAULTS.

Notable keys:
- `embedding_model`: Model for vector embeddings (default: `thenlper/gte-large`)
- `llm_aux_role`: Hermes config role for LLM extraction (default: None = heuristic)
- `sleep_schedule`: Cron expression for sleep cycles (default: `0 */12 * * *`)
- `db_path_override`: Override default `cashew/brain.db` path

## Threading Model

- `sync_turn()`: Hot path, returns <10ms. Pushes (user, assistant, session_id) tuple onto bounded `queue.Queue(maxsize=16)`.
- Worker thread: Single non-daemon thread drains queue, calls Cashew extraction API.
- Sentinel: `_SHADOW = object()` sent to queue on shutdown, worker exits gracefully.
- `shutdown()`: Joins worker with 30s timeout, logs WARNING on timeout, never raises.

## Testing

- `tmp_path` fixture for all file I/O
- `HF_HUB_OFFLINE=1` + `fake_embedder` autouse fixture blocks real models
- `CASHEW_*` env vars stripped in conftest
- `agent.memory_provider` ABC stubbed in `sys.modules`
