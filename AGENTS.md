# AGENTS.md

## Project

Hermes Agent memory provider plugin backed by Cashew (local SQLite knowledge graph with embeddings). Repo root also serves as `$HERMES_HOME/plugins/cashew/` after `hermes plugins install`.

Thin adapter around upstream [cashew-brain](https://github.com/rajkripal/cashew) — the plugin delegates retrieval, schema management, and LLM operations to upstream. Custom code is limited to Hermes integration surface (config, tools, threading).

## Current State (v0.4.0)

- **LLM integration**: `llm_aux_role` config key references `auxiliary.<role>` in Hermes `config.yaml`. Plugin reads Hermes config, resolves credentials, wires `model_fn` callable to upstream. No API keys in plugin config.
- **Retrieval**: Three-tier (sqlite-vec → BFS → keyword), delegated to upstream `retrieve_recursive_bfs()`.
- **Schema**: Upstream `core.db.ensure_schema()` — no custom migration layer.
- **Threading**: Bounded `queue.Queue(maxsize=16)` + single non-daemon worker. Sentinel is `_SHADOW = object()`.
- **Open issues**: 0 remaining. #15 (privacy / exclude_tags), #35 (spec compliance), #36 (on_pre_compress), and #38 (sleep cycle redesign) all resolved.

## Project Conventions

### Open Source

This is a public open source project. All contributions must follow:

- **CONTRIBUTING.md** — DCO sign-off (`git commit -s`), Conventional Commits, PR process
- **Branch from `main`**, open a PR, get review, merge. No direct pushes to main for features.
- **One PR per logical change.** Don't mix bugfixes with refactoring with features.
- **Run tests locally** before pushing (`pytest` — must be green or pre-existing failures only).
- **Issue templates** for bug reports and feature requests.

### Pull Requests

```bash
git checkout -b fix/description   # or feat/, refactor/, docs/, ci/
git add <files>
git commit -s -m "type(scope): description"
git push -u origin HEAD
# Open PR via GitHub UI or gh CLI
```

### Release Process

Releases are triggered by pushing a `v*` tag. The release workflow gates on tests passing, builds the wheel, and publishes to PyPI via trusted publishing.

```bash
# After PR is merged to main:
git checkout main && git pull
# Update version in pyproject.toml and CHANGELOG.md
git commit -s -m "chore: bump to vX.Y.Z"
git tag vX.Y.Z && git push origin main --tags
# Create GitHub Release with notes:
gh release create vX.Y.Z --title "vX.Y.Z — Title" --notes "..."
```

The `auxiliary.memory` convention is designed for any Hermes memory provider. When adding features, prefer delegation to upstream over reimplementation.

## Repo Layout

- `__init__.py` (root) — thin re-export shim; flat-entry loader path. **Do not import Cashew dependencies here.**
- `plugins/memory/cashew/__init__.py` — provider implementation (CashewMemoryProvider + register).
- `plugins/memory/cashew/config.py` — CashewConfig dataclass, DEFAULTS, load/save, schema (31 keys).
- `plugins/memory/cashew/tools.py` — JSON envelope builders for tool call responses.
- `plugins/memory/cashew/plugin.yaml` — bundled-loader hook manifest (hooks: [on_session_end]).
- `plugin.yaml` (root) — `hermes plugins install` manifest, must stay in sync with the above.
- `tests/` — pytest, MemoryManager stub + fake `agent.memory_provider` ABC in `sys.modules`.
- `.github/workflows/tests.yml` — **do not rename**; filename locked for trusted-publisher config.
- `.github/workflows/release.yml` — **do not rename**; OIDC bound to filename.
- `.planning/` — historical GSD workflow artifacts from v0.1/v0.2 development. Schematic reference only; current work is tracked in GitHub issues.

`plugins/` and `plugins/memory/` are PEP 420 namespace packages — do NOT add `__init__.py`.

## Key Constraints

- **No `~/.hermes` writes.** All paths scope under `hermes_home`. Tests use `tmp_path` fixture.
- **`sync_turn` must return <10 ms.** Bounded queue + non-daemon worker.
- **Cashew dependency**: `cashew-brain>=1.1.0,<2.0.0` on PyPI.
- **`HF_HUB_OFFLINE=1`** set in `conftest.py` before any Cashew import. Embedding model must be mocked in tests.
- **`CASHEW_*` env vars stripped** in `conftest.py` to prevent Hermes session leak into tests.
- **Silent degrade** on all Cashew failures — log WARNING, return empty, never raise into Hermes.

## Developer Commands

```bash
pip install -e ".[dev]"                # install with dev deps
pytest                                 # full suite
pytest tests/test_name.py -xvs          # single file
python3 -m pytest                       # macOS fallback
```

### Dev Install Quirks

Hermes runs from `~/.hermes/hermes-agent/venv/`. For dev installs in that venv, a symlink is needed:

```bash
~/.hermes/hermes-agent/venv/bin/python3 -m pip install -e ".[dev]"
ln -sf "$PWD/plugins/memory/cashew" ~/.hermes/hermes-agent/plugins/memory/cashew
```

End users who install via `hermes plugins install` are not affected — directory-based loading bypasses the entry point.

CI runs `pytest -xvs` with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`. Log-scan fails if `Downloading.*MiniLM` appears.

## LLM Integration

The plugin supports optional LLM-powered extraction via Hermes' auxiliary model infrastructure:

- Config key: `llm_aux_role` in `cashew.json` (default: `None` → heuristic-only)
- User config: `auxiliary.memory` section in Hermes `config.yaml`
- Plugin reads Hermes config, resolves API key from config or well-known env var, constructs OpenAI-compatible callable
- Both `_drain_once` (sync worker) and `cashew_extract` (tool) pass the callable to upstream

See README.md for setup guide, CHANGELOG.md for release history, and CONTRIBUTING.md for contribution guidelines.
