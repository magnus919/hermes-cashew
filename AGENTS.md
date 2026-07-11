# AGENTS.md

## Project

Hermes Agent memory provider plugin backed by Cashew (local SQLite knowledge graph with embeddings). Repo root also serves as `$HERMES_HOME/plugins/cashew/` after `hermes plugins install`.

Thin adapter around upstream [cashew-brain](https://github.com/rajkripal/cashew) — the plugin delegates retrieval, schema management, and LLM operations to upstream. Custom code is limited to Hermes integration surface (config, tools, threading).

## Current State

- **LLM integration**: `llm_aux_role` config key references `auxiliary.<role>` in Hermes `config.yaml`. Plugin reads Hermes config, resolves credentials, wires `model_fn` callable to upstream. No API keys in plugin config.
- **Retrieval**: Three-tier (sqlite-vec → BFS → keyword), delegated to upstream `retrieve_recursive_bfs()`.
- **Schema**: Upstream `core.db.ensure_schema()` — no custom migration layer.
- **Threading**: Bounded `queue.Queue(maxsize=16)` + single daemon worker. Shutdown rejects new producers, drains accepted turns ahead of `_SHUTDOWN = object()`, and defers cleanup if the bounded join times out.

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

Releases are triggered by pushing a `v*` tag. The release workflow gates on tests passing, auto-syncs plugin manifest versions, builds the wheel, and publishes to PyPI via trusted publishing.

**Important:** You only bump the version in `pyproject.toml` and `CHANGELOG.md` — CI automatically syncs both `plugin.yaml` files to match before building. See [Plugin Version Manifest](#plugin-version-manifest).

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
- `plugins/memory/cashew/config.py` — CashewConfig dataclass, load/save, 37 compatibility defaults, and a 16-field runtime-backed setup schema.
- `plugins/memory/cashew/tools.py` — JSON envelope builders for tool call responses.
- `plugins/memory/cashew/plugin.yaml` — bundled-loader hook manifest (hooks: [on_session_end, on_pre_compress]). Must match `plugin.yaml` (root).
- `plugin.yaml` (root) — `hermes plugins install` manifest. CI auto-syncs version from `pyproject.toml` on release.
- `tests/` — pytest, MemoryManager stub + fake `agent.memory_provider` ABC in `sys.modules`.
- `.github/workflows/tests.yml` — **do not rename**; filename locked for trusted-publisher config.
- `.github/workflows/release.yml` — **do not rename**; OIDC bound to filename.
- `.planning/` — historical GSD workflow artifacts from v0.1/v0.2 development. Schematic reference only; current work is tracked in GitHub issues.

`plugins/` and `plugins/memory/` are PEP 420 namespace packages — do NOT add `__init__.py`.

## Plugin Version Manifest

Hermes-cashew has **three** version-bearing files, and they must all agree:

| File | What it controls | Who bumps it |
|------|------------------|-------------|
| `pyproject.toml` | PyPI package version (`pip install`, wheel metadata) | **You** — in the release commit |
| `plugin.yaml` (root) | Version shown by `hermes plugins install` / `hermes plugin status` | **CI** — auto-synced in the `build` job of `release.yml` |
| `plugins/memory/cashew/plugin.yaml` | Version in bundled-loader installs (same manifest format) | **CI** — auto-synced (same step as root) |

**The rule:** you only bump `pyproject.toml` when cutting a release. CI reads the version from `pyproject.toml` and patches both `plugin.yaml` files with `sed` before building the wheel and sdist. This happens in the `Sync plugin.yaml versions to pyproject.toml` step of the release workflow, immediately after version-tag validation.

`pyproject.toml` is the source of truth for the current package version;
`CHANGELOG.md` owns release history. Do not add a separate "current version"
claim to prose documentation.

If you ever need to check that all three are in sync locally:
```bash
PYPROJ="$(grep -Po '^version = \"\K[^\"]+' pyproject.toml)"
for f in plugin.yaml plugins/memory/cashew/plugin.yaml; do
  YAML="$(grep -Po '^version: \K.*' "$f")"
  if [ "$PYPROJ" != "$YAML" ]; then
    echo "MISMATCH: $f says $YAML, pyproject.toml says $PYPROJ"
  fi
done
```

## Key Constraints

- **No `~/.hermes` writes.** All paths scope under `hermes_home`. Tests use `tmp_path` fixture.
- **`sync_turn` must return <10 ms.** Bounded queue + daemon worker; shutdown gives already-accepted turns a bounded opportunity to drain.
- **Cashew dependency**: `cashew-brain>=1.1.0,<2.0.0` on PyPI.
- **sqlite-vec** enables vector similarity search. It's a standard dependency
  (not optional) — the plugin requires it. If your platform doesn't support
  sqlite-vec's native extension, the plugin degrades gracefully to keyword + BFS
  search (see logging on startup for the fallback message).
- **`HF_HUB_OFFLINE=1`** set in `conftest.py` before any Cashew import. Embedding model must be mocked in tests.
- **`CASHEW_*` env vars stripped** in `conftest.py` to prevent Hermes session leak into tests.
- **Silent degrade** on all Cashew failures — log WARNING, return empty, never raise into Hermes.

## Developer Commands

```bash
pip install -e ".[dev]"                # install with dev deps
pip install -e ".[dev]"                 # install with dev deps
pip install -e .                        # minimal install
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

## graphify

Graphify output is a local, generated aid and is not committed to the repository.

Rules:
- If `graphify-out/GRAPH_REPORT.md` exists locally, read it before architecture work; if `graphify-out/wiki/index.md` exists, navigate it instead of raw generated files.
- When Graphify is installed, prefer `graphify query`, `graphify path`, or `graphify explain` for cross-module questions.
- After modifying code, run `graphify update .` when Graphify is available. A fresh clone must remain usable without Graphify or generated output.
