# Contributing to hermes-cashew

Thank you for considering contributing to **hermes-cashew** — a Hermes Agent memory provider plugin backed by the Cashew thought graph. This project is part of the [Hermes Agent](https://hermes-agent.nousresearch.com) ecosystem and follows its conventions where applicable.

**Table of Contents**

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Development Install](#development-install)
  - [Running Tests](#running-tests)
  - [Understanding the Architecture](#understanding-the-architecture)
- [How to Contribute](#how-to-contribute)
  - [Your First Pull Request](#your-first-pull-request)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Features](#suggesting-features)
  - [Pull Requests](#pull-requests)
- [Commit Guidelines](#commit-guidelines)
  - [Conventional Commits](#conventional-commits)
  - [Signing Your Commits (DCO)](#signing-your-commits-dco)
- [Code Style & Quality](#code-style--quality)
- [Testing](#testing)
- [Release Process](#release-process)
- [License](#license)

---

## Code of Conduct

This project is governed by the [Apache 2.0 License](./LICENSE) and the following community norms:

- **Be respectful and constructive** — assume good faith, even when disagreeing
- **Be patient** — maintainers are volunteers with limited time
- **Credit others' work** — if you build on someone else's contribution, acknowledge it
- **No personal attacks** — critique ideas, not people

## Getting Started

### Development Install

hermes-cashew depends on an immutable, hash-pinned upstream `cashew-brain`
source archive, SQLite 3.35 or newer, and Hermes Agent. The SQLite minimum is
required by upstream's legacy v1 migration and applies to the published PyPI
`1.2.1` code as well as the selected source. For development:

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
# Clone the repo
git clone https://github.com/magnus919/hermes-cashew
cd hermes-cashew

# Install the locked development environment
uv sync --frozen --extra dev
```

**If you're developing with a local Hermes Agent install** (the typical use case when testing the plugin as your active memory provider), Hermes runs from its own virtual environment at `~/.hermes/hermes-agent/venv/`. A dev install must be done in *that* venv, and a symlink is required because Hermes inserts `~/.hermes/hermes-agent/` at the front of `sys.path`, making its `plugins/__init__.py` a regular package that blocks PEP 420 namespace resolution.

```bash
# 1. Install into Hermes venv
~/.hermes/hermes-agent/venv/bin/python3 -m pip install -e ".[dev]"

# 2. Symlink so import resolves within Hermes' plugins tree
ln -sf "$PWD/plugins/memory/cashew" \
       ~/.hermes/hermes-agent/plugins/memory/cashew
```

Without the symlink, the entry-point loader fails with `ModuleNotFoundError`. End users who install via `hermes plugins install` are not affected — that path clones to `$HERMES_HOME/plugins/cashew/` and uses directory-based loading, bypassing the entry point entirely.

### Running Tests

```bash
uv run --frozen --extra dev pytest                          # full suite
uv run --frozen --extra dev pytest tests/test_name.py -xvs  # single file
uv run --frozen --extra dev python scripts/check-recursive-symlinks.py plugins/memory/cashew/  # before directory tools
```

Tests require **no network access**. The embedding model is mocked automatically — `conftest.py` sets `HF_HUB_OFFLINE=1` before any Cashew import. CI enforces this with a log-scan step that fails if `Downloading.*MiniLM` appears.

CI exercises the declared Python 3.10 minimum and the current 3.12 runtime from
the frozen lockfile. If a local development symlink points back into
`plugins/memory/cashew/`, the recursive-link checker reports the link and a
remediation; it never deletes a user file.

### Understanding the Architecture

Graphify is an optional local aid. Its generated `graphify-out/` directory is
not committed, so a fresh clone does not require it. If Graphify is installed,
generate or refresh the graph before using it:

```bash
# Generate local output, then view the high-level report
graphify update .
cat graphify-out/GRAPH_REPORT.md

# Query the graph for relationships
graphify query "your question about the codebase"
graphify path "ModuleA" "ModuleB"
```

Key architectural points:
- **Dual layout**: root `__init__.py` (re-export shim) + `plugins/memory/cashew/__init__.py` (real implementation)
- **PEP 420 namespace packages**: `plugins/` and `plugins/memory/` intentionally have **no** `__init__.py`
- **Ownership boundary**: Cashew owns graph algorithms and schema management;
  the adapter owns Hermes lifecycle, profile paths, queueing, cron integration,
  fallback behavior, and compatibility migration guards. Do not remove an
  adapter safeguard until the upstream replacement is merged, pinned where
  required, and covered at the runtime boundary.
- **Threading**: `sync_turn()` enqueues onto `queue.Queue(maxsize=16)` drained by a single **daemon** worker thread. Shutdown rejects new producers, drains accepted turns ahead of `_SHUTDOWN = object()`, and defers cleanup if its bounded join times out.
- **Silent degrade**: all Cashew failures log `WARNING` with `exc_info=True` and return neutral values — never raise into Hermes.

## How to Contribute

### Your First Pull Request

If this is your first contribution here, start with an issue labeled
[`good first issue`](https://github.com/magnus919/hermes-cashew/labels/good%20first%20issue)
or ask on an existing issue whether it is ready for implementation. Comment on
the issue before coding so contributors do not duplicate work.

1. Fork the repository and create a focused branch from the latest `main`.
2. Install the locked development environment with
   `uv sync --frozen --extra dev`.
3. Make one logical change, including a regression test for behavior changes.
4. Run `uv run --frozen --extra dev pytest` locally and correct every new
   failure before opening the PR.
5. Sign every commit for the DCO with `git commit -s` and use a Conventional
   Commit subject.
6. Open a ready-for-review PR, link the issue with `Closes #N` or `Fixes #N`,
   and describe both the change and its verification.

Review happens on the GitHub pull request. CI must pass, review conversations
must be resolved, and a maintainer may ask for focused follow-up commits before
merging. Please keep the branch current with `main` and avoid force-pushing
after review has begun.

### Reporting Bugs

Before filing a bug report:

1. **Search existing issues** — check if it's already reported
2. **Check for known workarounds** in the [README](./README.md#troubleshooting)

If the bug is new, use the [bug report template](./.github/ISSUE_TEMPLATE/bug_report.md). A good report includes:

- Clear steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, Hermes version, hermes-cashew version/commit)
- Full error output or logs

**Security vulnerabilities** should not be reported via public issues. Follow
the private disclosure process in the [security policy](./SECURITY.md).

### Suggesting Features

Use the [feature request template](./.github/ISSUE_TEMPLATE/feature_request.md). Frame your suggestion around the **problem** you're solving, not just your proposed solution. This helps maintainers evaluate whether there's a better approach.

Feature requests that align with the project's goal (being a thin, reliable Hermes → Cashew adapter) are more likely to be accepted. Features that add complexity or drift from the adapter pattern may be deferred.

### Pull Requests

**One PR per logical change.** Don't mix bugfixes with refactoring with features — maintainers may want to accept one and reject another.

#### Before You Code

1. **Discuss first** for non-trivial changes (more than ~100 lines, architecture changes, new features). Open an issue or discussion to get maintainer feedback before implementing.
2. **Check CONTRIBUTING.md** — you're reading it. Good.
3. **Fork the repo** (if you don't have write access).
4. **Branch from `main`** using a descriptive name:

| Pattern | Example |
|---------|---------|
| `fix/description` | `fix/vec-search-rowid-mismatch` |
| `feat/description` | `feat/add-ollama-embedding-support` |
| `refactor/description` | `refactor/extract-db-migration-module` |
| `docs/description` | `docs/api-reference-typo` |
| `ci/description` | `ci/pypi-version-auto-detection` |

5. **Make focused commits** — one logical change per commit.

#### When Submitting

- **Include tests** for new code. If you're fixing a bug, add a test that fails before your fix and passes after.
- **Run the full test suite** locally before pushing — don't rely on CI to catch basic failures.
- **Update documentation** if your change affects user-facing behavior (README, config keys, tool schemas).
- **Reference related issues** in the PR body using `Closes #N` or `Fixes #N`.

#### After Submitting

- **Monitor CI** — if it fails, fix it promptly. Don't leave a broken PR sitting.
- **Respond to review feedback** — address each comment, even if just to acknowledge.
- **Don't force-push** after a reviewer has looked at your PR. Add fixup commits instead. Force-pushing destroys the review history. Maintainers can squash on merge.

## Commit Guidelines

### Conventional Commits

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
type(scope): short description

Longer body explaining the change, wrapped at 72 characters.
Reference related issues in the body.

Signed-off-by: Your Name <email>
```

**Types:**

| Type | When to Use |
|------|-------------|
| `feat` | A new feature or capability |
| `fix` | A bug fix |
| `refactor` | Code restructuring without behavior change |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `ci` | CI/CD changes |
| `chore` | Maintenance, dependencies, tooling |
| `perf` | Performance improvement |

**Scope** is optional but encouraged — use the module or area affected (e.g., `fix(vec-search):`, `feat(config):`, `test(migration):`).

### Signing Your Commits (DCO)

This project requires the **Developer Certificate of Origin (DCO)** — a lightweight certification that you have the right to contribute the code under the project's license.

To certify, add `Signed-off-by: Your Name <email>` to every commit. Use the `-s` flag:

```bash
git commit -s -m "fix: correct vec_embeddings rowid lookup"

# The -s flag automatically appends:
# Signed-off-by: Your Name <your@email.com>
```

By signing off, you certify that:

> The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; OR I received the contribution under an appropriate open source license and I have the right to submit that contribution with modifications under the same license.
>
> — [Developer Certificate of Origin v1.1](https://developercertificate.org/)

## Code Style & Quality

- **Python**: Follow [PEP 8](https://peps.python.org/pep-0008/) with a line length of 88 (Black-compatible)
- **Type hints**: Use them for all public APIs and new functions
- **Imports**: Standard library → third-party → local, separated by blank lines
- **No network in tests**: All tests run in offline mode. Embedding models must be mocked.
- **Logging, not print**: Use `logging.getLogger(__name__)` — Hermes manages log handlers

CI enforces Ruff, mypy, Vulture, duplicate-code, dead-feature-flag, and unused-
dependency checks in addition to the full pytest suite.

## Testing

This project uses **pytest** with `pytest-asyncio` and `pytest-mock`.

- **Test location**: `tests/` directory, one file per module or feature area
- **Fixtures** live in `tests/conftest.py`
- **No network access**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` are set before any Cashew import
- **Temp paths**: All tests use `tmp_path` fixture — no writes to `~/.hermes` or other real paths
- **Test patterns**:
  - `test_*_e2e.py` — end-to-end lifecycle tests
  - `test_*_quality*.py` — retrieval quality metrics
  - `test_*_migration.py` — schema migration tests
  - `test_*_config*` — config round-trip and override tests
  - `test_handle_tool_call.py` — tool schema and dispatch tests
  - `test_recall.py`, `test_retrieval.py` — recall and retrieval path tests
  - `test_memory_manager_e2e.py` — full provider lifecycle with MemoryManager stub

## Release Process

Maintainers handle releases. The process is:

The current direct Cashew source pin is an explicit publication hold. The
release workflow fails before tests or build until a tested upstream release
replaces that direct URL; do not create a release tag while the hold is active.

1. Version is bumped in `pyproject.toml` and `CHANGELOG.md` is updated
2. A tag is pushed (`vX.Y.Z`)
3. For non-RC tags, CI publishes to PyPI via OIDC trusted publishing
4. `v*-rc*` tags run tests and build uploadable artifacts, but skip publication; non-RC tags publish to PyPI directly. There is no TestPyPI publication job.

## License

By contributing to hermes-cashew, you agree that your contributions will be licensed under the [Apache 2.0 License](./LICENSE).
