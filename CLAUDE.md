# CLAUDE.md

Claude Code contributors must follow [AGENTS.md](./AGENTS.md), which is the
single source of truth for this repository's architecture, integration
contracts, development commands, and contribution conventions.

In particular:

- Read [CONTRIBUTING.md](./CONTRIBUTING.md) before making changes.
- Branch from `main`; use one signed, Conventional Commit PR per logical change.
- Keep the root loader dependency-free and preserve the PEP 420 namespace
  package layout described in `AGENTS.md`.
- Scope storage under the supplied `hermes_home`, keep `sync_turn()` under
  10 ms, and silently degrade on Cashew failures.
- Run the full offline test suite before pushing.

Do not duplicate version numbers or detailed runtime behavior here. The package
version is owned by `pyproject.toml`, release history by `CHANGELOG.md`, and
current engineering guidance by `AGENTS.md`.
