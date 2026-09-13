# Release-candidate readiness for milestone #211

Recorded against `feat/release-candidate-211` at the integration head that
will be published from this checkout. This is a draft evidence record, not a
release approval or a statement that the milestone is ready to close.

## Dependency and integration record

The candidate is based on PR #237 head
`2ecafd053885196e5f004b998e6e52d3ca161cc7`, which is itself stacked on the
immutable Cashew candidate work in PR #236. The selected upstream source is
Cashew PR #136 head `ac090ce75ffd2e97dac257cee9430628c68aa241`, archive
SHA-256 `0777dcb89bde8e0d6103786358c93c0ad3fd4ffb7f8a210194c37b211ae4c28b`.
The source still reports package version `1.2.1`; the commit, archive digest,
and installed `core/sleep.py` digest are the provenance signals.

The reviewed draft ranges were integrated in dependency order, preserving
their signed commits where possible:

| Draft | Integrated head | Scope |
| --- | --- | --- |
| #231 | `21e58c91b59e1d383c14271425ef3ff2c049157b` | bounded read-only integrity audit (#206) |
| #232 | `745e84bbbb336febab99d988bc9132f8a69df0c2` | profile-scoped cron reconciliation (#203) |
| #235 | `cf76cf61d5216334d7727cb139c06a567dea164b` | opt-in consolidation contention benchmark (#205) |
| #234 | `2d47304db262d7db8dc14900a15c2fffc98ae622` | embedding compatibility ownership split (#208) |
| #233 | `e51c521a16e0b21a91921fbcbd4c548b452e23b5` | runtime architecture documentation (#210) |

The provider-split conflict was resolved by retaining the reviewed embedding
compatibility extraction and reapplying the later #232 cron reconciliation
methods. The documentation conflict was resolved in favor of the current
upstream adapter contract; stale prose describing the deleted local sleep
algorithm was not carried forward.

## Evidence collected

- The focused integration boundary suites pass: `137 passed, 2 skipped`.
  The skips require the optional Hermes `cron.jobs` host module, which is not
  installed in this local test environment.
- The full locked test suite passes after the final benchmark update:
  `539 passed, 3 skipped, 2 warnings` in 69.89 seconds. The skips require the
  optional Hermes `cron.jobs` host module; the warnings are the existing Sentry
  SDK deprecation warning.
- `ruff`, `mypy`, `vulture`, duplicate detection, dead-flag detection,
  `deptry`, and the Cashew provenance verifier pass. The verifier confirms the
  lock/archive digest and candidate `core/session.py` and `core/sleep.py`
  digests; uv's `direct_url.json` has empty `archive_info`, so the archive hash
  is installer-lock evidence rather than a post-install rehash claim.
- A clean wheel and sdist build and `scripts/verify-distributions.py` pass.
  The temporary flat-install smoke passes from the extracted sdist, including
  registration, generated script loading, and bounded subprocess completion.
  That smoke intentionally fakes only the model worker and sleep call to keep
  the packaging lane offline; the real upstream candidate subprocess contract
  is exercised separately for both flat and development layouts by
  `tests/test_sleep_cron_script.py`.
- The full multiprocess and lifecycle suites exercise temporary shared brains,
  extraction during sleep, migration overlap, lock/crash bounds, profile
  isolation, cache identity, no-LLM mode, and Hermes lifecycle shutdown. All
  storage is under pytest or `/private/tmp`; no live `~/.hermes` path is used.
- The retargeted #205 benchmark invokes `sleep_adapter.run_sleep_cycle`, wraps
  upstream `core.sleep` phase functions, uses the real sqlite-vec table when
  available, measures a competing writer, and proves the capped candidate run
  commits both directed rows. The gte-large fixture uses cosine `0.92`, which
  is above the candidate's `0.90` cross-link threshold and below its `0.94`
  dedup threshold.

## Remaining gates

This draft must remain open and must not be marked ready until the upstream and
stacked provenance gates are reviewed:

1. Upstream PR #136 must merge, or its exact reviewed commit must be replaced
   by a later tested pin. PR #236 must then pass its immutable provenance and
   archive verification checks, followed by PR #237's thin-adapter review.
2. #206's read-only audit is implemented and tested against synthetic
   corruption. Its `apply_integrity_repairs` entry point deliberately returns a
   structured `repair_unavailable` result because the selected upstream
   version does not expose a stable connection-aware repair contract. This
   candidate does not claim opt-in repair, rollback, or automatic reconstruction
   of merged facts.
3. The optional real Hermes `cron.jobs` host lane must run in CI or a pinned
   Hermes environment before #203/#211 can claim the host scheduler acceptance
   criteria. The local flat smoke is packaging evidence, not a substitute for
   that host module.

No release tag, PyPI publication, live profile migration, or automatic memory
provider re-enablement is part of this candidate.
