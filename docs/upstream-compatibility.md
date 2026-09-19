# Upstream compatibility and workaround ownership

This document records the compatibility baseline for hermes-cashew and the
retirement boundary for safeguards that exist because Cashew or Hermes did not
provide a tested contract. It is an inventory and decision record for issue
[#187](https://github.com/magnus919/hermes-cashew/issues/187). Issue
[#188](https://github.com/magnus919/hermes-cashew/issues/188) selected and
tested an immutable upstream source baseline while retaining the workarounds
owned by later issues.

## Baseline and release policy

Audit date: **2026-09-12 UTC**.

The historical #187 inventory used these references:

| Component | Exact reference | Availability and provenance |
| --- | --- | --- |
| hermes-cashew | `b14d7b013ac0e95314f78cca6cacfef202a30654` | This repository's `origin/main` at the issue audit baseline. |
| cashew-brain | `1.2.1`, PyPI | The released artifact inspected during the inventory. It does not contain the schema-contention fix selected by #188. |
| Hermes Agent | `990473a79c6b0396b0a648fdd85ee8f7a5c267d3` | The [immutable upstream commit](https://github.com/NousResearch/hermes-agent/commit/990473a79c6b0396b0a648fdd85ee8f7a5c267d3) statically inspected during #187. |

Issue #188 tested this candidate baseline:

| Component | Exact reference | Availability and provenance |
| --- | --- | --- |
| hermes-cashew | `cc1e31c66b1f0cf72079d91497e21c7347c47c99` plus the issue #188 dependency, test, workflow, and documentation change | This was `origin/main` when the final candidate was refreshed and tested. It includes the merged #199 real-Hermes integration lane. |
| cashew-brain | `fcb4919ac37144bfbeb822eaafc668a4bdceb791` | Immutable [composite fork archive](https://github.com/magnus919/true/archive/fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz), locked with archive SHA-256 `23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3`. Tree `29d97fb93c7998c97be0da8f19b90c6523f9d1e9` combines PR 136 (`ac090ce75ffd2e97dac257cee9430628c68aa241`) and PR 137 (`cb940f34c15460b87831748b2e702334c1c5fbd0`). The installed `core/session.py` SHA-256 is `0ce60cc63adf4fb7136581aee722bb10e9a344e556b6fb98f4d46855d53c36cd`. |
| Hermes Agent loader/lifecycle evidence | `990473a79c6b0396b0a648fdd85ee8f7a5c267d3` | The #199 lane merged in wrapper commit `cc1e31c66b1f0cf72079d91497e21c7347c47c99`. Its flat and development loader, auxiliary routing, lifecycle, and cron contracts also passed in a fresh environment containing the exact #188 source pin. The embedding model and external client were deterministic fakes; Hermes, SQLite 3.47.1, and the Cashew package boundary were real. This is bounded compatibility evidence, not a published Hermes minimum-version promise or native embedding-crash proof. |

The selected source is a composite fork at `fcb4919ac37144bfbeb822eaafc668a4bdceb791`,
tree `29d97fb93c7998c97be0da8f19b90c6523f9d1e9`, combining PR 136
(`ac090ce75ffd2e97dac257cee9430628c68aa241`) and PR 137
(`cb940f34c15460b87831748b2e702334c1c5fbd0`). Production installs use the
exact source archive and digest below rather than floating on Cashew `main`.
The composite contains the commit proposed as PR 136 before that PR merged
upstream, plus the still-noncanonical PR 137 commit. It is not equivalent to
the current tip of the merged PR 136 branch. A later dependency change may
replace it with a canonical release only after independently verifying the same
contracts:

```text
https://github.com/magnus919/true/archive/fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz
sha256=23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3
```

The source archive builds normally, but its metadata still says `1.2.1`, the
same version already published on PyPI. Version metadata is therefore not
valid provenance. Install verification checks both `direct_url.json` and the
installed `core/session.py` digest.

A public hermes-cashew wheel cannot declare this direct URL as a normal PyPI
dependency: PyPI rejects `Requires-Dist` entries using direct URLs
([setuptools dependency guidance](https://setuptools.pypa.io/en/stable/userguide/dependency_management.html#direct-url-dependencies)).
The release workflow fails before tests or build while the source pin is
present. Existing published hermes-cashew packages remain unchanged and still
resolve their released Cashew range; `pip install hermes-cashew` from PyPI does
not obtain this source baseline.

Hermes also rejects direct URLs in plugin manifests. GitHub and local wheel
installs can resolve the dependency metadata, but flat plugin users must run
the explicit `uv pip install --python … --reinstall` command in the main
README and verify provenance. The same command is required after `hermes
plugins update cashew`, `hermes update`, or a Hermes environment rebuild. This
is an approved interim delivery constraint, not an automatic-healing claim.

The selected source includes the schema fast path and the PR 136 pending-batch
fix. It still has no supported device-injection contract, and several wrapper
calls still target private upstream functions. The adapter delegates
consolidation to that selected upstream pipeline.

The selected source requires SQLite 3.35 or newer for legacy v1 migrations
because upstream uses `ALTER TABLE ... DROP COLUMN`. The published PyPI
`1.2.1` artifact contains the same migration and the same undeclared SQLite
minimum; #188 exposed this compatibility requirement but did not introduce it.
The provenance verifier prints SQLite's version and source ID and rejects older
runtimes before migration. This is separate from the WAL-reset safety versions
tracked by #191: satisfying 3.35 alone does not establish WAL safety.

## Ownership boundary

Hermes integration belongs here: provider lifecycle, profile-scoped paths,
tool envelopes, queueing, cron registration, auxiliary-model resolution,
loader compatibility, and failure isolation. Cashew engine algorithms, schema
ownership, embedding service behavior, and consolidation semantics belong
upstream once a released and tested contract exists. Every retirement below
requires a regression test at the real boundary and a rollback path before the
local safeguard changes.

The module split makes that ownership concrete:

| Module | Owner and reason |
| --- | --- |
| [`sleep_adapter.py`](../plugins/memory/cashew/sleep_adapter.py) | Hermes coordination: profile admission, maintenance lease, journal policy, and embedding-worker handoff around upstream `core.sleep`. |
| [`embedding_compat.py`](../plugins/memory/cashew/embedding_compat.py) | Pinned compatibility boundary: generation-scoped service/cache facades and the explicit upstream singleton assignments listed below. |
| [`config.py`](../plugins/memory/cashew/config.py) and [`tools.py`](../plugins/memory/cashew/tools.py) | Hermes configuration, auxiliary-role resolution, and tool envelopes. |
| [`__init__.py`](../plugins/memory/cashew/__init__.py) | Provider lifecycle and delegation orchestration; retrieval, schema, extraction, and consolidation algorithms remain upstream-owned. |

## Private upstream seam inventory

A repository-wide census of private upstream attribute access and private
imports found the following retained seams. `embedding_compat.py` exports
`UPSTREAM_COMPATIBILITY_SHIMS`, covering
`core.config.config.embedding_model`,
`core.embedding_service._default_service`, and
`core.embedding_service._KNOWN_DIMS[model]`. The pre-compress hook also calls
`core.session._create_node` and `core.session._set_node_tags` because the pinned
source has no public write-and-tag API. The matrix below links their provenance,
regression coverage, and retirement conditions.

The packaged, opt-in [`sleep_benchmark.py`](../plugins/memory/cashew/sleep_benchmark.py)
temporarily instruments upstream `_find_pairs`, `_batch_cross_links`,
`_run_dedup`, `_compute_metrics`, `_garbage_collect`, `_evaluate_permanence`,
`_promote_core_memories`, and `_embed_orphans` to report phase timing. Those
private seams are diagnostic-only: the runtime
[`sleep_adapter.py`](../plugins/memory/cashew/sleep_adapter.py) calls the public
`core.sleep.run_sleep_cycle` API. The benchmark contract is covered by
[`test_sleep_benchmark.py`](../tests/test_sleep_benchmark.py); retire or revise
its instrumentation when a pinned upstream change renames those phases or
provides a public timing hook.

## Workaround and compatibility inventory

The status is deliberately conservative. “Retain” means the current behavior
stays in place while a replacement is proved. “Replace” means an upstream
replacement is the intended destination, not that this issue implements it.
“Remove” is reserved for duplicate behavior with a verified owner.

Current ownership: consolidation runs through
the pinned upstream `core.sleep` pipeline via
[sleep_adapter.py](../plugins/memory/cashew/sleep_adapter.py). A full tracked-file
and import census found no supported caller of the historical `sleep_refactor`
path, so the transition shim was removed. Read-only integrity inspection uses
explicitly confirmed repair delegation.
Rows below are updated where that changes their status; historical evidence
references are unchanged.

| Surface and evidence in `b14d7b0` | Original symptom / current responsibility | Upstream replacement and status | Decision, retirement prerequisite, and required regression |
| --- | --- | --- | --- |
| [`embedding_compat.py`](../plugins/memory/cashew/embedding_compat.py) and provider binding | The released Cashew 1.2.1 artifact exposes model-name and dimension globals but no supported provider device injection. The adapter must select the configured model, force safe CPU fallback, and prevent one Hermes profile/session from inheriting another's backend state. The historical device/host-crash cause is not recovered; the protection is observable in the compatibility module and its tests. | The selected source still constructs `SentenceTransformer` without a device parameter; commit `394d6cffe308ba106df8db750248f1ff272bb814` only guards daemon dimension mismatch. No replacement contract exists. | **Retain.** The compatibility facade is now isolated by integration responsibility, while the three upstream shim assignments stay in the provider binding path. Replace only after a released upstream model/device contract proves CPU fallback, model dimension, daemon mismatch, and two-profile isolation in tests. Preserve rollback to the patched path. |
| Embedding-dimension and `vec_embeddings` migration/create in [`__init__.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L821-L1080) | Older brains can contain the rowid-shaped vec schema or vectors from a previous model dimension. The adapter detects/rebuilds the incompatible virtual table, derives the configured dimension, and falls back when sqlite-vec is unavailable. The original host-crash history is unknown; the old schema failure is documented in the code and commit `43eea9e`. | The selected source creates the current `node_id` vec table and contains dimension/daemon guards (`233b75684025656cde46bf4a2af9bca416ef5145`, `394d6cffe308ba106df8db750248f1ff272bb814`). It does not prove migration of every legacy adapter-created table. | **Retain, then replace migration ownership.** #188 must test old/new schema, mixed dimensions, persistence, and rollback before local migration is removed. The provider's profile path guard and neutral fallback remain Hermes responsibilities. |
| Local consolidation engine [`sleep_refactor.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/sleep_refactor.py) (candidate discovery, cross-links, connected-component dedup, rewiring, GC, permanence, orphan embedding) | The adapter needed a scheduled, bounded sleep path with edge caps, source filtering, advisory locking, and profile-scoped persistence. The history records the implementation and fixes but does not establish a single original upstream failure for every phase. | An earlier candidate added the shorter upstream `core/sleep.py` and later permanence, timestamp, embedding, and vector fixes (`c7ddfc6`, `550ae97`, `3fe1436`, `15a28a9`). Its capped pending-batch defect was subsequently fixed by upstream PR 136 (`ac090ce7`), which is included in the currently selected composite `fcb4919`. | **Removed.** [sleep_adapter.py](../plugins/memory/cashew/sleep_adapter.py) remains the Hermes coordination boundary and delegates to `core.sleep.run_sleep_cycle`. The repository-wide census found only the transition module, its compatibility test, documentation, and stale tooling entries; supported cron, benchmark, package, and flat-loader paths already import `sleep_adapter`. Consolidation safety contracts continue to regress the repaired upstream behavior. |
| Keyword fallback [`_keyword_search`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1900-L1950) and neutral retrieval fallback around [`retrieve_recursive_bfs`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1580-L1810) | Recall must remain usable when sqlite-vec, an embedding model, or an upstream retrieval call is unavailable. The fallback is an availability safeguard, not a second ranking engine. | Upstream retrieval remains the engine owner; upstream does not provide this Hermes failure-isolation contract. | **Retain as adapter responsibility.** #202/#204 must prove truthful status, context correctness, bounded work, and deterministic fallback before any simplification. |
| Hermes-specific result enrichment and access accounting [`_enrich_results` / `_update_access_metrics`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1081-L1157) | Cashew retrieval returns engine-shaped rows; Hermes needs stable context formatting and profile/session access updates. | No upstream replacement is implied: this is integration behavior. | **Retain.** Keep tests at the provider boundary; do not move it into Cashew engine code. |
| Direct persistence helpers [`_create_node` and `_set_node_tags`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1388-L1435) | The pre-compress hook needs to persist forest-level insight nodes and tags. These underscore-prefixed Cashew functions are private and can change without compatibility guarantees. | Upstream `core.session` owns the helpers; no public persistence API is present in the tested release. | **Replace when available; retain now.** Require an upstream public write API, tag/embedding regression coverage, and rollback before changing this path. |
| `core.context.ContextRetriever` initialization and availability probe ([`__init__.py#L49-L55`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L49-L55), [`#L500-L535`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L500-L535)) | Provider startup must degrade cleanly when Cashew is absent while creating the upstream context retriever only after profile-scoped paths exist. The constructor and availability behavior are upstream-owned. | `core.context.ContextRetriever` is present in the released package, but no Hermes-specific lifecycle adapter is supplied upstream. | **Retain and pin.** #199 must run the real provider initialization/availability contract against the exact Hermes and Cashew refs; replacement requires an upstream lifecycle API and failure-path regression. |
| Direct session/retrieval/embedding/backup calls: `core.session.end_session` and `think_cycle` ([`__init__.py#L1159-L1240`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1159-L1240)), `core.retrieval.retrieve_recursive_bfs` ([`#L1580-L1810`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1580-L1810)), `core.embeddings.embed_nodes` ([`#L1388-L1435`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1388-L1435)), `core.db.ensure_schema` and `core.backup.create_backup` ([`#L821-L980`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L821-L980)) | These calls cross the package boundary directly. The underscored helpers above are private; the others are public-looking functions whose signatures and result shapes are still upstream-owned. | `cashew-brain` owns these engine operations. The selected source adds schema fast-path and resource/dimension fixes, but no released stable adapter API. | **Retain and pin.** Exercise schema, writes, backups, locking, commit uncertainty, and result shapes against the exact dependency. Replace private calls when upstream publishes a stable contract; do not reimplement engine algorithms here. |
| `core.embedding_service.resolve_embedding_dim` ([`__init__.py#L947-L950`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L947-L950), [`#L1061-L1070`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1061-L1070)) and `core.permanence.promote_permanent_nodes` ([`sleep_refactor.py#L447-L460`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/sleep_refactor.py#L447-L460)) | The adapter needed the configured vector dimension before creating sqlite-vec and delegated permanence promotion during the local sleep pipeline. Both functions are upstream implementation details at the integration boundary. | The selected source improves dimension and permanence safety (`394d6cf`, `550ae97`), but those functions still lack a stable adapter API. | **Replaced with the engine.** Upstream `core.sleep` now calls `promote_permanent_nodes` and resolves dimensions internally; the adapter no longer invokes either function. Replace again only when upstream publishes a stable public API for them. |
| Provider lock and profile isolation (`fcntl`, temporary backup, and Hermes-home-derived paths) in [`__init__.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L930-L980) and [`config.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/config.py#L502-L570) | Concurrent provider processes must not corrupt a shared brain, and user-configured paths must remain below the active Hermes profile. This is the adapter's trust and data-safety boundary. | Cashew schema and engine locks do not establish Hermes profile isolation. | **Retain.** #189/#191/#197 must prove multi-process contention, lock release, crash recovery, and path traversal rejection. Never retire these protections as “upstream cleanup.” |
| Sleep cron registration/deduplication and standalone script import fallback ([`__init__.py#L87-L98`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L87-L98), [`#L355-L376`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L355-L376), [`sleep_cron_script.py#L64-L86`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/sleep_cron_script.py#L64-L86)) | Hermes cron jobs persist across restarts and the plugin must avoid duplicate sleep jobs while working from both bundled and flat installs. The script has two import paths because the loader and installed package expose different module layouts. | Hermes `cron.jobs` and the shared directory loader are the platform contracts; Cashew does not own them. | **Retain pending contract verification.** #186/#199/#203 must prove one job per profile, correct `HERMES_HOME`, no-agent execution, both loaders, and shutdown/restart behavior before dedup or import fallback is simplified. |
| Dual loader and manifest contract: root [`__init__.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/__init__.py#L1-L90), nested provider [`register`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L2160-L2172), and root/nested [`plugin.yaml`](https://github.com/magnus919/hermes-cashew/tree/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew) manifests | Hermes bundled discovery and `hermes plugins install` do not use the same import path. The root shim must stay dependency-light and namespace packages must not be shadowed. | Hermes plugin loader supports directory modules and entry points; no single-path guarantee is assumed. | **Retain.** Test both loader paths against the exact Hermes revision; remove duplication only after a platform contract and rollback are documented. |
| Standalone config schema and auxiliary LLM resolution [`config.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/config.py#L24-L655) | Hermes owns `config.yaml` and auxiliary roles; the plugin persists its own 17-field `cashew.json`, translates environment overrides, constrains paths, and resolves the configured `auxiliary.<role>` callable. This duplication exists because the provider has setup fields and must also run from a cron subprocess. | The current Hermes host exposes the supported [`agent.auxiliary_client.get_text_auxiliary_client(task)`](https://github.com/NousResearch/hermes-agent/blob/990473a79c6b0396b0a648fdd85ee8f7a5c267d3/agent/auxiliary_client.py#L4911-L4917) resolver. The plugin baseline does not use it; it constructs its own OpenAI-compatible callable. The `MemoryProvider` contract supplies setup schema/save hooks, and the current revision adds optional methods (`unavailable_reason`, `recall_status`, `handle_tool_call(**kwargs)`, `on_delegation`, and `backup_paths`) that are not yet implemented by this baseline. | **Retain integration ownership until #200, then evaluate replacement.** #198/#200/#199 must verify callable shape, task routing, credentials, retry/failure semantics, atomic persistence, and backward-compatible migration before adopting the host resolver or consolidating config. |

## Platform scope and follow-on ownership

The baseline is intended for Python `>=3.10` on POSIX hosts (Linux and macOS)
with SQLite `>=3.35`, the standard `sqlite-vec` dependency, and a locally
available or otherwise configured SentenceTransformer model. The Python minor
version does not guarantee its linked SQLite version, so installation
verification checks the actual runtime. The current provider imports `fcntl`
at module load, so Windows is outside the supported scope until a platform-safe
lock implementation is delivered. sqlite-vec failure is intentionally a
degraded semantic-search mode; it does not authorize dropping keyword/BFS
fallbacks.

The source pin remains interim until a suitable upstream release is tested.
Locking, persistence, consolidation safety, crash boundaries, loader contracts,
auxiliary models, embedding isolation, and cron reconciliation remain explicit
adapter contracts in this inventory rather than inferred completion from issue
state.

No other workaround in this inventory is declared retired. A future
retirement PR must update this matrix with the replacement release/ref,
regression command, exact verified head, and rollback trigger.

## Executable verification coverage

The inventory was produced by checking the baseline tree and the exact upstream
refs, not by relying on issue prose. A contributor can repeat the static audit
with these commands from a clean checkout:

```bash
git grep -nE 'core\.|from core|import core|from cron|_patch_upstream_embedding|sys\.path|plugin.yaml' -- \
  plugins/memory/cashew __init__.py plugin.yaml
git grep -nE 'sqlite|vec_embeddings|retrieve_recursive_bfs|sleep|keyword|auxiliary|loader' -- \
  plugins/memory/cashew tests
git show b14d7b013ac0e95314f78cca6cacfef202a30654:pyproject.toml | \
  grep -E 'cashew-brain|requires-python'
```

The existing regression suites exercise the current protections and are the
starting point for retirement tests:

| Contract | Existing executable coverage | Replacement gate |
| --- | --- | --- |
| CPU/model selection and embedding dimensions | `uv run --frozen --extra dev pytest tests/test_embedding_device.py tests/test_migration.py -q` | Keep green while #188 proves the released candidate replacement. |
| Upstream session calls, private pre-compress writes, and neutral failures | `uv run --frozen --extra dev pytest tests/test_sync_worker.py tests/test_cashew_extract.py tests/test_on_pre_compress.py -q` | #188/#191 must add real multi-process persistence evidence before private calls change. |
| Vector/keyword/BFS retrieval fallback | `uv run --frozen --extra dev pytest tests/test_retrieval.py tests/test_recall.py tests/test_handle_tool_call.py -q` | #202/#204 own truthful status and semantic equivalence. |
| Sleep phases and cron import/reconciliation paths | `uv run --frozen --extra dev pytest tests/test_sleep_cron_script.py tests/test_sleep_cron_reconcile.py tests/test_consolidation_safety_contracts.py -q` | Upstream delegation landed in #245; remaining loader and reconciliation proof is owned by #199/#203 contracts. |
| Config, profile isolation, manifests, and lifecycle | `uv run --frozen --extra dev pytest tests/test_config_roundtrip.py tests/test_no_home_leak.py tests/test_initialize_lifecycle.py tests/test_documentation_contracts.py -q` | Keep both loader paths, profile isolation, and manifest contracts green. |

The final #187 verification target is the full offline suite:

```bash
uv run --frozen --extra dev pytest
```

### Integrity repair delegation

The selected composite includes the connection-aware `core.integrity` contract
from upstream PR #137. The adapter delegates `inspect_integrity(conn, ...)`
and explicitly confirmed `apply_integrity_repairs(conn=conn, ...)` to that
module. The caller supplies an open connection inside its outer transaction and
retains ownership of locking, backup, commit, rollback, post-repair inspection,
and close. The path-based operator CLI wraps the same delegation with the
Hermes-owned maintenance admission, persisted model-identity check, verified
SQLite backup, `BEGIN IMMEDIATE` writer exclusion, rollback on pre-commit
failure, and before/after inspection. It does not implement repair algorithms
or change journal mode. Repairs preserve model identity and the maintenance
epoch, so the profile-scoped content-to-embedding cache remains valid and does
not require its own exclusive lease; model-changing work must continue to use
the graph-to-cache migration admission. Older installations without
`core.integrity` remain fail-closed.

The subprocess fixture in `tests/fixtures/cashew-pr137/` is test evidence only.
Its `core/integrity.py` bytes and SHA-256 are checked against upstream PR #137;
the installed composite is checked separately by the selected-source baseline
tests. No production code is copied from the fixture and no upstream maintainer
merge is required for this baseline.

This document does not claim that a local source build, a static import, or a
passing component test proves production compatibility. A future baseline must
record the exact dependency artifact, Hermes revision, CI result, and boundary
test at the same head.

## Deterministic consolidation measurements

Issue [#205](https://github.com/magnus919/hermes-cashew/issues/205) has an
opt-in benchmark for comparing the pinned upstream sleep adapter with future
revisions. It creates a temporary SQLite database, seeds
deterministic vectors and orphan rows, delays a fake embedding client, and
records per-phase wall time, selected work, edge rows committed, orphan rows
repaired, a competing SQLite commit, and whether a participating shared-lock
writer was admitted during the maintenance lease:

```bash
python3 scripts/benchmark-sleep-contention.py --nodes 32 128 --orphans 4 --delay-ms 50
```

The default orthogonal fixture isolates phase cost. To exercise the edge cap,
use `--pair-similarity 0.92 --max-edges 1`; the pinned upstream adapter is
expected to report truthful directed-row accounting when the cap is reached.
That is measurement evidence for
[#193](https://github.com/magnus919/hermes-cashew/issues/193), not a production
claim. The harness never opens `~/.hermes` and does not alter the production
sleep implementation.
