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
| cashew-brain | `ac090ce75ffd2e97dac257cee9430628c68aa241` | Immutable [upstream source archive](https://github.com/rajkripal/cashew/archive/ac090ce75ffd2e97dac257cee9430628c68aa241.tar.gz), locked with archive SHA-256 `0777dcb89bde8e0d6103786358c93c0ad3fd4ffb7f8a210194c37b211ae4c28b`. The installed `core/session.py` SHA-256 is `0ce60cc63adf4fb7136581aee722bb10e9a344e556b6fb98f4d46855d53c36cd`; the candidate-specific `core/sleep.py` SHA-256 is `2d8cd35ee2ce45f6b2eb3eb4dc1396ab6e157711c0a7eadfbf5e4afe6823b7db`. This is the reviewed head of upstream PR [#136](https://github.com/rajkripal/cashew/pull/136); the downstream pin remains a draft until that PR merges.
| Hermes Agent loader/lifecycle evidence | `990473a79c6b0396b0a648fdd85ee8f7a5c267d3` | The #199 lane merged in wrapper commit `cc1e31c66b1f0cf72079d91497e21c7347c47c99`. Its flat and development loader, auxiliary routing, lifecycle, and cron contracts also passed in a fresh environment containing the exact #188 source pin. The embedding model and external client were deterministic fakes; Hermes, SQLite 3.47.1, and the Cashew package boundary were real. This is bounded compatibility evidence, not a published Hermes minimum-version promise or native embedding-crash proof. |

The candidate is the reviewed upstream PR #136 head rather than a mutable fork
branch or a floating `main` reference. It contains the bounded sleep contract,
schema-repair and contention fixes, and the other changes recorded in issue
#193. Production installs must remain on the currently selected baseline until
upstream merges #136; this draft exercises the exact candidate archive so the
downstream replacement can be reviewed at the real dependency boundary:

```text
https://github.com/rajkripal/cashew/archive/ac090ce75ffd2e97dac257cee9430628c68aa241.tar.gz
sha256=0777dcb89bde8e0d6103786358c93c0ad3fd4ffb7f8a210194c37b211ae4c28b
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

The historical #188 source includes the schema fast path but does not resolve
every upstream gap. In particular, its capped cross-link loop could stop before
flushing its pending batch. The reviewed #136 candidate at `ac090ce` includes
the bounded flush fix and its regression coverage; this draft exercises that
candidate, but it remains unmerged upstream and has not replaced the wrapper's
local consolidation engine. The candidate still has no supported
device-injection contract, and several wrapper calls still target private
upstream functions. None of those adapter safeguards may be retired until the
candidate is merged and the downstream replacement gates pass.

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

## Workaround and compatibility inventory

The status is deliberately conservative. “Retain” means the current behavior
stays in place while a replacement is proved. “Replace” means an upstream
replacement is the intended destination, not that this issue implements it.
“Remove” is reserved for duplicate behavior with a verified owner.

| Surface and evidence in `b14d7b0` | Original symptom / current responsibility | Upstream replacement and status | Decision, retirement prerequisite, and required regression |
| --- | --- | --- | --- |
| [`_patch_upstream_embedding`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L276-L340) | The released Cashew 1.2.1 artifact exposes model-name and dimension globals but no supported provider device injection. The adapter must select the configured model, force safe CPU fallback, and prevent one Hermes profile/session from inheriting another's backend state. The historical device/host-crash cause is not recovered; the protection is observable in the patch and its tests. | The selected source still constructs `SentenceTransformer` without a device parameter; commit `394d6cffe308ba106df8db750248f1ff272bb814` only guards daemon dimension mismatch. No replacement contract exists. | **Retain.** Replace only after a released upstream model/device contract proves CPU fallback, model dimension, daemon mismatch, and two-profile isolation in tests. Preserve rollback to the patched path. |
| Embedding-dimension and `vec_embeddings` migration/create in [`__init__.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L821-L1080) | Older brains can contain the rowid-shaped vec schema or vectors from a previous model dimension. The adapter detects/rebuilds the incompatible virtual table, derives the configured dimension, and falls back when sqlite-vec is unavailable. The original host-crash history is unknown; the old schema failure is documented in the code and commit `43eea9e`. | The selected source creates the current `node_id` vec table and contains dimension/daemon guards (`233b75684025656cde46bf4a2af9bca416ef5145`, `394d6cffe308ba106df8db750248f1ff272bb814`). It does not prove migration of every legacy adapter-created table. | **Retain, then replace migration ownership.** #188 must test old/new schema, mixed dimensions, persistence, and rollback before local migration is removed. The provider's profile path guard and neutral fallback remain Hermes responsibilities. |
| Historical local consolidation engine [`sleep_refactor.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/sleep_refactor.py) (candidate discovery, cross-links, connected-component dedup, rewiring, GC, permanence, orphan embedding) | This was a downstream workaround for upstream sleep defects and made the wrapper own a second consolidation implementation. | The reviewed #136 candidate at `ac090ce` contains the bounded flush fix and regression coverage. This draft delegates the algorithm to the pinned public `core.sleep.run_sleep_cycle` contract while retaining only Hermes admission, locking, journal policy, embedding-worker, and result-boundary code. | **Remove after upstream #136 merges and this draft passes CI/review.** Until then, the candidate pin and this stacked draft remain open; the historical module path is a transition shim only. |
| Keyword fallback [`_keyword_search`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1900-L1950) and neutral retrieval fallback around [`retrieve_recursive_bfs`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1580-L1810) | Recall must remain usable when sqlite-vec, an embedding model, or an upstream retrieval call is unavailable. The fallback is an availability safeguard, not a second ranking engine. | Upstream retrieval remains the engine owner; upstream does not provide this Hermes failure-isolation contract. | **Retain as adapter responsibility.** #202/#204 must prove truthful status, context correctness, bounded work, and deterministic fallback before any simplification. |
| Hermes-specific result enrichment and access accounting [`_enrich_results` / `_update_access_metrics`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1081-L1157) | Cashew retrieval returns engine-shaped rows; Hermes needs stable context formatting and profile/session access updates. | No upstream replacement is implied: this is integration behavior. | **Retain.** Keep tests at the provider boundary; do not move it into Cashew engine code. |
| Direct persistence helpers [`_create_node` and `_set_node_tags`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1388-L1435) | The pre-compress hook needs to persist forest-level insight nodes and tags. These underscore-prefixed Cashew functions are private and can change without compatibility guarantees. | Upstream `core.session` owns the helpers; no public persistence API is present in the tested release. | **Replace when available; retain now.** Require an upstream public write API, tag/embedding regression coverage, and rollback before changing this path. |
| `core.context.ContextRetriever` initialization and availability probe ([`__init__.py#L49-L55`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L49-L55), [`#L500-L535`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L500-L535)) | Provider startup must degrade cleanly when Cashew is absent while creating the upstream context retriever only after profile-scoped paths exist. The constructor and availability behavior are upstream-owned. | `core.context.ContextRetriever` is present in the released package, but no Hermes-specific lifecycle adapter is supplied upstream. | **Retain and pin.** #199 must run the real provider initialization/availability contract against the exact Hermes and Cashew refs; replacement requires an upstream lifecycle API and failure-path regression. |
| Direct session/retrieval/embedding/backup calls: `core.session.end_session` and `think_cycle` ([`__init__.py#L1159-L1240`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1159-L1240)), `core.retrieval.retrieve_recursive_bfs` ([`#L1580-L1810`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1580-L1810)), `core.embeddings.embed_nodes` ([`#L1388-L1435`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1388-L1435)), `core.db.ensure_schema` and `core.backup.create_backup` ([`#L821-L980`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L821-L980)) | These calls cross the package boundary directly. The underscored helpers above are private; the others are public-looking functions whose signatures and result shapes are still upstream-owned. | `cashew-brain` owns these engine operations. The candidate main ref adds schema fast-path and resource/dimension fixes, but no released stable adapter API. | **Retain and pin.** #188/#189/#191/#206 must exercise schema, writes, backups, locking, and result shapes against the exact dependency. Replace private calls when upstream publishes a stable contract; do not reimplement engine algorithms here. |
| `core.embedding_service.resolve_embedding_dim` ([`__init__.py#L947-L950`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L947-L950), [`#L1061-L1070`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L1061-L1070)) and upstream permanence handling | The adapter passes the Hermes-owned worker dimension into upstream and validates it at the public sleep boundary. | Candidate `ac090ce` owns the sleep pipeline's permanence and dimension behavior; the adapter retains no parallel implementation. | **Verify at the pinned candidate boundary.** Follow-up changes must update the candidate pin and contract tests together. |
| Provider lock and profile isolation (`fcntl`, temporary backup, and Hermes-home-derived paths) in [`__init__.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L930-L980) and [`config.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/config.py#L502-L570) | Concurrent provider processes must not corrupt a shared brain, and user-configured paths must remain below the active Hermes profile. This is the adapter's trust and data-safety boundary. | Cashew schema and engine locks do not establish Hermes profile isolation. | **Retain.** #189/#191/#197 must prove multi-process contention, lock release, crash recovery, and path traversal rejection. Never retire these protections as “upstream cleanup.” |
| Sleep cron registration/deduplication and standalone script import fallback ([`__init__.py#L87-L98`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L87-L98), [`#L355-L376`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L355-L376), [`sleep_cron_script.py#L64-L86`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/sleep_cron_script.py#L64-L86)) | Hermes cron jobs persist across restarts and the plugin must avoid duplicate sleep jobs while working from both bundled and flat installs. The script has two import paths because the loader and installed package expose different module layouts. | Hermes `cron.jobs` and the shared directory loader are the platform contracts; Cashew does not own them. | **Retain pending contract verification.** #186/#199/#203 must prove one job per profile, correct `HERMES_HOME`, no-agent execution, both loaders, and shutdown/restart behavior before dedup or import fallback is simplified. |
| Dual loader and manifest contract: root [`__init__.py`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/__init__.py#L1-L90), nested provider [`register`](https://github.com/magnus919/hermes-cashew/blob/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew/__init__.py#L2160-L2172), and root/nested [`plugin.yaml`](https://github.com/magnus919/hermes-cashew/tree/b14d7b013ac0e95314f78cca6cacfef202a30654/plugins/memory/cashew) manifests | Hermes bundled discovery and `hermes plugins install` do not use the same import path. The root shim must stay dependency-light and namespace packages must not be shadowed. | Hermes plugin loader supports directory modules and entry points; no single-path guarantee is assumed. | **Retain.** #199 must test both loader paths against the exact Hermes revision; #208 may remove duplication only after a platform contract and rollback are documented. |
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

The milestone implementation issues own the next decisions:

- [#188](https://github.com/magnus919/hermes-cashew/issues/188) consumes the
  candidate's actual fixed source and establishes its interim source-install
  baseline; a PyPI release remains blocked until a suitable upstream release.
- [#189](https://github.com/magnus919/hermes-cashew/issues/189),
  [#190](https://github.com/magnus919/hermes-cashew/issues/190), and
  [#191](https://github.com/magnus919/hermes-cashew/issues/191) own lock,
  contention, persistence, and retry correctness.
- [#192](https://github.com/magnus919/hermes-cashew/issues/192) and
  [#193](https://github.com/magnus919/hermes-cashew/issues/193) own the
  consolidation safety specification and any engine replacement.
- [#197](https://github.com/magnus919/hermes-cashew/issues/197),
  [#199](https://github.com/magnus919/hermes-cashew/issues/199),
  [#200](https://github.com/magnus919/hermes-cashew/issues/200),
  [#201](https://github.com/magnus919/hermes-cashew/issues/201),
  [#203](https://github.com/magnus919/hermes-cashew/issues/203), and
  [#208](https://github.com/magnus919/hermes-cashew/issues/208) own crash
  boundaries, Hermes lifecycle/loader contracts, auxiliary models, embedding
  isolation, cron reconciliation, and structural cleanup.

No workaround in this inventory is declared retired. A future retirement PR
must update this matrix with the replacement release/ref, regression command,
exact verified head, and rollback trigger.

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
| Upstream sleep boundary and cron import/reconciliation paths | `uv run --frozen --extra dev pytest tests/test_consolidation_safety_contracts.py tests/test_sleep_cron_script.py tests/test_sleep_cron_reconcile.py -q` | #193 owns the adapter and loader proof; the candidate pin in #236 must remain its immutable base. |
| Config, profile isolation, manifests, and lifecycle | `uv run --frozen --extra dev pytest tests/test_config_roundtrip.py tests/test_no_home_leak.py tests/test_initialize_lifecycle.py tests/test_documentation_contracts.py -q` | #198/#199/#208 own contract consolidation. |

The final #187 verification target is the full offline suite:

```bash
uv run --frozen --extra dev pytest
```

This document does not claim that a local source build, a static import, or a
passing component test proves production compatibility. A future baseline must
record the exact dependency artifact, Hermes revision, CI result, and boundary
test at the same head.

## Deterministic consolidation measurements

Issue [#205](https://github.com/magnus919/hermes-cashew/issues/205) has an
opt-in benchmark for comparing the current local sleep adapter with a future
upstream replacement. It creates a temporary SQLite database, seeds
deterministic vectors and orphan rows, delays a fake embedding client, and
records per-phase wall time, selected work, edge rows committed, orphan rows
repaired, a competing SQLite commit, and whether a participating shared-lock
writer was admitted during the maintenance lease:

```bash
python3 scripts/benchmark-sleep-contention.py --nodes 32 128 --orphans 4 --delay-ms 50
```

The default orthogonal fixture isolates phase cost. To exercise the edge cap,
use `--pair-similarity 0.8 --max-edges 1`; the current pinned local adapter is
expected to report `bounded_integrity: false` because its known pending batch
is not flushed when the cap is reached. That is measurement evidence for
[#193](https://github.com/magnus919/hermes-cashew/issues/193), not a production
claim. The harness never opens `~/.hermes` and does not alter the production
sleep implementation.
