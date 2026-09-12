# hermes-cashew

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin
that stores conversation context in a local [Cashew](https://github.com/rajkripal/cashew)
thought graph with semantic search and automatic context recall. Get from zero to
a working install in under five minutes.

The plugin auto-generates `cashew.json` with defaults on first load, can use an
explicit Hermes `auxiliary.memory` mapping for LLM-powered extraction, adds
forest-level insight extraction via `on_pre_compress`, and runs graph
consolidation on a persistent Hermes cron schedule.

## Integrity audit

Existing profiles can be inspected without opening a writable database:

```bash
python -m plugins.memory.cashew.integrity /path/to/brain.db
```

The audit uses a canonical shared maintenance lease and SQLite URI read-only
mode. It reports schema and provider provenance, ordinary embedding validity,
vector-index parity when the installed extension can be verified, orphan rows,
referential graph defects, and permanence contradictions. It never runs schema
migrations, decay, consolidation, embedding services, or repair as a side
effect. Reports include only bounded counts and reason codes; historical merge
intent remains an explicit manual-review item because it cannot be reconstructed
from the stored graph safely.

The explicit operator apply surface is currently fail-closed:

```bash
python -m plugins.memory.cashew.integrity --apply --confirm /path/to/brain.db
```

It returns a structured `stable_targeted_repair_api_unavailable` result and
does not open or create the profile. Deterministic repair will be enabled only
after Cashew exposes a connection-aware API with atomic ordinary/vector writes,
verified backups, and repeatable post-repair checks. Do not use broad sleep or
whole-database re-embedding as an integrity repair substitute.

## Prerequisites

- [Hermes Agent](https://github.com/nousresearch/hermes-agent) installed
- `cashew-brain` from the reviewed upstream source commit documented below
- SQLite 3.35 or newer, required for Cashew's legacy v1 schema migration
- `sqlite-vec` — enables vector similarity search; included in the manual install below

## Install

```bash
hermes plugins install magnus919/hermes-cashew
```

This clones the repository to `~/.hermes/plugins/cashew/` and registers the
plugin entry point. This interim source baseline uses a direct URL that the
current Hermes dependency installer intentionally rejects. Install the exact
source archive into the Hermes environment before setup:

```bash
CASHEW_PIN='cashew-brain @ https://github.com/magnus919/true/archive/fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz#sha256=23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3'
uv pip install \
  --python ~/.hermes/hermes-agent/venv/bin/python3 \
  --reinstall "$CASHEW_PIN" sqlite-vec
~/.hermes/hermes-agent/venv/bin/python3 \
  ~/.hermes/plugins/cashew/scripts/verify-cashew-baseline.py
```

This archive is a temporary composite fork of upstream Cashew. It combines PR
136 (`ac090ce75ffd2e97dac257cee9430628c68aa241`) and PR 137
(`cb940f34c15460b87831748b2e702334c1c5fbd0`) at tree
`29d97fb93c7998c97be0da8f19b90c6523f9d1e9`. The fork preserves provenance while
those changes are reviewed upstream; replace this pin with the canonical
upstream release once both changes are available there.

The lockfile enforces the archive digest during installation; uv may omit that
digest from the installed PEP 610 metadata, so verification requires the exact
source URL and validates a recorded digest when one is present. The verification
step is required because the selected source and the older
PyPI release both report version `1.2.1`. A version-only check cannot tell them
apart. It also checks the linked SQLite version and source ID. Both the selected
source and the existing PyPI `1.2.1` code require SQLite 3.35 or newer to
migrate a legacy v1 database because that migration uses `ALTER TABLE ... DROP
COLUMN`; this requirement was discovered by the source-pin gate rather than
introduced by it. `hermes memory setup` may warn that it refused the direct
URL; that warning is expected and does not replace or downgrade a manually
installed candidate.

### Optional diagnostics

Diagnostics remain disabled unless explicitly enabled. The normal plugin install
uses runtime dependencies only. To enable either optional Sentry or OpenTelemetry
diagnostics, install the tracing extra into Hermes's environment first:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python3 \
  'hermes-cashew[tracing]'
```

Then set `HERMES_CASHEW_SENTRY_DSN` for Cashew's isolated diagnostics worker,
or set `HERMES_CASHEW_OTEL_ENABLED=1` for Cashew spans. Generic `SENTRY_*` and
`OTEL_*` host settings do not enable Cashew diagnostics. Events and spans contain
only fixed operation codes and allowlisted bounded metadata.

After install, run setup and restart the gateway:

```bash
hermes memory setup
hermes gateway restart
```

## Register with Hermes

After installing, set cashew as the active memory provider:

```bash
hermes config set memory.provider cashew
hermes gateway restart
```

Or use the interactive setup (v0.2.0 now includes cashew in the provider picker):

```bash
hermes memory setup
```

### Update and rollback

`hermes plugins update cashew`, `hermes update`, and a Hermes virtual-environment
rebuild do not install this direct URL automatically. Re-run the exact
`uv pip install` and provenance verification commands above after any of those
operations, then restart the gateway.

To roll the engine back to the published upstream release:

```bash
uv pip install \
  --python ~/.hermes/hermes-agent/venv/bin/python3 \
  --reinstall 'cashew-brain==1.2.1'
hermes gateway restart
```

Do not run `hermes memory setup` expecting it to restore the source pin. To
return to the selected source later, re-run the full pinned install and
provenance verification commands.

## Zero-Config Startup

hermes-cashew works out of the box — all 17 persisted configuration fields have
sane defaults and are backed by current runtime behavior. On first agent
startup, the plugin creates `~/.hermes/cashew.json` with the full default
configuration. It never edits Hermes `config.yaml`; until an auxiliary role is
explicitly configured there, extraction remains heuristic-only.

Edit `~/.hermes/cashew.json` only if you want to override specific defaults.
The file is never overwritten once it exists:

```bash
# Optional: override individual defaults
cat > ~/.hermes/cashew.json << 'EOF'
{
  "recall_k": 10,
  "think_interval": 15,
  "user_domain": "user"
}
EOF
```

### Full Config Reference

#### Core

| Key | Default | Description |
|-----|---------|-------------|
| `cashew_db_path` | `cashew/brain.db` | Path to SQLite DB, relative to `hermes_home` |
| `embedding_model` | `thenlper/gte-large` | Sentence-transformers model for embeddings (1024-dim) |
| `embedding_device` | `cpu` | Sentence-transformers device; `cpu` avoids unstable native accelerators |
| `llm_aux_role` | `memory` | Hermes auxiliary role for LLM-powered extraction; requires `auxiliary.memory` in `config.yaml` |
| `auto_extraction` | `true` | Auto-extract knowledge from conversation turns |
| `sync_queue_timeout` | `30.0` | Seconds to wait for sync worker drain on shutdown |

#### Retrieval

| Key | Default | Description |
|-----|---------|-------------|
| `recall_k` | `5` | Context fragments returned per query |
| `prefetch_k` | `3` | Nodes to pre-warm into context on each turn |
| `prefetch_cues` | `3` | Cue phrases to send to LLM for prefetch generation |

#### Domains & Classification

| Key | Default | Description |
|-----|---------|-------------|
| `user_domain` | `user` | Domain label for user messages |
| `ai_domain` | `ai` | Domain label for AI messages |

#### Sleep Cycle

| Key | Default | Description |
|-----|---------|-------------|
| `sleep_cycles` | `true` | Enable the refactored sleep cycle (cross-linking, dedup, GC, dreams) |
| `sleep_schedule` | `"every 12h"` | Cron schedule for sleep cycle |
| `sleep_max_nodes` | `2000` | Max nodes per sleep cycle tick |
| `think_cycles` | `true` | Enable periodic insight generation (think cycle) |
| `think_interval` | `10` | Turns between think cycle runs (0 = disable) |

Cashew validates the effective JSON and `CASHEW_*` configuration before use.
`recall_k` and `prefetch_k` allow 1–20; `prefetch_cues` allows 0–20;
`think_interval` allows 0–10,000; `sleep_max_nodes` allows 1–2,000; and
`sync_queue_timeout` must be finite and between 0 and 300 seconds. Invalid
JSON or values are left in place so they can be corrected instead of being
silently overwritten. These numeric ceilings are a new validation policy.

`embedding_device` defaults to `cpu`. SentenceTransformer model loading and
encoding run in one provider-owned child process. A backend process exit cannot
terminate Hermes, and an interactive caller stops waiting after a bounded
interval if the worker hangs. Set the option to `auto` or an explicit device
such as `mps`, `cuda`, or `cuda:0` after validating that backend. A non-CPU
startup failure retries once on CPU; Hermes never loads the model as a fallback.

This boundary is deliberately narrow. SQLite, sqlite-vec, NumPy operations in
the adapter, Cashew graph algorithms, and auxiliary LLM calls still run in the
Hermes process. The child adds startup and local IPC overhead; deterministic
transport checks are recorded in CI, while real-model latency and memory use
remain hardware/model dependent and are not represented by those fake-model
measurements.

#### Legacy settings

Starting with v0.11.0, the following deprecated parse-only settings are removed
because cashew-brain 1.x has no supported runtime contract for them:

`default_domain`, `auto_classify`, `domain_classifications`,
`domain_separation_enabled`, `token_budget`, `walk_depth`,
`similarity_threshold`, `access_weight`, `temporal_weight`, `clustering_eps`,
`clustering_min_samples`, `novelty_threshold`, `max_think_iterations`,
`think_cycle_nodes`, `gc_mode`, `gc_threshold`, `gc_grace_days`,
`gc_protect_types`, `gc_think_cycle_penalty`, `decay_pruning`, and
`pattern_detection`.

These keys never changed provider behavior. Existing files continue to load;
removed keys are ignored and pruned the next time `hermes memory setup` saves
the provider configuration. Remove matching `CASHEW_*` environment variables,
because they are no longer part of the adapter's configuration surface.

#### Feature Flags

Experimental features gated behind boolean toggles. All default to `false`.
Enable in `cashew.json` under the `_features` key:

```json
{"_features": {"experimental_batch_sync": true}}
```

| Key | Default | Description |
|-----|---------|-------------|
| `experimental_batch_sync` | `false` | Drain up to 8 sync turns per worker iteration instead of one-at-a-time |

The former `experimental_parallel_retrieval` setting is retired. Existing
profiles still load; saving removes that setting. Recall always uses upstream
retrieval first, with keyword fallback for an empty result or failure. This
removes the competing retrieval threads and timing-dependent result selection;
it does not impose a deadline on the normal upstream call.

Environment variables override config values: prefix any key with `CASHEW_`
(e.g. `CASHEW_RECALL_K=10`).

## Verify the Install

```bash
hermes gateway restart   # ensure gateway picks up the new plugin
hermes memory status
```

Expected output shows `Provider: cashew` with `Plugin: installed` and `Status: available`.

## How It Works

`hermes-cashew` provides two LLM-accessible tools:

- **`cashew_query`** — searches the local thought graph for context relevant to
  the current conversation. Uses sqlite-vec for semantic search; optional
  `domain`, `tag`, and `exclude_tags` filters narrow the result set.
- **`cashew_extract`** — explicitly persists a conversation turn into the graph.
  The agent can call this when it judges a turn contains worth-remembering knowledge.

Both tools are registered automatically when Hermes loads the plugin.
On each session start, `prefetch()` retrieves relevant context from the graph
and injects it into the system prompt.

## Privacy Controls (Optional)

Nodes in the thought graph can carry tags. The `cashew_query` tool accepts an
`exclude_tags` parameter to filter out nodes with specific tags from results:

```json
{"query": "prior decisions", "exclude_tags": ["vault:private"]}
```

This works in both the vector search and keyword fallback paths. Common use cases:

- **Privacy**: Tag sensitive nodes with `vault:private` to exclude them from
  group or shared contexts
- **Domain isolation**: Exclude nodes from specific domains during broad queries
- **Declassification**: Remove exclusion to reveal previously private nodes

## LLM Integration

`llm_aux_role` selects a Hermes auxiliary role. Its default, `"memory"`, is a
selector rather than an implicit configuration: add a complete
`auxiliary.memory` mapping to Hermes `config.yaml` to opt in. The plugin never
creates or changes that mapping.

```yaml
auxiliary:
  memory:
    provider: your-provider
    model: your-model
```

For each extraction, Cashew uses Hermes' public auxiliary-client resolver for
the selected profile's transport and authentication, then makes one bounded
chat-completions request. Prompts are capped at 32,000 characters, responses at
1,024 tokens, and the caller waits up to 30 seconds. A timed-out provider call
cannot be forcibly cancelled and may finish later; no more than four such
backend calls may be outstanding process-wide, and later requests fail closed
until a slot is released. Cashew intentionally does not use Hermes'
task-level retry and fallback ladder for these calls.

The configured role is used when Cashew next performs extraction, insight, or
dream work. If Hermes cannot resolve it, Cashew silently uses its heuristic
extractor instead.

To disable LLM extraction (heuristic-only mode), set `llm_aux_role` to null in
`cashew.json`:

```json
{"llm_aux_role": null}
```

An absent, null, or incomplete role mapping also keeps extraction heuristic-only.

### What the LLM enables upstream

- **LLM extraction** — structured knowledge extraction with typed nodes,
  confidence scores, tags, and domain assignment
- **Think cycles** — cross-domain synthesis, generates `insight` nodes
  from clusters of related knowledge. Runs every `think_interval` sync
  turns (default 10). Set `think_interval` to 0 to disable.
- **Sleep synthesis** — Graph consolidation pipeline: cross-linking, dedup,
  garbage collection, permanence evaluation, core memory promotion, and
  LLM-powered dream generation. Runs as a **Hermes cron job** on a configurable
  schedule (default: every 12 hours), not at session boundaries. The cron script
  reads `cashew.json` at runtime and uses the same explicit auxiliary role when
  one is configured; otherwise it runs without an LLM.
  Processes up to `sleep_max_nodes` per cycle (default 2,000).
- **Pre-compress insight extraction** — Before context compression discards
  old messages, extracts conversation-arc patterns (topic shifts, framing
  changes, implicit decisions) using a dedicated LLM prompt. Creates
  `insight`/`observation` nodes in the graph. Requires `llm_aux_role`
  configuration. Silent-degrades without LLM.

Without an explicitly enabled `llm_aux_role` mapping, the plugin uses
heuristic-only extraction — no API calls and no LLM cost.

**Design note:** The `auxiliary.memory` convention is provider-agnostic.
Any memory provider plugin can declare `llm_aux_role` and reference the
same `auxiliary.memory` section, making this a standard pattern across
the Hermes plugin ecosystem.

## Sleep Cycle Cron Scheduling

hermes-cashew runs its graph consolidation pipeline (cross-linking, dedup,
garbage collection, permanence evaluation, core memory promotion) as a
**Hermes ``no_agent`` cron job**, not at session boundaries. This means
``/new`` returns instantly — no synchronous sleep cycle work blocks the
start of a new session.

### When the cron job is registered

The cron job is created during plugin initialization (``initialize()``) only
when **all** of the following are true:

| Condition | Config Key | Default | Behavior if false |
|-----------|-----------|---------|-------------------|
| Sleep cycles enabled | ``sleep_cycles`` | ``true`` | Cron not registered |
| Schedule non-empty | ``sleep_schedule`` | ``\"every 12h\"`` | Cron not registered; set to ``\"\"`` to disable |
| Provider init succeeds | — | — | Exception caught, ``_config`` set to ``None``, cron never reached |
| Hermes cron module available | — | — | ``ImportError`` caught, WARNING logged |
| ``create_job()`` succeeds | — | — | Exception caught, WARNING logged |
| Matching profile-owned job already registered | — | — | Existing job is adopted without changing its schedule |

The job persists across ordinary provider shutdown and is adopted by a later
initialize for the same Hermes profile. Reconciliation serializes concurrent
initializers, tags ownership with an opaque profile token, and only replaces or
disables a job carrying that token. It never deletes a similarly named job that
does not prove it belongs to this profile.

### When the cron job runs

On the configured schedule (default ``every 12h``), the Hermes scheduler
executes ``$HERMES_HOME/scripts/cashew-sleep-cycle.py`` as a ``no_agent``
script. Registration embeds the complete validated effective configuration,
including JSON/default values and any valid ``CASHEW_*`` overrides, so a cron
daemon cannot drift from its provider's DB, model, device, limits, or auxiliary
role. Changes to JSON or environment settings take effect on the next provider
initialize, which atomically refreshes the script and reconciles the job. The
cycle uses LLM dream synthesis only when that embedded configuration has an
explicit configured auxiliary role; otherwise it runs with no LLM overhead.

The generated script is pinned to the Cashew installation that registered the
job. This keeps one Hermes profile from loading another profile's provider.
After moving, reinstalling, or changing the plugin layout, reinitialize
Cashew to refresh the script and cron registration. Development installs may
use the documented ``$HERMES_HOME/hermes-agent/plugins/memory/cashew`` symlink
to an external checkout.

### What happens during a cron tick

1. Uses the validated effective configuration embedded when the job was
   registered; it does not reread ``cashew.json`` or ambient ``CASHEW_*``
   values during a tick
2. Selects up to ``sleep_max_nodes`` (default 2,000) eligible nodes
3. Computes pairwise cosine similarity (vectorized numpy)
4. Creates and repairs cross-links using the configured model-profile thresholds
5. Deduplicates near-identical nodes through maximal-clique consolidation
6. Runs garbage collection on low-fitness isolated nodes
7. Promotes frequently-accessed nodes to permanent / core memory status
8. Prints a JSON summary (captured by the cron scheduler's output log)

Without an explicit auxiliary role, no LLM-powered dream generation occurs in
cron mode. Cross-linking, dedup, and GC still provide the graph-maintenance
benefit without a provider dependency in the subprocess.

### Maintenance lock scope

Embedding-dimension migration and the synchronous portion of a sleep cycle use
the same nonblocking advisory lock derived from the configured
``cashew_db_path``. A contended migration defers and a contended cycle skips;
the lock file is retained because a process exit releases its ``flock``
descriptor automatically. Do not delete an old lock file to recover a cycle:
unlinking it can let a second pathname refer to a different inode while the
original process still holds the lock.

When an API caller enables ``background_dream=True``, the returned cycle
summary records ``dream_pending`` and the daemon uses its own connection; it is
not guarded by the synchronous maintenance lock. This advisory lock is not a complete
shared-brain writer-coordination policy; broader coordination is tracked in
[#191](https://github.com/magnus919/hermes-cashew/issues/191).

### Config reference

| Key | Default | Description |
|-----|---------|-------------|
| ``sleep_schedule`` | ``\"every 12h\"`` | Cron expression or interval string. Set to ``\"\"`` to disable cron-based scheduling entirely. Examples: ``\"every 30m\"``, ``\"0 */2 * * *\"``, ``\"0 3 * * *\"`` (daily at 3am). |
| ``sleep_max_nodes`` | ``2000`` | Maximum number of eligible nodes considered in one sleep cycle. Higher values can increase consolidation work and tick time. |

## Semantic Search

`sqlite-vec` enables vector similarity search and is installed automatically as a
standard dependency. If your platform doesn't support sqlite-vec's native extension,
the plugin degrades gracefully to keyword-based retrieval — still functional,
but less precise.

sqlite-vec is a standard dependency and will always be loaded at startup.

## Uninstall

Normal provider shutdown intentionally preserves the profile-owned cron job.
Before removing the plugin, disable sleep in Cashew setup (set
``sleep_cycles`` to ``false`` or ``sleep_schedule`` to ``""``) and initialize
the provider once. Reconciliation then removes only the job whose ownership
matches that Hermes profile. Afterwards use the host-supported removal flow:

```bash
hermes plugins remove cashew
hermes config set memory.provider built-in   # revert to built-in memory
rm -rf ~/.hermes/cashew   # optional: remove the local graph data
```

## Troubleshooting

### `Plugin: NOT installed` in `hermes memory status`

1. **cashew-brain not installed in Hermes venv** — install and verify the exact
   source baseline from [Install](#install). If `uv` is unavailable, bootstrap
   pip in the Hermes environment and pass the same quoted `CASHEW_PIN` value:
   ```bash
   CASHEW_PIN='cashew-brain @ https://github.com/magnus919/true/archive/fcb4919ac37144bfbeb822eaafc668a4bdceb791.tar.gz#sha256=23765a473ab550db86856fd4b8f0a011d6046196f2e87ebb263a752e8d114db3'
   ~/.hermes/hermes-agent/venv/bin/python3 -m ensurepip
   ~/.hermes/hermes-agent/venv/bin/python3 -m pip install \
     --force-reinstall "$CASHEW_PIN" sqlite-vec
   ```

2. **Stale pycache or entry point not registered** — If cashew-brain is installed
   but the plugin still shows NOT installed:
   ```bash
   cd ~/.hermes/plugins/cashew && \
     ~/.hermes/hermes-agent/venv/bin/python3 -m pip install -e .
   hermes gateway restart
   ```

### `Status: not available`

The plugin is available when cashew-brain is importable. Check:
```bash
~/.hermes/hermes-agent/venv/bin/python3 -c "from core.context import ContextRetriever; print('ok')"
```
If this fails, cashew-brain is not installed in the Hermes venv (see above).

### Hermes-agent venv has no `pip`

Hermes-agent creates a minimal venv without pip. Bootstrap it first:

```bash
~/.hermes/hermes-agent/venv/bin/python3 -m ensurepip
~/.hermes/hermes-agent/venv/bin/python3 -m pip install <package>
```

Do **not** run `pip install` from outside the venv targeting the hermes python,
or the package will land in the wrong environment.

### Embedding model download on first use

`cashew-brain` bundles sentence-transformers. The first retrieval operation may
trigger a ~500 MB embedding model download. To avoid this in automated environments:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 hermes ...
```

### Embedding dimension migration

When `embedding_model` changes, the provider compares both stored embedding
rows and the sqlite-vec table with the configured model's dimension during
initialization. A mismatch is repaired before background workers start:

1. a consistent SQLite backup is written under `cashew/backups/`;
2. cashew-brain re-embeds active nodes and recreates `vec_embeddings` at the
   configured dimension;
3. the provider validates the new dimensions before enabling retrieval.

If backup, migration, or validation fails, the provider logs a warning and
restores the backup. Thought nodes are not discarded. Stop other Hermes or
Cashew processes before deliberately changing `embedding_model`, then restart
Hermes and allow the one-time migration to finish before issuing queries.

## Development

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then use
the committed lockfile for a reproducible development environment:

```bash
git clone https://github.com/magnus919/hermes-cashew
cd hermes-cashew
uv sync --frozen --extra dev
uv run --frozen --extra dev pytest
```

Tests require no network access and mock the embedding model automatically
(`HF_HUB_OFFLINE=1` is set by `tests/conftest.py`).

## Architecture Notes

The plugin uses a dual-path loading strategy to support both `pip install -e .`
(development) and `hermes plugins install` (flat-entry loader):

- **pip / test path**: Python's namespace package mechanism resolves
  `plugins.memory.cashew` to `plugins/memory/cashew/__init__.py` via `sys.path`
- **flat-entry path**: Hermes loads `~/.hermes/plugins/cashew/__init__.py` as
  `_hermes_user_memory.cashew`. The root `__init__.py` detects this context
  and exec's the nested implementation with `sys.modules` patched so relative
  imports resolve correctly

## License

See [LICENSE](./LICENSE).
