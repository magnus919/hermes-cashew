# hermes-cashew Runbooks

Troubleshooting guides for common operational issues.

## SQLite Lock Contention

**Symptom:** `database is locked` errors in logs or `RuntimeError` in sync worker.

**Cause:** Multiple Hermes sessions or concurrent sleep cycles contending for the same `brain.db`.

**Resolution:**
1. Check for stale sleep lock files: `ls -la $HERMES_HOME/cashew/brain.db.sleep.lock`
2. The plugin self-heals locks older than 60 minutes on startup.
3. If stuck, stop all Hermes processes and manually remove: `rm $HERMES_HOME/cashew/brain.db.sleep.lock`
4. Restart Hermes. The plugin sets `PRAGMA busy_timeout=5000` on both connections.

## Embedding Model Download Failures

**Symptom:** Slow first startup, `HF_HUB_OFFLINE=1` preventing download, or disk space issues.

**Resolution:**
1. The plugin uses `thenlper/gte-large` by default (~670MB).
2. Ensure `HF_HUB_OFFLINE` is not set in the runtime environment.
3. Check disk space: at least 2GB free recommended.
4. To pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('thenlper/gte-large')"`

## Sleep Cycle Not Running

**Symptom:** Cron-based sleep cycles never execute.

**Cause:** In v0.8.0+, every session start adopts the existing cron job rather than re-registering. If no job exists, one is created with a 12-hour schedule.

**Resolution:**
1. Check `$HERMES_HOME/jobs.json` for `cashew-sleep-cycle` entry.
2. Verify Hermes cron subsystem is active: `hermes cron list` (or equivalent).
3. If missing, delete `$HERMES_HOME/jobs.json` and restart Hermes — the plugin will recreate it.
4. Logs should show: `sleep: registered cron job <id> (schedule=0 */12 * * *)`

## Hermes Config Resolution

**Symptom:** LLM extraction not working, falling back to heuristic.

**Cause:** `llm_aux_role` references `auxiliary.memory` in Hermes `config.yaml`, but the section may be missing or misconfigured.

**Resolution:**
1. Ensure `$HERMES_HOME/config.yaml` has:
   ```yaml
   auxiliary:
     memory:
       model: gpt-4o-mini
       api_key: sk-...
       base_url: https://api.openai.com/v1
   ```
2. Set `llm_aux_role` in `$HERMES_HOME/cashew.json` to `"memory"`.
3. Logs should show: `using llm_aux_role='memory' model=gpt-4o-mini`

## Sync Queue Overflow

**Symptom:** `WARNING: cashew sync queue full, dropping oldest pending turn` in logs.

**Cause:** Sync worker can't keep up with incoming turns (queue bounded at 16 entries).

**Resolution:**
1. Check for slow LLM extraction (if using `llm_aux_role`).
2. Disable LLM extraction by removing `llm_aux_role` from `cashew.json`.
3. Ensure the SQLite database is on fast storage (not network/NFS).
4. The plugin gracefully drops oldest entries; no data corruption.
