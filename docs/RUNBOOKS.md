# ARS Operational Runbooks

Terminal-first procedures. Every command is copy-pasteable. Keep this in sync with the code.

## Model rollback

Symptom: a promoted model regresses in production (WER/latency/complaints).

```bash
python -m ars.registry list                 # see production + previous
python -m ars.registry rollback             # atomic blue-green: previous -> production
python -m ars.registry list                 # confirm the flip
```

The registry write is atomic (temp file + rename). Rollback is instant and reversible
(`python -m ars.registry promote <version>` re-promotes). No model files are deleted.

## Rule retirement

Symptom: a confusion rule over-corrects (false-correction alarm / customer complaint).

1. Find the rule id in `configs/rules/rules-<lang>.yaml`.
2. Set its `status: retired` (never delete — keep the audit trail and its golden cases).
3. Redeploy the API (rules load at startup and on SIGHUP).
4. Confirm: the rule no longer appears in `Keydetector` fires; its golden cases are skipped.

For a softer step, set `status: approved` (log-only) instead of retired.

## Flywheel pause

Symptom: bad judge labels or a data incident; stop the loop from acting.

```bash
# configs/default.yaml
flywheel:
  enabled: false        # weekly_cycle becomes a no-op; harvesting/telemetry continue
```

Re-enable by setting `true`. In-flight cycles are idempotent (state in `cycle_state`), so a
pause mid-cycle is safe to resume.

## Judge outage

Symptom: the LLM judge provider is down or rate-limited; the weekly cycle must not wedge.

- Every judge call has a timeout; a batch that fails is retried once, then the cycle proceeds
  with the items it has (harvested-but-unjudged items roll into next week's window).
- To switch providers: set `judge.provider` / `judge.model` (Anthropic ↔ OpenAI). A cheaper
  model may only become the **production** judge after the calibration gate
  (`python -m ars.judge.calibrate`, ≥90% agreement vs claude-sonnet-5).
- MockJudge is never used in production; it exists only for tests/simulation.

## Disk-full

Symptom: writes fail; `df -h` shows the data volume near capacity.

1. Run retention (dry-run first): `python -m ars.ops.retention` inspects; then purge raw audio
   past `ops.retention_days` (transcripts/metrics are kept — soft-delete, no orphaned rows).
2. Clear regenerable artifacts: `reports/` (except READMEs), `_staging/` source archives,
   old `models/ct2|merged` for retired versions.
3. Never delete `models/registry.json`, dataset `manifest.parquet`, or the SQLite DB.

## Restore-from-backup

Symptom: DB or registry corruption.

1. Stop the API and the flywheel (`flywheel.enabled: false`).
2. Restore `data/db/ars.db` and `models/registry.json` from the latest backup.
3. Datasets and models are content-addressed by manifest — re-fetch with
   `make download-data` / rebuild via the phase scripts if audio is missing.
4. Restart; confirm `GET /v1/model` returns the expected production entry.
