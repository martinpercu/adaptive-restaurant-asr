# STATUS — Build Progress Tracker

> Maintained by the builder. One row per phase; update the row **when the phase's exit checklist is fully green**, never before. Evidence = paths to reports + key metric numbers (must match the report files).

| Phase | Name | Status | Closed on | Gate evidence |
|-------|------|--------|-----------|---------------|
| 0 | Foundations | in progress | — | — |
| 1 | Baseline ASR | not started | — | — |
| 2 | Noise Lab | not started | — | — |
| 3 | Axis 1 — Preprocessing | not started | — | — |
| 4 | Axis 2 — LoRA | not started | — | — |
| 5 | Axis 3 — Keydetector | not started | — | — |
| 6 | Flywheel | not started | — | — |
| 7 | Hardening & Ops | not started | — | — |

Status values: `not started` → `in progress` → `blocked (see DECISIONS)` → `done`.

## Current focus

**Phase 0 — Foundations (in progress).** Scaffold complete and CI-green: `make lint` clean, `make test` = 37 passed. Phase-0 gate (`make gate PHASE=0`) = 4 passed, 2 skipped (dataset/corpus tests skip cleanly until the local heavy path runs). Delivered: pyproject/Makefile/docker/CI, `configs/` (default + noise taxonomy + demo menu + confusion seed), `src/ars/{config,storage,contracts,eval.normalize}` + `noise_lab.taxonomy`, `scripts/{download_datasets,tts_corpus,make_fixtures,audio_io}`, unit + acceptance tests. TTS generator verified via `--dry-run`: 500 utts/lang, 100% confusion-target coverage both langs.

Remaining before phase 0 → done (local heavy path, needs sign-off — big downloads + Piper):
1. `make download-data` — OpenSLR es series + LibriSpeech + DEMAND/MUSAN/UrbanSound8K (several GB). Downloaders are written but **untested against live network** (archive layouts / Zenodo URLs to confirm).
2. Install Piper + **verify voice names** in `scripts/tts_corpus.py::VOICES` against the real Piper catalog (some are best-guess), then `make tts-corpus` and `make fixtures`.
3. Re-run `make gate PHASE=0` with datasets present (all 6 green) and paste evidence here + flip the row to `done`.

## Key numbers (fill as they exist)

| Metric | es | en | Source run |
|--------|----|----|------------|
| Baseline clean WER | — | — | — |
| Top-3 damaging subtypes (NDI) | — | — | — |
| Best candidate noisy WER improvement | — | — | — |
| KER with keydetector (confusion set) | — | — | — |
