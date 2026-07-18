# STATUS — Build Progress Tracker

> Maintained by the builder. One row per phase; update the row **when the phase's exit checklist is fully green**, never before. Evidence = paths to reports + key metric numbers (must match the report files).

| Phase | Name | Status | Closed on | Gate evidence |
|-------|------|--------|-----------|---------------|
| 0 | Foundations | done | 2026-07-17 | `make gate PHASE=0` = **6 passed**; `make test` = 37 passed, ruff clean. Data: es 3.0 h / 6 accents (ar,cl,co,pe,pr,ve @0.5 h), en 2.5 h (LibriSpeech); TTS 500+500 utts, 100% confusion-target coverage; DEMAND staged. (MUSAN/UrbanSound8K deferred → DECISIONS.) |
| 1 | Baseline ASR | done | 2026-07-17 | `make gate PHASE=1` = **16 passed**; `make test` = 67 passed, ruff clean. Baseline (whisper-small int8, model 0.1.0): clean WER es **0.062** / en **0.029**; menu-TTS KER es 0.416 / en 0.036. Reports `reports/baseline/baseline-{es,en}.json`. Registry `0.1.0` production. Slow latency companion `test_latency_budget` passed (RTF ≤ 0.6, whisper-small int8). |
| 2 | Noise Lab | done | 2026-07-18 | `make gate PHASE=2` = **20 passed**; `make test` = 89, ruff clean. Noise bank 1670 clips / 7 subtypes (DEMAND + UrbanSound8K); eval-matrix es+en 1320 rows each (SNR ±0.5 dB, 100%). Sensitivity run `run-20260718T010058Z-0.1.0`. **Top-3 NDI (both langs): BB, CA, BC.** NDI monotonic across levels; no warnings. (CB music deferred → DECISIONS.) |
| 3 | Axis 1 — Preprocessing | not started | — | — |
| 4 | Axis 2 — LoRA | not started | — | — |
| 5 | Axis 3 — Keydetector | not started | — | — |
| 6 | Flywheel | not started | — | — |
| 7 | Hardening & Ops | not started | — | — |

Status values: `not started` → `in progress` → `blocked (see DECISIONS)` → `done`.

## Current focus

**Phase 0 — Foundations: DONE (2026-07-17).** Exit checklist green: `make gate PHASE=0` = 6 passed with data present; `make test` = 37 passed; ruff clean. Bootstrapped locally — clean speech `data/clean/{es,en}` (es 3.0 h balanced across 6 LatAm accents, en 2.5 h LibriSpeech), TTS domain corpus `data/datasets/tts-{es,en}-v1` (500 utts/lang, 100% confusion-target coverage, 6 real Piper voices), DEMAND noise staged under `data/_staging/noise_sources/`. torch pinned to 2.11.x to match torchaudio (ABI); MUSAN + UrbanSound8K deferred to phase 2 (disk) — see DECISIONS.

**Phase 1 — Baseline ASR: DONE (2026-07-17).** VAD (Silero) → Faster-Whisper (whisper-small int8) → hallucination guard → keydetector stub → telemetry; FastAPI service; eval harness (WER/CER/KER/hallucination). Registry `0.1.0` production. Frozen baselines in `reports/baseline/`. Clean read-speech WER is healthy (es 6.2% / en 2.9%); the menu-order gap is large in es (KER 41.6%) and small in en (KER 3.6%) — es domain adaptation is the primary flywheel opportunity. Fixed an `utterance_id` collision bug that had inflated es WER to 49% (see DECISIONS). Baselines run at `--limit 450` per dataset (n_utts ≥ 200/lang for the gate).

**Phase 2 — Noise Lab: DONE (2026-07-18).** Noise bank curated (7 subtypes, 1670 clips, recording-level split, DEMAND + UrbanSound8K), deterministic SNR mixer (speech-active RMS, ±0.5 dB), eval-matrix corpora (es+en, 1320 rows each), sensitivity run for model 0.1.0 → NDI + heatmaps + ANALYSIS. **NDI top-3 (both languages): BB construction, CA dining-babble, BC car-cabin.** Babble outranks stationary as theory predicts; family B (drive-thru) dominates, matching the deployment prior. NDI is strictly monotonic across levels (mixer validated). CB music deferred (needs MUSAN); AA/AB are DKITCHEN time-range proxies. This NDI ranking steers phases 3–4.

**Next: Phase 3 — Axis 1 (Preprocessing).** Noise classifier + targeted mitigation registry + auto-generated `configs/mitigation_policy.yaml`; measured WER improvement on top-damage cells (BB/CA/BC) with zero harm on clean audio. Read `plan/phases/phase-3-axis1-preprocessing.md`. Phase 3 will add a `--with-preprocess` flag to the sensitivity harness to re-measure cells post-mitigation. MUSAN/CB music still deferred; `make fixtures` still optional.

## Key numbers (fill as they exist)

| Metric | es | en | Source run |
|--------|----|----|------------|
| Baseline clean WER | 0.062 | 0.029 | run-20260717T205131Z-0.1.0 (es), run-20260717T211306Z-0.1.0 (en) |
| Baseline menu-TTS KER | 0.416 | 0.036 | run-20260717T210321Z-0.1.0 (es), run-20260717T212437Z-0.1.0 (en) |
| Top-3 damaging subtypes (NDI) | — | — | — |
| Best candidate noisy WER improvement | — | — | — |
| KER with keydetector (confusion set) | — | — | — |
