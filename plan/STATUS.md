# STATUS — Build Progress Tracker

> Maintained by the builder. One row per phase; update the row **when the phase's exit checklist is fully green**, never before. Evidence = paths to reports + key metric numbers (must match the report files).

| Phase | Name | Status | Closed on | Gate evidence |
|-------|------|--------|-----------|---------------|
| 0 | Foundations | done | 2026-07-17 | `make gate PHASE=0` = **6 passed**; `make test` = 37 passed, ruff clean. Data: es 3.0 h / 6 accents (ar,cl,co,pe,pr,ve @0.5 h), en 2.5 h (LibriSpeech); TTS 500+500 utts, 100% confusion-target coverage; DEMAND staged. (MUSAN/UrbanSound8K deferred → DECISIONS.) |
| 1 | Baseline ASR | done | 2026-07-17 | `make gate PHASE=1` = **16 passed**; `make test` = 67 passed, ruff clean. Baseline (whisper-small int8, model 0.1.0): clean WER es **0.062** / en **0.029**; menu-TTS KER es 0.416 / en 0.036. Reports `reports/baseline/baseline-{es,en}.json`. Registry `0.1.0` production. Slow latency companion `test_latency_budget` passed (RTF ≤ 0.6, whisper-small int8). |
| 2 | Noise Lab | not started | — | — |
| 3 | Axis 1 — Preprocessing | not started | — | — |
| 4 | Axis 2 — LoRA | not started | — | — |
| 5 | Axis 3 — Keydetector | not started | — | — |
| 6 | Flywheel | not started | — | — |
| 7 | Hardening & Ops | not started | — | — |

Status values: `not started` → `in progress` → `blocked (see DECISIONS)` → `done`.

## Current focus

**Phase 0 — Foundations: DONE (2026-07-17).** Exit checklist green: `make gate PHASE=0` = 6 passed with data present; `make test` = 37 passed; ruff clean. Bootstrapped locally — clean speech `data/clean/{es,en}` (es 3.0 h balanced across 6 LatAm accents, en 2.5 h LibriSpeech), TTS domain corpus `data/datasets/tts-{es,en}-v1` (500 utts/lang, 100% confusion-target coverage, 6 real Piper voices), DEMAND noise staged under `data/_staging/noise_sources/`. torch pinned to 2.11.x to match torchaudio (ABI); MUSAN + UrbanSound8K deferred to phase 2 (disk) — see DECISIONS.

**Phase 1 — Baseline ASR: DONE (2026-07-17).** VAD (Silero) → Faster-Whisper (whisper-small int8) → hallucination guard → keydetector stub → telemetry; FastAPI service; eval harness (WER/CER/KER/hallucination). Registry `0.1.0` production. Frozen baselines in `reports/baseline/`. Clean read-speech WER is healthy (es 6.2% / en 2.9%); the menu-order gap is large in es (KER 41.6%) and small in en (KER 3.6%) — es domain adaptation is the primary flywheel opportunity. Fixed an `utterance_id` collision bug that had inflated es WER to 49% (see DECISIONS). Baselines run at `--limit 450` per dataset (n_utts ≥ 200/lang for the gate).

**Next: Phase 2 — Noise Lab.** Noise taxonomy is already seeded; build the noise-bank curation, deterministic SNR mixer (uses `ars.vad.speech_active_rms`), eval-matrix corpus builder, sensitivity run, and NDI ranking. Read `plan/phases/phase-2-noise-lab.md`. Note: MUSAN + UrbanSound8K still need downloading for the outdoor/babble subtypes (deferred in phase 0 → DECISIONS). `make fixtures` (small WER fixtures) still optional/pending.

## Key numbers (fill as they exist)

| Metric | es | en | Source run |
|--------|----|----|------------|
| Baseline clean WER | 0.062 | 0.029 | run-20260717T205131Z-0.1.0 (es), run-20260717T211306Z-0.1.0 (en) |
| Baseline menu-TTS KER | 0.416 | 0.036 | run-20260717T210321Z-0.1.0 (es), run-20260717T212437Z-0.1.0 (en) |
| Top-3 damaging subtypes (NDI) | — | — | — |
| Best candidate noisy WER improvement | — | — | — |
| KER with keydetector (confusion set) | — | — | — |
