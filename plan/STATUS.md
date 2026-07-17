# STATUS — Build Progress Tracker

> Maintained by the builder. One row per phase; update the row **when the phase's exit checklist is fully green**, never before. Evidence = paths to reports + key metric numbers (must match the report files).

| Phase | Name | Status | Closed on | Gate evidence |
|-------|------|--------|-----------|---------------|
| 0 | Foundations | done | 2026-07-17 | `make gate PHASE=0` = **6 passed**; `make test` = 37 passed, ruff clean. Data: es 3.0 h / 6 accents (ar,cl,co,pe,pr,ve @0.5 h), en 2.5 h (LibriSpeech); TTS 500+500 utts, 100% confusion-target coverage; DEMAND staged. (MUSAN/UrbanSound8K deferred → DECISIONS.) |
| 1 | Baseline ASR | not started | — | — |
| 2 | Noise Lab | not started | — | — |
| 3 | Axis 1 — Preprocessing | not started | — | — |
| 4 | Axis 2 — LoRA | not started | — | — |
| 5 | Axis 3 — Keydetector | not started | — | — |
| 6 | Flywheel | not started | — | — |
| 7 | Hardening & Ops | not started | — | — |

Status values: `not started` → `in progress` → `blocked (see DECISIONS)` → `done`.

## Current focus

**Phase 0 — Foundations: DONE (2026-07-17).** Exit checklist green: `make gate PHASE=0` = 6 passed with data present; `make test` = 37 passed; ruff clean. Bootstrapped locally — clean speech `data/clean/{es,en}` (es 3.0 h balanced across 6 LatAm accents, en 2.5 h LibriSpeech), TTS domain corpus `data/datasets/tts-{es,en}-v1` (500 utts/lang, 100% confusion-target coverage, 6 real Piper voices), DEMAND noise staged under `data/_staging/noise_sources/`. torch pinned to 2.11.x to match torchaudio (ABI); MUSAN + UrbanSound8K deferred to phase 2 (disk) — see DECISIONS.

**Next: Phase 1 — Baseline ASR.** VAD (Silero) + Faster-Whisper service, eval harness, telemetry; deliver baseline WER reports (es+en) and anti-hallucination tests. Read `plan/phases/phase-1-baseline-asr.md` before starting. `make fixtures` (WER fixtures) still pending — generate when phase 1 needs real-audio WER tests.

## Key numbers (fill as they exist)

| Metric | es | en | Source run |
|--------|----|----|------------|
| Baseline clean WER | — | — | — |
| Top-3 damaging subtypes (NDI) | — | — | — |
| Best candidate noisy WER improvement | — | — | — |
| KER with keydetector (confusion set) | — | — | — |
