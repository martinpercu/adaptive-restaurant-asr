# STATUS — Build Progress Tracker

> Maintained by the builder. One row per phase; update the row **when the phase's exit checklist is fully green**, never before. Evidence = paths to reports + key metric numbers (must match the report files).

| Phase | Name | Status | Closed on | Gate evidence |
|-------|------|--------|-----------|---------------|
| 0 | Foundations | done | 2026-07-17 | `make gate PHASE=0` = **6 passed**; `make test` = 37 passed, ruff clean. Data: es 3.0 h / 6 accents (ar,cl,co,pe,pr,ve @0.5 h), en 2.5 h (LibriSpeech); TTS 500+500 utts, 100% confusion-target coverage; DEMAND staged. (MUSAN/UrbanSound8K deferred → DECISIONS.) |
| 1 | Baseline ASR | done | 2026-07-17 | `make gate PHASE=1` = **16 passed**; `make test` = 67 passed, ruff clean. Baseline (whisper-small int8, model 0.1.0): clean WER es **0.062** / en **0.029**; menu-TTS KER es 0.416 / en 0.036. Reports `reports/baseline/baseline-{es,en}.json`. Registry `0.1.0` production. Slow latency companion `test_latency_budget` passed (RTF ≤ 0.6, whisper-small int8). |
| 2 | Noise Lab | done | 2026-07-18 | `make gate PHASE=2` = **20 passed**; `make test` = 89, ruff clean. Noise bank 1670 clips / 7 subtypes (DEMAND + UrbanSound8K); eval-matrix es+en 1320 rows each (SNR ±0.5 dB, 100%). Sensitivity run `run-20260718T010058Z-0.1.0`. **Top-3 NDI (both langs): BB, CA, BC.** NDI monotonic across levels; no warnings. (CB music deferred → DECISIONS.) |
| 3 | Axis 1 — Preprocessing | done | 2026-07-18 | `make gate PHASE=3` = **11 passed, 1 xfailed** (test 8, CPU denoisers over budget → DECISIONS); `make test` = 93, ruff clean. Classifier `0.1.0`: clean_recall **1.0** (hard gate ✓), subtype F1 0.72 / family F1 0.84 (proxy ceiling, superseded → phase 7). Policy generated = **all `none`** (spectral_gate hurt: clean WER es 0.36→0.56; residual = all 7 subtypes → phase 4). |
| 4 | Axis 2 — LoRA | not started | — | — |
| 5 | Axis 3 — Keydetector | not started | — | — |
| 6 | Flywheel | not started | — | — |
| 7 | Hardening & Ops | not started | — | — |

Status values: `not started` → `in progress` → `blocked (see DECISIONS)` → `done`.

## Current focus

**Phase 0 — Foundations: DONE (2026-07-17).** Exit checklist green: `make gate PHASE=0` = 6 passed with data present; `make test` = 37 passed; ruff clean. Bootstrapped locally — clean speech `data/clean/{es,en}` (es 3.0 h balanced across 6 LatAm accents, en 2.5 h LibriSpeech), TTS domain corpus `data/datasets/tts-{es,en}-v1` (500 utts/lang, 100% confusion-target coverage, 6 real Piper voices), DEMAND noise staged under `data/_staging/noise_sources/`. torch pinned to 2.11.x to match torchaudio (ABI); MUSAN + UrbanSound8K deferred to phase 2 (disk) — see DECISIONS.

**Phase 1 — Baseline ASR: DONE (2026-07-17).** VAD (Silero) → Faster-Whisper (whisper-small int8) → hallucination guard → keydetector stub → telemetry; FastAPI service; eval harness (WER/CER/KER/hallucination). Registry `0.1.0` production. Frozen baselines in `reports/baseline/`. Clean read-speech WER is healthy (es 6.2% / en 2.9%); the menu-order gap is large in es (KER 41.6%) and small in en (KER 3.6%) — es domain adaptation is the primary flywheel opportunity. Fixed an `utterance_id` collision bug that had inflated es WER to 49% (see DECISIONS). Baselines run at `--limit 450` per dataset (n_utts ≥ 200/lang for the gate).

**Phase 3 — Axis 1 (Preprocessing): DONE (2026-07-18).** Noise classifier (log-mel CNN, 0.9 MB), mitigation-chain registry (none/spectral_gate/deepfilternet/demucs), derived-policy generation + runtime `PolicyPreprocessor` (off/log_only/active), wired into the API. The classifier's hard gate — **clean_recall = 1.0** (clean audio never triggers mitigation) — passes; subtype/family F1 plateau on the proxy noise bank (AA≈AB, BC→BA) and are superseded to phase-7 real audio. Measured effectiveness (spectral_gate vs none, es+en, 7 subtypes) → **honest all-`none` policy**: no CPU-viable chain helps and spectral_gate harms clean/ASR (trust ΔWER, not "sounds cleaner"). All residual noise damage is now phase-4's target. test 8 (≥5% top-3 gain) xfailed: neural denoisers exceed the 400 ms CPU budget — superseded by phase-4's ≥15% noisy-WER gate (DECISIONS 2026-07-18). deepfilternet needs a Rust toolchain (unavailable here).

**Phase 2 — Noise Lab: DONE (2026-07-18).** Noise bank curated (7 subtypes, 1670 clips, recording-level split, DEMAND + UrbanSound8K), deterministic SNR mixer (speech-active RMS, ±0.5 dB), eval-matrix corpora (es+en, 1320 rows each), sensitivity run for model 0.1.0 → NDI + heatmaps + ANALYSIS. **NDI top-3 (both languages): BB construction, CA dining-babble, BC car-cabin.** Babble outranks stationary as theory predicts; family B (drive-thru) dominates, matching the deployment prior. NDI is strictly monotonic across levels (mixer validated). CB music deferred (needs MUSAN); AA/AB are DKITCHEN time-range proxies. This NDI ranking steers phases 3–4.

**Next: Phase 4 — Axis 2 (LoRA).** Damage-weighted synthetic-noise training (weights ∝ phase-2 NDI: BB/CA/BC hardest), LoRA adapters over the Whisper base, CT2 export, anti-catastrophic-forgetting regression suite. Gate: ≥15% relative noisy-WER improvement both langs, no clean regression. This is now the named mechanism for the phase-3 residual (all 7 subtypes — no preprocessing helped). Read `plan/phases/phase-4-axis2-lora.md`. Note: LoRA training needs the GPU path (cloud); on CPU expect only a smoke-scale run. MUSAN/CB music still deferred; `make fixtures` still optional.

## Key numbers (fill as they exist)

| Metric | es | en | Source run |
|--------|----|----|------------|
| Baseline clean WER | 0.062 | 0.029 | run-20260717T205131Z-0.1.0 (es), run-20260717T211306Z-0.1.0 (en) |
| Baseline menu-TTS KER | 0.416 | 0.036 | run-20260717T210321Z-0.1.0 (es), run-20260717T212437Z-0.1.0 (en) |
| Top-3 damaging subtypes (NDI) | — | — | — |
| Best candidate noisy WER improvement | — | — | — |
| KER with keydetector (confusion set) | — | — | — |
