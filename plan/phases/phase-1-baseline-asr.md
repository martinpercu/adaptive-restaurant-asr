# Phase 1 — Baseline ASR Service

**Goal:** a working transcription service (VAD → Faster-Whisper → guards) with telemetry, plus the evaluation harness, and the **baseline metrics report** that every later phase is measured against.
**Depends on:** phase 0 gate. **Estimated effort:** 4–6 days.

## Deliverables

- `src/ars/vad/` — Silero VAD wrapper returning `VadResult`.
- `src/ars/asr/engine.py` — `WhisperEngine` on faster-whisper; `asr/guard.py` — hallucination guard; `asr/prompt_builder.py` — menu → `initial_prompt`.
- `src/ars/api/` — FastAPI service per [02 §3](../02-architecture.md); JSONL telemetry per [03 §9](../03-data-spec.md); optional audio persistence via `ingest`.
- `src/ars/eval/` — `normalize`, WER/CER/KER/hallucination metrics, `python -m ars.eval.run --dataset <id> --model-version <v>` writing `metric_runs` + `reports/eval/<run_id>.json`.
- `reports/baseline/baseline-es.json`, `baseline-en.json` — clean-speech baseline for the production candidate model (whisper-small int8 to start; record exact model in the report).
- Model registry initialized: `models/registry.json` with entry `0.1.0`, `stage: production`, `base_model: whisper-small`, no adapter.

## Tasks

### 1.1 VAD
Silero VAD (pip `silero-vad`, torch backend). `detect(audio, sr) -> VadResult`. Config: thresholds, min segment 250 ms, merge gap 100 ms. Compute `speech_rms` over active frames (needed by the phase-2 mixer — same function, single implementation).

### 1.2 Engine + guards
Per [02 §4](../02-architecture.md). Constructor takes a registry entry (model path/size, compute type). `transcribe(audio, sr, language=None, initial_prompt=None) -> RawTranscript`. Language clamped to {es,en} per [01 §2](../01-conventions.md). Guards exactly as specced; each guard adds its flag.

### 1.3 API + telemetry
Wire VAD → (preprocess pass-through stub honoring `PreprocessReport`) → engine → (keydetector pass-through stub) → response. Telemetry line per request. `store_audio: true` persists WAV + sidecar + `utterances` row.

### 1.4 Eval harness
`ars.eval`: implement [03 §6](../03-data-spec.md) exactly. Runner loads a dataset manifest, transcribes with a given registry version, computes per-lang WER/CER/KER/hallucination rate, writes report JSON + DB row. Must accept `--limit N` and `--seed` for smoke runs.

### 1.5 Baseline run
Run eval on the clean TTS corpus + public clean sets (`eval-clean-es-v1`, `eval-clean-en-v1` — build these dataset manifests from held-out phase-0 data; hold-out split by speaker/voice, seed 1337). Save baseline reports. These numbers are frozen: later gates compare against them by run_id reference, never by re-running.

## Acceptance tests (`tests/acceptance/test_phase1_*.py`)

1. `test_api_contract` — POST WAV fixture → 200, response validates as `FinalTranscript`, trace has all latency keys.
2. `test_pure_noise_returns_empty` — white-noise fixture (no speech) → `text == ""`, `low_speech_gated` flag, **no** ASR invocation (assert via engine spy).
3. `test_repetition_guard` — feed a synthetic `RawTranscript` with a looping 3-gram to guard → truncated + flagged.
4. `test_language_clamp` — engine never returns a language outside {es,en} (parametrize with fixtures).
5. `test_wer_harness_known_values` — hand-computed toy set (5 ref/hyp pairs) → exact WER/CER/KER values (assert to 3 decimals).
6. `test_normalize_rules` — table-driven cases for both languages per [03 §6](../03-data-spec.md).
7. `test_baseline_reports_exist_and_valid` — reports parse, contain both langs, `n_utts ≥ 200` per lang.
8. (`slow`, local) `test_latency_budget` — 5 s fixture end-to-end RTF ≤ 0.6 on whisper-small int8.

CI note: engine tests in CI run `whisper-tiny`; the baseline report task uses the real candidate size locally.

## Exit checklist
- [ ] `make gate PHASE=1` green (CI subset green too).
- [ ] Baseline WER recorded for es and en in `reports/baseline/` and referenced in STATUS.md (expect roughly 10–25% WER on clean TTS/read speech; if > 40%, something is broken — investigate before proceeding).
- [ ] `models/registry.json` has the `0.1.0` production entry.

## Pitfalls
- `condition_on_previous_text=False` is mandatory — leaving it on causes cross-segment hallucination loops.
- Do not resample inside the engine; ingest guarantees 16 kHz. Assert it.
- jiwer ≥ 3 changed its API (`process_words`); write one wrapper, not scattered calls.
