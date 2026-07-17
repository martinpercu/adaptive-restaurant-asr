# Phase 2 — Noise Lab (Sensitivity Diagnostics)

**Goal:** the diagnostic engine that answers *"which noise, at which level, hurts the model most?"*: curated noise bank, deterministic SNR mixer, matrix corpus builder, sensitivity runner, and the first **Noise Damage Index (NDI)** report for the baseline model. This report is the steering input for phases 3, 4 and 6.
**Depends on:** phase 1 gate. **Estimated effort:** 5–7 days.

## Deliverables

- `src/ars/noise_lab/curate.py` — staging → noise bank with taxonomy labels.
- `src/ars/noise_lab/mixer.py` — SNR-accurate mixing.
- `src/ars/noise_lab/build_corpus.py` — clean × subtype × level matrix datasets.
- `src/ars/noise_lab/sensitivity.py` — full-matrix evaluation + NDI + heatmaps.
- `data/noise_bank/` populated + manifest ([03 §2](../03-data-spec.md)).
- Datasets `eval-matrix-es-v1`, `eval-matrix-en-v1`.
- `reports/sensitivity/<run_id>/` for model `0.1.0` ([03 §3](../03-data-spec.md)).

## Tasks

### 2.1 Noise bank curation
`python -m ars.noise_lab.curate --source demand --subtype AB ...`: cut staged source audio into clips of 10–30 s, resample to canonical format, assign `clip_id`, subtype, license, and `split` (train/eval, **split by source recording**, never by clip — leakage guard). Initial mapping (extend as needed, record in DECISIONS.md):

| Taxonomy | Sources |
|----------|---------|
| AA dishes-cutlery | DEMAND DKITCHEN segments with transient clatter; UrbanSound8K where applicable |
| AB fryer-extractor | DEMAND DKITCHEN stationary segments (hum/sizzle) |
| AC kitchen-babble | MUSAN babble (dense, few speakers) |
| BA drive-thru-traffic | UrbanSound8K engine_idling + street_music-free traffic; DEMAND TMETRO as proxy |
| BB construction | UrbanSound8K jackhammer + drilling |
| BC car-cabin | **constructed composites**: MUSAN speech (1–2 sparse speakers) layered over UrbanSound8K engine_idling at −5..0 dB relative; the curation script generates these as first-class bank clips |
| CA dining-babble | MUSAN babble (sparse) + DEMAND PCAFETER |
| CB music | MUSAN music |

Where a source mixes characters (DKITCHEN has both hum and clatter), curation may hand-pick segment time-ranges; record ranges in a `curation.yaml` so the step is reproducible. Minimum per subtype: 10 clips train + 4 clips eval.

### 2.2 SNR mixer
`mix(clean, noise, snr_db, seed) -> (mixed, achieved_snr_db, gain)`:

1. `speech_rms` = RMS over VAD-active frames of `clean` (reuse phase-1 VAD, cache per clean utterance).
2. Pick a random (seeded) noise window of `len(clean)`; loop-pad with crossfade if the clip is shorter.
3. `gain = speech_rms / (noise_rms · 10^(snr_db/20))`; `mixed = clean + gain·noise_window`.
4. If `max(|mixed|) > 0.99`: scale the **whole mix** down to peak 0.99 (preserves SNR); record `peak_scaled`.
5. Recompute achieved SNR from the components actually used; must be within ±0.5 dB of target.

Pure function of `(clean_id, clip_id, snr_db, seed)` — identical inputs give bit-identical output.

### 2.3 Matrix corpus builder
`build_corpus --langs es en --clean-per-lang 60 --levels 05 10 15`: select 60 held-out clean utterances per lang (seeded, keyword-bearing utterances preferred: ≥ 80% must contain ≥ 1 domain keyword), cross with every subtype × canonical level using **eval-split** noise clips (round-robin). Output per lang: 60 clean rows + 60 × 7 × 3 = 1260 mixed rows, manifest per [03 §1](../03-data-spec.md). Runtime sanity: ~2.5 k files, few minutes.

### 2.4 Sensitivity runner
`sensitivity --dataset eval-matrix-es-v1 --model-version 0.1.0`: evaluate every cell with the phase-1 harness (**preprocessing disabled** — this measures raw model sensitivity; a flag `--with-preprocess` exists for phase 3), aggregate per `(subtype, level)`, compute NDI per [03 §3](../03-data-spec.md), emit `matrix.parquet`, `ndi.json`, `heatmap-{lang}.png`. Console summary: top-3 damaging subtypes per lang with ΔWER.

### 2.5 Analysis note
Write `reports/sensitivity/<run_id>/ANALYSIS.md` (generated skeleton + human/LLM-filled): which subtypes dominate, whether babble (AC/CA) outranks stationary (AB) as theory predicts, ceiling effects observed, recommendation for phase-3 targets (top-3 subtypes) and phase-4 sampling weights. Since the primary deployment channel is the drive-thru, sanity-check the ranking against that prior: family B plus BC (car-cabin voices) are expected to dominate in production even if dining-room subtypes score high on public-proxy audio.

## Acceptance tests (`tests/acceptance/test_phase2_*.py`)

1. `test_mixer_snr_accuracy` — for synthetic speech-like fixture + each of {white, pink, babble-like} fixture noise at SNR {+10, 0, −5, −10}: `abs(achieved − target) ≤ 0.5` dB.
2. `test_mixer_determinism` — same inputs+seed twice → identical arrays; different seed → different noise window.
3. `test_mixer_clipping_guard` — construct a case that would clip → peak ≤ 0.99 **and** achieved SNR still within tolerance.
4. `test_noise_bank_manifest` — all subtypes in taxonomy, licenses present, no source recording spans train and eval splits, every subtype ≥ min clips.
5. `test_matrix_corpus_complete` — manifest has exactly the expected cell counts per (subtype, level); all files exist; `snr_db_achieved` within tolerance for 100% of rows.
6. `test_ndi_computation` — feed a hand-built `matrix.parquet` with known metric values → exact expected NDI ranking (fixture-based, no model).
7. `test_sensitivity_report_valid` — real report for model 0.1.0 exists, schema-valid, covers full canonical grid for both langs.
8. Monotonicity sanity (warning, not failure): mean WER(15) ≥ mean WER(10) ≥ mean WER(05) per subtype; if violated, the runner prints a prominent warning (usually a mixing bug).

## Exit checklist
- [ ] `make gate PHASE=2` green.
- [ ] NDI report for 0.1.0 exists for both languages; ANALYSIS.md names the top-3 target subtypes.
- [ ] STATUS.md updated with the top-3 ranking (this feeds phases 3–4).

## Pitfalls
- Whole-file RMS instead of speech-active RMS silently shifts every SNR by several dB — the #1 way this phase goes wrong. The mixer must call the same VAD used in production.
- Loop-padding without crossfade adds click transients that behave like AA noise and contaminate other subtypes.
- Never mix with train-split noise clips in eval datasets (and vice versa in phase 4).
