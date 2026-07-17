# Phase 4 — Axis 2: Damage-Weighted LoRA Fine-Tuning

**Goal:** the "model itself" axis: PEFT/LoRA training whose data sampling is weighted by the NDI ranking, guarded by anti-forgetting regression, exported to CTranslate2, and registered per the model registry contract. This phase builds the *repeatable training pipeline*; phase 6 automates its triggering.
**Depends on:** phase 3 gate (NDI + effectiveness reports; residual-damage subtypes are the priority). **Estimated effort:** 7–10 days. **Hardware:** two documented paths — cloud GPU (RunPod/Lambda: ≥ 24 GB VRAM for whisper-medium LoRA) or local NVIDIA GPU; iteration runs use whisper-small (≥ 12 GB). All scripts device-agnostic (`accelerate`).

## Deliverables

- `src/ars/training/dataset_builder.py` — damage-weighted noisy training set generator.
- `src/ars/training/train_lora.py` — HF Transformers + PEFT training loop.
- `src/ars/training/regression.py` — anti-forgetting suite.
- `src/ars/training/export_ct2.py` — merge + CTranslate2 conversion + parity check.
- `docker/Dockerfile.gpu` + `docs section` in this phase for cloud provisioning/sync.
- Datasets `train-noisy-xx-v1`, `eval-noisy-es-v1`, `eval-noisy-en-v1`.
- First promoted adapter: registry entry `0.2.0` (whisper-small + LoRA) — or documented failure analysis.

## Tasks

### 4.1 Training dataset builder
`build --size-hours 20 --ndi reports/sensitivity/<run>/ndi.json --seed 1337`:

- Source clean pool: phase-0 clean speech + TTS domain corpus (train split), both langs (≈ 50/50).
- Composition: **30% clean** (anti-forgetting floor) + 70% mixed.
- Mixed sampling: subtype drawn from `softmax(NDI / T)` with `T=0.5` (config); level drawn uniformly from a *continuous* SNR range per level band (05: +8..+15 dB, 10: −2..+2 dB, 15: −7..−3 dB) — continuous SNR prevents overfitting to three exact ratios. **Train-split noise clips only.**
- If phase 3 left residual-damage subtypes (no effective mitigation), multiply their NDI weight ×1.5 before softmax (config knob) — the model must compensate where preprocessing can't.
- Manifest per [03 §1](../03-data-spec.md) with full mix provenance. Builder is deterministic given (config, NDI file, seed).

### 4.2 Eval sets
`eval-noisy-<lang>-v1`: same construction as the phase-2 matrix but from a *disjoint* clean hold-out and eval-split noise, sized ≥ 40 utts × 7 subtypes × 3 levels per lang. Frozen for the life of the gate (new versions require DECISIONS.md entry).

### 4.3 LoRA training
- Base: `openai/whisper-small` for iteration; `whisper-medium` once the recipe passes gates on small.
- PEFT config: LoRA `r=32, alpha=64, dropout=0.05`, target modules `q_proj,v_proj` (encoder+decoder attention). Only adapter weights trainable.
- Recipe: bf16 (fp16 fallback), lr 5e-4 cosine with 10% warmup, effective batch ≥ 32 via grad accumulation, 3–5 epochs, eval every half-epoch on a 200-utt dev slice, early stop on dev WER (patience 3). SpecAugment via the HF feature extractor defaults.
- Bilingual: single multilingual base, mixed-language batches; loss unchanged; language token from each sample's `lang`.
- Outputs: `models/adapters/lora-<ISOweek>/` with adapter weights + `training_meta.json` (dataset id, NDI run, config hash, seed, curves).

### 4.4 Regression suite (anti-catastrophic-forgetting)
Fixed corpus `regression-keywords-v1`: TTS renders (clean, 3 voices/lang) of (a) every menu item in `configs/menu/demo.yaml`, (b) every restaurant-side word in the idea.txt confusion tables, (c) 50 generic non-domain sentences per lang. Metric: keyword recall + clean WER. Gate: keyword recall drop ≤ 1% absolute vs base model; generic clean WER regression ≤ 2% relative.

### 4.5 Export + parity
Merge adapter → full model → `ct2-transformers-converter` (quantize int8_float16). Parity: transcribe 20 eval utterances with HF (greedy) and CT2 (greedy); WER-vs-each-other ≤ 1.0 absolute point. Registry entry written with `stage: "candidate"`, gates block filled per [03 §5](../03-data-spec.md).

### 4.6 Gate evaluation
Runner compares candidate vs baseline **on identical datasets, preprocessing off** (isolate the axis):
- `noisy_wer_rel_improvement ≥ 0.15` per lang (eval-noisy sets) — the idea.txt promotion bar.
- `clean_wer_rel_regression ≤ 0.02` per lang.
- Regression suite (4.4) passes.
All three → flip entry to `stage: "production"` (manual in this phase; automated in phase 6), bump minor version.

### Cloud path notes
Provisioning script uploads: repo (git archive), `train-noisy` dataset (tar via rsync/s3), configs. Returns: adapter dir + meta. Keep a `make train-remote` target wrapping it. Never train on data that isn't in a manifest.

## Acceptance tests (`tests/acceptance/test_phase4_*.py`)

1. `test_dataset_builder_composition` — built manifest: 30% ±2% clean; subtype histogram matches softmax(NDI/T) within χ² tolerance; SNRs inside level bands; zero eval-split noise clips; deterministic (same seed → same manifest hash).
2. `test_dataset_builder_residual_boost` — hand-built NDI + residual list → boosted subtype sampled ×~1.5 more.
3. `test_lora_smoke` (`gpu` or CPU-tiny) — 20 steps on whisper-tiny + 50-utt dataset: loss decreases, adapter saves/loads, only LoRA params have grads.
4. `test_regression_suite_computation` — toy hyp/ref fixtures → exact recall numbers; gate logic table-driven (pass/fail cases).
5. `test_ct2_parity` (`slow`) — parity ≤ 1.0 WER point on the 20-sample set.
6. `test_registry_candidate_entry` — entry schema-valid, gates block complete, exactly one production entry remains.
7. `test_gate_evaluation_logic` — fake metric inputs → promote / reject decisions exactly per 4.6 (including one-lang-passes → reject).
8. (`slow`, the real gate) `test_candidate_beats_baseline` — actual candidate metrics meet 4.6 on both langs.

## Exit checklist
- [ ] `make gate PHASE=4` green; test 8 evidence in STATUS.md (table: baseline vs candidate per lang, clean and noisy).
- [ ] Post-train sensitivity re-run (`noise_lab.sensitivity` on candidate) saved — the NDI delta shows *which* subtypes the fine-tune fixed; attach to ANALYSIS.
- [ ] Registry has promoted `0.2.0`; rollback command documented and tested (`python -m ars.registry rollback`).

## Pitfalls
- Whisper prompt/token plumbing: ensure the language token matches each sample; a fixed `es` token on `en` batches silently trains garbage.
- Don't eval with the keydetector or preprocessing on — this gate isolates axis 2.
- LoRA on fp16 base with lr 1e-3 diverges often; 5e-4 + warmup is the stable default.
- If 15% relative improvement isn't reached on whisper-small: first check mixer SNR distribution and language-token bug above, then raise dataset hours to 40, then move to medium — in that order. Document each attempt in DECISIONS.md.
