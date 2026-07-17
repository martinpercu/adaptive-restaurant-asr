# ARS — Adaptive Restaurant Speech

A self-improving, bilingual (Spanish + English) speech-to-text system built specifically for restaurant environments: drive-thru lanes, counters, and kitchens. Built for a **drive-thru voice automation company** targeting the **USA + LATAM markets** — which drives the bilingual scope: US English plus a broad Spanish accent spectrum (LatAm dialects and US Spanish speakers). The primary deployment channel is the **drive-thru lane**, which makes outdoor and in-car noise — traffic, idling engines, passengers shouting from the back seat — first-class concerns. This repository currently contains the **complete implementation plan** (under [plan/](plan/)) written so that an LLM builder — or a human team — can construct the system end to end, phase by phase, with acceptance tests gating every step.

> **Building it?** Start with [CLAUDE.md](CLAUDE.md) (builder protocol), then [plan/00-overview.md](plan/00-overview.md).

---

## 1. The problem

Restaurants are among the hardest environments for ASR. The failure is not just "it's loud" — it is two acoustically distinct enemies:

- **Stationary/ambient noise** — extractor hoods, fryers, idling car engines at the drive-thru, cutlery clatter. Broadband, persistent, partially maskable.
- **Babble noise** — other humans talking (kitchen staff, the dining room, the kid in the back seat shouting "¡y papas fritas!"). This is the dangerous one: it *is* speech, so the model can't separate "signal" from "noise" by spectral character alone.

On top of the acoustics, restaurants add domain-specific failure modes:

- **Whisper hallucinations**: fed near-pure noise, Whisper doesn't return silence — it invents text or loops phrases ("gracias por su compra, gracias por su compra…").
- **Out-of-vocabulary menu items**: launch a "MegaCheddar Blast" and the base model hears "me da una chedar de las".
- **Phonetic confusion pairs**: under noise, *cocina* becomes *bocina*, *soup* becomes *soap*, *fries* becomes *flies*. These errors are systematic, not random — which makes them fixable by rule.

## 2. The core idea: a Data Flywheel with a diagnosis engine

A one-shot fine-tune decays as menus, stores, and noise conditions change. ARS is instead a **continuous loop**: production audio → error and noise diagnosis → targeted improvements → gated automatic deployment → better production audio handling → repeat.

Two design principles distinguish ARS from a generic ASR pipeline:

**(a) We always know the domain.** Every request comes from a restaurant. That prior is exploited at every stage: the menu is injected into Whisper's `initial_prompt`, the correction engine assumes food-service vocabulary, the noise classifier only needs to distinguish restaurant noise types.

**(b) We diagnose before we optimize.** The **Noise Lab** measures *which specific noise type at which level* damages the model, and improvements are aimed at the measured top offenders — not at "noise" in general.

## 3. The Noise Lab (sensitivity diagnostics)

The centerpiece. It works like a controlled experiment:

1. **Taxonomy.** Every noise asset is classified: family → subtype → level. Examples: `noise-AA` (kitchen: dishes/cutlery), `noise-AB` (kitchen: fryer/extractor), `noise-AC` (kitchen babble), `noise-BA` (drive-thru traffic), `noise-BB` (construction), `noise-CA` (dining babble), `noise-CB` (music). Levels: `05` / `10` / `15` = SNR +10 / 0 / −5 dB, where **level 10 means the noise is exactly as loud as the clean speech** (the product definition "volumen 10 es idéntico al volumen del audio limpio").
2. **Matrix corpus.** Clean utterances (`cl-es-00042`…) are deterministically mixed with every subtype at every level: `cl-es-00042__noise-AB-15.wav`. Same speech, same noise recording, only the variable under study changes.
3. **Sensitivity run.** Any model version is evaluated over the full matrix, producing per-cell WER, CER, **KER** (keyword error rate over menu terms and known confusion words), and hallucination rate.
4. **Noise Damage Index (NDI).** Cells aggregate into a per-subtype damage ranking. This ranking is the steering wheel of the whole system: it decides which noises get dedicated mitigation, which get oversampled in training, and where confusion-pair mining should look first.

Because the mixer, corpus, and runner are cheap and deterministic, the sensitivity matrix is re-run for every model candidate and after every real-world noise recording batch — the diagnosis stays current as conditions change.

## 4. Three axes of continuous self-improvement

Every improvement cycle acts on three independent, individually-tested axes:

### Axis 1 — Before the model (`preprocess`)
A lightweight noise **classifier** identifies which taxonomy subtype is present in the incoming audio (trained for free on Noise Lab data — including speech+noise mixtures, since that's what production looks like). A **mitigation policy** maps each subtype to a denoising chain: spectral gating for stationary hum, DeepFilterNet for general noise, source separation (Demucs) evaluated for babble. Crucially, the policy is **generated from measurement**: a chain is enabled for a subtype only if it improves WER on that subtype's matrix cells in both languages, within latency budget, without harming clean audio. Denoisers that "sound better" but transcribe worse are automatically rejected.

### Axis 2 — The model (`training`)
PEFT/**LoRA** adapters over a Whisper base (small for iteration, medium as the production target), trained on synthetically-noised data whose sampling is **weighted by the NDI** — the model trains hardest on what hurts it most, and hardest of all on subtypes where axis 1 found no effective mitigation. Guardrails: a fixed anti-catastrophic-forgetting regression corpus (menu terms + confusion words + generic sentences), a ≥15% relative noisy-WER improvement gate per language, a ≤2% clean-WER regression cap, and CTranslate2 export with parity checks for fast int8 inference via Faster-Whisper.

### Axis 3 — After the model (`keydetector`)
A deterministic post-ASR correction engine — the exploitation of "we know it's a restaurant":

- **Menu lexicon pass**: hypothesis n-grams are matched against the store's menu by normalized text, fuzzy ratio, and *phonetic keys* (double metaphone for English; a rule-based Spanish key that folds seseo/yeísmo — `vaso` and `bazo` share a key by design). "mega chedar blas" → "MegaCheddar Blast".
- **Confusion pair rules**: a curated, versioned rule base (`bocina→cocina` near food-context words, `soap→soup` in order context) seeded from linguistic analysis and **grown automatically by the flywheel**. Mined errors become rules, *not* training data: a rule ships in minutes with a mandatory golden test pair (one positive, one negative), needs no GPU, and is instantly reversible.

Over-correction is treated as worse than under-correction: ambiguous rules run in log-only mode until evidence promotes them, a false-correction-rate gate (≤0.5%) protects general speech, and every fired correction is logged with its rule id for audit.

## 5. The autonomous flywheel

Weekly, orchestrated (Prefect), human-in-the-loop only where judgment is genuinely needed:

1. **Harvest** low-confidence production utterances (plus a random control sample) from telemetry.
2. **LLM-as-a-Judge** (pluggable provider — Anthropic `claude-sonnet-5` by default, with OpenAI `gpt-4o-mini`/`gpt-4.1-nano` as calibrated low-cost alternates): each transcript is checked for semantic coherence against the store menu and — when available — the **POS ticket** for the same order (if the customer got a Coca and the ticket says Coca, the transcript core was probably right). Verdicts route items to auto-labeling, confusion-candidate mining, or a human **review queue** (a terminal CLI).
3. **Pair mining**: alignment of hypothesis vs reference surfaces systematic substitutions; with enough consistent evidence they become candidate rules with auto-generated golden-test skeletons, and ride a lifecycle `candidate → approved (log-only) → active` driven by accumulated evidence.
4. **Retraining** triggers on data volume, NDI drift, or schedule — reusing the phase-4 pipeline and gates unchanged.
5. **Shadow deployment**: the candidate runs in parallel on live traffic (never adding latency to responses); after ≥500 utterances it is promoted **blue-green** only if it passes all gates. Rollback is one command.
6. **Noise stats loop**: per-store noise profiles (from the axis-1 classifier's production predictions) feed back into sensitivity re-runs and mitigation priorities.

The entire cycle is proven on **simulated traffic with injected known errors** (a planted novel confusion pair, a planted noise shift) before touching real audio — the acceptance test for the flywheel is literally "did it find what we hid?".

## 6. Production inference path

```
audio (16 kHz mono) → Silero VAD gate → noise classifier → targeted denoise chain
  → Faster-Whisper CT2 (menu terms in initial_prompt, language clamped to es/en)
  → hallucination guard (no-speech/logprob thresholds + repetition truncation)
  → keydetector (lexicon + rules) → final transcript + full decision trace
```

If VAD finds no speech, the request returns empty **without invoking the model** — the primary hallucination defense. Every stage logs its decision so the flywheel can learn from production traces. Latency budget: ≤3 s end-to-end for a 5 s utterance on CPU int8 (RTF ≤ 0.6).

## 7. Bootstrap data (zero proprietary audio required)

The system bootstraps entirely from no-auth public data + offline TTS, and is designed to progressively swap in real recordings later (phase 7 defines the field-recording protocol):

| Need | Source |
|------|--------|
| Clean Spanish speech (multi-accent LatAm) | OpenSLR crowdsourced series: SLR61 (AR), SLR71 (CL), SLR72 (CO), SLR73 (PE), SLR74 (PR), SLR75 (VE) — balanced hours per accent |
| Clean English speech | LibriSpeech dev/test-clean |
| Domain phrases (orders, menu terms, confusion words) | Grammar-based order generator rendered with Piper TTS, both languages, multiple voices |
| Kitchen / café / meeting noise | DEMAND (DKITCHEN, PCAFETER, OMEETING…) |
| Babble & music | MUSAN |
| Traffic / construction | UrbanSound8K |

One known gap is tracked explicitly: **Mexican Spanish** — the dominant accent among US Spanish speakers — has no no-auth public read corpus. Bootstrap coverage comes from `es_MX` TTS voices in the domain corpus; Common Voice (auth-walled) is an optional phase-7 enrichment; real field recordings are the definitive fix. Dataset manifests carry an `accent` column so evaluations can always be sliced per accent.

## 8. Implementation plan

The build is 8 strictly-sequential phases, each with explicit deliverables, step-by-step tasks, acceptance tests, and an exit checklist. See [plan/00-overview.md](plan/00-overview.md) for the full map.

| Phase | Delivers | Hard gate (summary) |
|-------|----------|---------------------|
| [0 — Foundations](plan/phases/phase-0-foundations.md) | repo, config, storage, Docker, CI, datasets, TTS corpus | manifests valid, CI green |
| [1 — Baseline ASR](plan/phases/phase-1-baseline-asr.md) | VAD + engine + guards + API + eval harness | frozen baseline WER (es/en); pure noise → empty output |
| [2 — Noise Lab](plan/phases/phase-2-noise-lab.md) | noise bank, SNR mixer, matrix corpus, sensitivity + NDI | mixer accurate to ±0.5 dB; full-matrix NDI report |
| [3 — Axis 1](plan/phases/phase-3-axis1-preprocessing.md) | noise classifier, denoiser chains, generated policy | ≥5% rel. WER gain on top-damage cells; zero clean harm |
| [4 — Axis 2](plan/phases/phase-4-axis2-lora.md) | damage-weighted LoRA pipeline, regression suite, CT2 export | ≥15% rel. noisy-WER gain per language; no forgetting |
| [5 — Axis 3](plan/phases/phase-5-axis3-keydetector.md) | lexicon, rule engine, golden framework, seed rules | KER gain; false corrections ≤0.5% |
| [6 — Flywheel](plan/phases/phase-6-flywheel.md) | harvester, judge, review CLI, miner, shadow, promotion | full simulated cycle finds the planted errors |
| [7 — Hardening](plan/phases/phase-7-hardening-ops.md) | dashboards, drift alarms, auth/privacy, field protocol | ops checklist; real-data onboarding ready |

Cross-cutting specs: [conventions](plan/01-conventions.md) (normative naming/formats), [architecture & contracts](plan/02-architecture.md), [data schemas](plan/03-data-spec.md), [test strategy](plan/testing/test-strategy.md). Progress lives in [plan/STATUS.md](plan/STATUS.md); resolved ambiguities in [plan/DECISIONS.md](plan/DECISIONS.md).

## 9. Hardware

Two documented paths, chosen per deployment:

- **Training**: cloud GPU (RunPod/Lambda; ≥24 GB VRAM for whisper-medium LoRA, ≥12 GB for small) *or* a local NVIDIA GPU. All training scripts are device-agnostic (`accelerate`); a `make train-remote` target handles cloud sync.
- **Inference**: CPU with int8 CTranslate2 quantization is the default (fits restaurant edge hardware); GPU float16 where available. The latency budget is set for the CPU path.

## 10. Glossary

| Term | Meaning |
|------|---------|
| **WER / CER** | Word / character error rate (via `jiwer`, on normalized text) |
| **KER** | Keyword error rate: share of domain keywords (menu items, critical terms) not recovered in the hypothesis |
| **NDI** | Noise Damage Index: per-subtype damage score `0.5·ΔWER_rel + 0.4·ΔKER_rel + 0.1·hallucination_rate`, averaged over levels |
| **SNR** | Signal-to-noise ratio, computed against speech-active RMS (VAD frames), in dB |
| **Level 05/10/15** | Noise volume codes → SNR +10 / 0 / −5 dB (10 = noise as loud as speech) |
| **Keydetector** | The axis-3 post-ASR correction engine (lexicon + confusion rules) |
| **Golden test** | Committed input→output case pair (positive + negative) required for every confusion rule |
| **Shadow deployment** | Candidate model running in parallel on live traffic, output logged but not returned |
| **Flywheel** | The weekly autonomous improvement cycle across all three axes |
