# CLAUDE.md — Working Protocol for ARS

**ARS (Adaptive Restaurant Speech)** is a self-improving, bilingual (**es + en**) **ASR** system
for restaurant **drive-thrus** (USA + LATAM). *ARS* is the project codename; *ASR* (Automatic
Speech Recognition) is the field — the near-anagram is intentional, not a typo.

**The system is built.** All 8 phases (0→7) are closed and CI-green. Your job now is to
**extend and maintain it without regressing anything**. This file tells you *how to work on it*;
[`plan/`](plan/) tells you *what it is and why it was built that way*; [`BUILD-REPORT.md`](BUILD-REPORT.md)
is the honest one-page record of what actually happened during the build.

> This is a **public portfolio showcase** (github.com/martinpercu/adaptive-restaurant-asr).
> Keep it honest and clean: real numbers only, no client name anywhere, and no AI-assistant
> attribution in git history (see §7).

---

## 1. Read order (before touching code)

1. [README.md](README.md) — what the system is, the results, the diagrams.
2. [BUILD-REPORT.md](BUILD-REPORT.md) — the real results, the bugs found, every judgment call.
3. [plan/00-overview.md](plan/00-overview.md) — the 8-phase map and document index.
4. [plan/01-conventions.md](plan/01-conventions.md) — **normative**: naming, formats, taxonomy, hard rules.
5. [plan/STATUS.md](plan/STATUS.md) — the per-phase state, key numbers, and the documented deferrals.
6. For any contract you touch: [plan/02-architecture.md](plan/02-architecture.md) and
   [plan/03-data-spec.md](plan/03-data-spec.md).

The `plan/` folder is the authoritative spec — **do not duplicate it here**. When this file and
`plan/` disagree, `plan/` wins for *what to build*; this file wins for *how to work*.

---

## 2. Where things live (module map)

Everything is under `src/ars/`. Every subsystem has a CLI (`python -m ars.<module> ...`) — no
notebook-only logic. Cross-module data always passes through the pydantic contracts.

| Area | Modules | What it does |
|------|---------|--------------|
| **Core** | `config.py` `contracts.py` `storage.py` `db.py` `registry.py` `telemetry.py` `pipeline.py` | settings, pydantic contracts, SQLite(WAL)+Parquet, model registry (blue-green), structlog, the inference pipeline |
| **Inference** | `vad/silero.py` · `asr/{engine,guard,prompt_builder}.py` · `api/app.py` | VAD gate → Faster-Whisper → hallucination guard → keydetector → FastAPI |
| **Noise Lab** | `noise_lab/{taxonomy,curate,mixer,build_corpus,sensitivity,ndi}.py` | taxonomy, noise-bank curation, SNR mixer, eval-matrix, sensitivity run → **NDI** |
| **Axis 1 — preprocess** | `preprocess/{classifier,train_classifier,denoisers,evaluate,gen_policy,policy}.py` | noise classifier, denoiser registry, measured→derived mitigation policy |
| **Axis 2 — LoRA** | `training/{dataset_builder,train_lora,regression,export_ct2,gate}.py` | NDI-weighted data, PEFT/LoRA, anti-forgetting, CT2 export+parity, promotion gate |
| **Axis 3 — keydetector** | `keydetector/{lexicon,phonetics,rules,pipeline,eval,passthrough}.py` | menu lexicon + confusion-rule engine (deterministic post-ASR correction) |
| **Flywheel** | `flywheel/{harvester,pair_miner,lifecycle,shadow,promote,review,flows,simulate}.py` · `judge/client.py` | harvest→judge→mine→shadow→promote loop + simulation harness |
| **Eval / Ops** | `eval/{run,baseline,metrics,normalize,dashboard}.py` · `ops/{drift,retention}.py` | WER/CER/KER/hallucination, dashboard, drift monitors, retention |

Config lives in `configs/` (taxonomy, menus, rules, seeds, **generated** policy). Reports and
evidence in [`reports/`](reports/). Ops docs in [`docs/`](docs/).

---

## 3. Current state (keep it true — mirror `plan/STATUS.md`)

- **All 8 phases done.** `make gate PHASE=0..7` all pass. Suite: **120 CI + 92 acceptance tests**,
  `ruff` clean. There is exactly **1 skip + 2 xfail**, and each one is a *documented* CPU/GPU
  boundary in [`plan/DECISIONS.md`](plan/DECISIONS.md) — never an untracked failure.
- **Headline numbers:** clean WER **es 6.2% / en 2.9%** (whisper-small int8, model `0.1.0`);
  NDI top-3 (both langs) **BB / CA / BC**; keydetector **0% false-correction**; LoRA smoke
  loss 5.60→1.18, CT2 parity 0.134; flywheel simulates a full weekly cycle.
- **Registry:** `0.1.0` = **production**; `0.2.0` = **candidate** (LoRA, not promoted — needs GPU).
- **This is a local, CPU-only build.** The honest deferrals (each wired to its resolution) are:
  - **Axis-2 ≥15% noisy-WER** → needs GPU (`docker/Dockerfile.gpu`); `0.2.0` stays candidate.
  - **Axis-1 mitigation & classifier F1** → neural denoisers over CPU budget / proxy-noise ceiling
    → phase-7 real field audio.
  - **Keydetector ≥10% KER** → phase-6 mining of real production confusion pairs.
  - **Data:** MUSAN, UrbanSound8K-full, CB-music, Common Voice, real field audio → onboarding
    via [`docs/FIELD-RECORDING-PROTOCOL.md`](docs/FIELD-RECORDING-PROTOCOL.md).

The single most valuable real-world next step is that field-recording onboarding: it replaces
public proxies with real drive-thru audio, after which the promotion gates run on real eval sets.

---

## 4. Hard rules (violating any of these is a regression)

1. **Never weaken, skip, or delete a gate/test to make it pass.** If a gate can't be met, stop and
   record the evidence + options in [`plan/DECISIONS.md`](plan/DECISIONS.md). Closed phases' tests
   are permanent — **keep them green forever.**
2. Never commit `data/`, `models/` weights, or heavy `reports/` artifacts. The `.gitignore` allows
   **only light text/image evidence** (`*.md/*.json/*.png/*.html`, classifier `eval.json`) for the
   showcase — never `*.parquet`, `*.wav`, `*.onnx`, `ct2/`, adapters, or the DB.
3. Never hand-edit generated files (`configs/mitigation_policy.yaml`, registries, reports).
   Regenerate them via their CLI.
4. Every stochastic process takes an explicit **seed (default 1337)** and records it in its manifest.
5. Every confusion rule ships with its golden **positive and negative** test in the same commit.
6. All cross-module data goes through the pydantic contracts in `src/ars/contracts.py` — no ad-hoc
   dicts between modules.
7. **No test may call external APIs.** The LLM judge is always `MockJudge` in tests. API keys come
   only from env (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `ARS_*`) — never committed or logged.
8. Timestamps UTC ISO-8601 `Z`. Audio is 16 kHz mono PCM float32 everywhere past ingest.
9. The Whisper hallucination defenses are sacred: VAD gate before the model,
   `condition_on_previous_text=False`, repetition guard after.

## When the plan is ambiguous or wrong

Choose the simplest option consistent with [plan/01-conventions.md](plan/01-conventions.md),
**record it in [plan/DECISIONS.md](plan/DECISIONS.md) before proceeding** (context / decision / why
/ impact), and continue. If it blocks a gate or changes a contract in
[plan/02](plan/02-architecture.md)/[03](plan/03-data-spec.md), stop and ask the product owner.

---

## 5. How to work on it

- **Small verified steps.** Implement a change, run its tests, then move on. Never batch untested.
- **Touching a subsystem?** Run its gate (`make gate PHASE=N`) plus `make test` before you commit.
  A change that reddens any closed phase's gate is not done.
- **New confusion rule?** Golden positive **and** negative in the same commit (rule 5). Prefer
  context-gated rules over lexicon changes; over-correction is worse than under-correction.
- **New model?** Go through the registry (`python -m ars.registry ...`), never hand-edit
  `models/registry.json`. Promotion is blue-green with an atomic rollback.
- **Regenerating policy/reports?** Use the CLI and let the seed + manifest record it.
- **Commits:** small, per-task, imperative (`phase2: SNR mixer with speech-active RMS`).

---

## 6. Commands

```bash
make setup            # uv sync (core + dev)
make lint             # ruff check + format check
make test             # CI-equivalent: pytest -m "not slow and not gpu and not network and not acceptance"
make gate PHASE=N     # acceptance tests for phase N (0..7)
make test-acceptance  # all acceptance gates
make download-data    # public dataset bootstrap (idempotent)
make tts-corpus       # Piper domain TTS corpus
make fixtures         # TTS test fixtures into data/fixtures/
make api              # run the FastAPI service locally
make cycle            # trigger a flywheel cycle manually

# reproduce the real evidence (local heavy path):
python -m ars.noise_lab.sensitivity --model-version 0.1.0   # NDI + heatmaps
python -m ars.flywheel.simulate --seed 1337                 # full simulated cycle
```

---

## 7. Git hygiene (showcase repo — non-negotiable)

- Remote `origin` → **github.com/martinpercu/adaptive-restaurant-asr** (public). Author on every
  commit is `martinpercu` only.
- A local **`.git/hooks/commit-msg` hook strips any AI-assistant attribution trailers**
  (`Co-Authored-By: Claude…`, `Claude-Session:`) from every commit message automatically. It is not
  tracked (lives in `.git/hooks/`), so **if this repo is re-cloned, reinstall it** before committing:
  ```sh
  printf '#!/bin/sh\nsed -i.bak "/^Co-Authored-By: Claude/d; /^Claude-Session:/d" "$1"\nrm -f "$1.bak"\n' \
    > .git/hooks/commit-msg && chmod +x .git/hooks/commit-msg
  ```
- Before any push, sanity-check: `git log --format='%B' | grep -c 'Claude'` must return `0`, and
  `git log --format='%an <%ae>' | sort -u` must show only `martinpercu`.
- **Never put the client's name** in any doc, code comment, commit, or report.

---

## 8. Bookkeeping you own

- [plan/STATUS.md](plan/STATUS.md): keep the phase table, "Current focus", key-numbers, and
  deferrals current. Update after every gate run and phase-affecting change.
- [plan/DECISIONS.md](plan/DECISIONS.md): append-only decision log (context / decision / why / impact).
- [BUILD-REPORT.md](BUILD-REPORT.md): the honest results narrative — refresh headline numbers if they
  change materially (e.g. a real GPU LoRA run).

---

## 9. Tech stack (fixed — no substitution without a DECISIONS entry)

Python ≥3.11 · uv · pydantic v2 · FastAPI · Silero VAD · faster-whisper (CTranslate2, int8 CPU /
fp16 GPU) · HF Transformers + PEFT (LoRA) · jiwer · RapidFuzz + jellyfish · noisereduce +
DeepFilterNet (+ Demucs optional) · SQLite (WAL) + Parquet manifests · Prefect (optional decorator)
· pluggable LLM-as-a-Judge (Anthropic `claude-sonnet-5` reference default; OpenAI `gpt-4o-mini` /
`gpt-4.1-nano` alternates behind a calibration gate) · structlog · pytest + ruff (line-length 100) ·
Docker (+ MinIO for S3-compatible storage). `torch` is pinned to 2.11.x to match `torchaudio` (ABI).

---

## 10. Domain constants to keep in your head

- **Languages:** exactly `es` and `en`; every metric gate must pass **per language**.
- **Noise codes:** `noise-<SUBTYPE>-<LEVEL>`; subtypes in `configs/noise_taxonomy.yaml` (families
  A/B/C/D, subtypes AA/AB/AC/BA/BB/BC/CA/CB); levels 05/10/15 → SNR +10/0/−5 dB against
  **speech-active** RMS (level 10 ⇔ noise as loud as speech).
- **NDI (Noise Damage Index)** = `0.5·ΔWER_rel + 0.4·ΔKER_rel + 0.1·hallucination_rate`, per subtype.
  It steers everything: axis-1 mitigation targets, axis-2 sampling weights, mining priorities.
  Current top-3 (both langs): **BB / CA / BC**.
- The **keydetector corrects with rules, not retraining**; over-correction is worse than under.
- Whisper hallucination defenses are sacred (see rule 9).
