# CLAUDE.md — Builder Protocol for ARS

You are building **ARS (Adaptive Restaurant Speech)**: a self-improving bilingual (es+en) restaurant ASR system for a **drive-thru voice automation company** (USA + LATAM markets). The complete plan lives in [plan/](plan/). Your job is to execute it phase by phase. This file tells you *how to work*; the plan tells you *what to build*.

## Read order (mandatory, before any code)

1. [README.md](README.md) — what the system is and why.
2. [plan/00-overview.md](plan/00-overview.md) — phases and document map.
3. [plan/01-conventions.md](plan/01-conventions.md) — **normative**: naming, formats, taxonomy, hard rules.
4. [plan/STATUS.md](plan/STATUS.md) — find the current phase.
5. The current phase doc in [plan/phases/](plan/phases/) — plus [plan/02-architecture.md](plan/02-architecture.md) and [plan/03-data-spec.md](plan/03-data-spec.md) for any contract you touch.

## Phase discipline

- Phases run **strictly in order**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Do not start phase N+1 until phase N's **exit checklist** is fully green and STATUS.md says `done` with evidence.
- A phase is done when `make gate PHASE=N` passes **and** its `slow`-marked gate tests have been run at least once with evidence (metric tables, report paths) pasted into STATUS.md.
- Once a phase closes, its acceptance tests join the permanent suite. Keep them green forever.
- Work in small verified steps: implement a task, run its tests, then move on. Never batch a whole phase untested.

## Hard rules (violating any of these is a build failure)

1. **Never weaken, skip, or delete a gate/test to make it pass.** If a gate can't be met, stop and record the evidence and options in [plan/DECISIONS.md](plan/DECISIONS.md).
2. Never commit `data/`, `models/` (except `models/registry.json`), or `reports/` contents beyond manifests/READMEs.
3. Never hand-edit generated files (`configs/mitigation_policy.yaml`, registries, reports).
4. Every stochastic process takes an explicit seed (default 1337) and records it in its output manifest.
5. Every confusion rule ships with its golden positive **and** negative test in the same commit.
6. All cross-module data passes through the pydantic contracts in `src/ars/contracts.py` — no ad-hoc dicts between modules.
7. No test may call external APIs; the LLM judge is always `MockJudge` in tests.
8. Timestamps UTC ISO-8601 `Z`. Audio is 16 kHz mono PCM float32 everywhere past ingest.

## When the plan is ambiguous or wrong

Choose the simplest option consistent with [plan/01-conventions.md](plan/01-conventions.md), **record it in [plan/DECISIONS.md](plan/DECISIONS.md) before proceeding** (context / decision / why / impact), and continue. If the ambiguity blocks a gate or changes a contract in [plan/02](plan/02-architecture.md)/[03](plan/03-data-spec.md), stop and ask the product owner instead.

## Bookkeeping you own

- [plan/STATUS.md](plan/STATUS.md): keep the phase table, "Current focus" note, and key-numbers table current. Update at every phase transition and after every gate run.
- [plan/DECISIONS.md](plan/DECISIONS.md): append-only decision log.
- Commits: small, per-task, imperative messages (`phase2: SNR mixer with speech-active RMS`). Never commit secrets; API keys come from env (`ARS_*`).

## Commands (available after phase 0 sets them up)

```bash
make setup            # uv sync, all groups
make lint             # ruff check + format check
make test             # CI-equivalent: pytest -m "not slow and not gpu and not network and not acceptance"
make gate PHASE=N     # acceptance tests for phase N
make download-data    # public dataset bootstrap (idempotent)
make fixtures         # generate TTS test fixtures into data/fixtures/
make api              # run the FastAPI service locally
make cycle            # trigger a flywheel cycle manually (phase 6+)
```

## Tech stack (fixed — do not substitute without a DECISIONS entry)

Python ≥3.11 · uv · pydantic v2 · FastAPI · Silero VAD · faster-whisper (CTranslate2, int8 CPU / fp16 GPU) · HF Transformers + PEFT (LoRA) · jiwer · RapidFuzz + jellyfish · noisereduce + DeepFilterNet (+ Demucs optional) · SQLite (WAL) + Parquet manifests · Prefect · pluggable LLM-as-a-Judge (Anthropic `claude-sonnet-5` reference default; OpenAI `gpt-4o-mini` / `gpt-4.1-nano` alternates behind a calibration gate) · structlog · pytest + ruff · Docker (+ MinIO for S3-compatible storage).

## Domain constants to keep in your head

- Languages: exactly `es` and `en`; every metric gate must pass **per language**.
- Noise codes: `noise-<SUBTYPE>-<LEVEL>`; subtypes registered in `configs/noise_taxonomy.yaml`; levels 05/10/15 → SNR +10/0/−5 dB against **speech-active** RMS (level 10 ⇔ noise as loud as speech).
- NDI (Noise Damage Index) rankings steer everything: axis-1 mitigation targets, axis-2 sampling weights, mining priorities.
- The keydetector corrects with rules, not retraining; over-correction is worse than under-correction.
- Whisper hallucination defenses are sacred: VAD gate before the model, `condition_on_previous_text=False`, repetition guard after.
