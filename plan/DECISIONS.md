# DECISIONS — Build Decision Log

> Append-only. When the plan is ambiguous, silent, or a gate cannot be met as written, record the decision here **before** acting on it. Newest first.

Format:

```
## YYYY-MM-DD — short title
- **Context:** what the plan says / doesn't say, what happened
- **Decision:** what was chosen
- **Why:** rationale, alternatives considered
- **Impact:** files/configs/gates affected
```

---

## 2026-07-17 — Phase 0: Python pinned to 3.12; `make setup` installs core+dev, not all groups
- **Context:** the plan says Python ≥ 3.11 and `make setup` = "uv sync, all groups". The build host has Python 3.14, for which `torch`/`ctranslate2`/`faster-whisper` wheels are not yet reliably published; and `--all-groups` pulls `torch`, `deepfilternet`, `demucs`, `prefect` — none needed until phases 3/4/6 — making onboarding and CI heavy.
- **Decision:** pin the project to Python 3.12 via `.python-version` and `requires-python = ">=3.11,<3.13"`. `make setup` runs `uv sync` (main + `dev` group only); `make setup-all` runs `uv sync --all-groups` for the training/preprocess/flywheel path. CI installs main+dev.
- **Why:** 3.12 has stable wheels for the whole core stack; deferring the heavy ML groups keeps CI fast and matches phase ordering (their deps land when their phase does). No gate depends on the deferred groups in phase 0.
- **Impact:** `pyproject.toml`, `Makefile`, `.python-version`, CI workflow. Revisit the upper Python bound when 3.13 wheels for torch/ctranslate2 are broadly available.

## 2026-07-17 — Phase 0 noise bootstrap: MUSAN + UrbanSound8K deferred to phase 2 (disk limit)
- **Context:** task 0.5 stages DEMAND + MUSAN + UrbanSound8K raw noise. The build host has only ~22 GiB free; MUSAN (~11 GB archive, similar extracted) and UrbanSound8K (~6 GB) together would overflow the disk mid-download. The phase-0 **gate does not test noise** (noise-bank manifests are a phase-2 deliverable; phase-0 acceptance test 4 only validates speech/TTS manifests).
- **Decision:** phase-0 `download-data` ingests the full speech corpora (OpenSLR es ×6 balanced ~0.5 h/accent, LibriSpeech ~2.5 h) plus **DEMAND** (small 16 kHz zips — real kitchen/cafeteria noise, covers subtypes AA/AB/CA). MUSAN (babble AC/CA, music CB) and UrbanSound8K (outdoor BA/BB/BC) are deferred to phase 2, to be fetched when curating the noise bank — as a subset or after freeing disk. Zenodo/OpenSLR URLs verified reachable.
- **Why:** unblocks phase 0 (gate is speech/TTS-only) without risking a disk-full failure; the large outdoor/babble corpora land exactly when phase 2 needs them. Drive-thru outdoor coverage (family B) is the phase-2 priority and will require UrbanSound8K + composite BC clips then.
- **Impact:** phase-0 download run scope; phase-2 curation must (re)fetch MUSAN + UrbanSound8K. No phase-0 gate affected. Downloaders for all sources remain in `scripts/download_datasets.py`.

## 2026-07-17 — `torch` promoted to a core dependency
- **Context:** the plan lists `torchaudio` in the core group as the sanctioned transforms/resampler (01 §1). `torchaudio` 2.11 ships with **no** declared `torch` dependency, so `uv sync` installed torchaudio alone; resampling 48 kHz OpenSLR/LibriSpeech audio to the canonical 16 kHz then failed with `ModuleNotFoundError: No module named 'torch'`.
- **Decision:** add `torch>=2.2,<3` explicitly to core `dependencies` (removed the now-redundant copy from the `train` group). No resampler substitution — soundfile for I/O, torchaudio for transforms, exactly as 01 §1 mandates.
- **Why:** torch is torchaudio's runtime engine; making the transitive dependency explicit is required for the ingest conversion path to work. Heavier `make setup`, but unavoidable for the mandated audio stack.
- **Impact:** `pyproject.toml` core + train groups; `scripts/audio_io.py` resample path.

## 2026-07-17 — Judge made provider-pluggable (adds OpenAI gpt-4o-mini / gpt-4.1-nano)
- **Context:** the product owner wants OpenAI's `gpt-4.1-nano` and `gpt-4o-mini` available as judge models alongside `claude-sonnet-5`; `OPENAI_API_KEY` supplied via `.env`.
- **Decision:** `JudgeClient` becomes a config-selected interface with per-provider implementations; structured outputs enforced on both providers; batch APIs for the weekly cycle; a **calibration gate** (≥ 90% verdict agreement vs `claude-sonnet-5` on ≥ 100 labeled items) before any cheaper model becomes the production judge; `.env` added to the phase-0 gitignore spec plus a committed `.env.example`; `openai` added to the `flywheel` dependency group.
- **Why:** cost flexibility with a measured quality guardrail instead of blind substitution — the OpenAI minis are far cheaper, but judge errors poison training labels and mined rules, so agreement must be proven first.
- **Impact:** 03-data-spec §8, phase-6 (6.2 + acceptance test 2), phase-0 (gitignore, deps), CLAUDE.md, README.

## 2026-07-17 — Drive-thru confirmed as primary channel
- **Context:** the product owner confirmed the company's primary channel is the drive-thru lane.
- **Decision:** added subtype `BC car-cabin` (passenger/kid voices + engine bed inside the customer's car — the "¡y papas fritas!" scenario), built as composite clips from MUSAN speech + UrbanSound8K engine noise; family B + BC flagged as the expected priority in sensitivity analysis; field-recording protocol prioritizes the drive-thru mic and logs wind/weather, with a `BD wind-weather` subtype reserved for when real recordings exist.
- **Why:** outdoor/vehicle noise dominates the primary channel; dining-room families stay in the grid but are secondary.
- **Impact:** 01-conventions §3.1, phase-2 (curation table + analysis), phase-7 §7.5, README, 00-overview.

## 2026-07-17 — Judge call spec corrected for Claude Sonnet 5 API surface
- **Context:** the plan originally specified "temperature 0, JSON-only system prompt" for the LLM-judge. Claude Sonnet 5 rejects non-default sampling parameters (400) and natively supports structured outputs.
- **Decision:** judge uses `claude-sonnet-5` with `output_config.format` (json_schema of the verdict contract) and no sampling parameters; the weekly cycle submits judge calls via the Message Batches API (50% price).
- **Why:** API accuracy + ~50% judge cost reduction with no downside at weekly cadence.
- **Impact:** 03-data-spec §8, phase-6 task 6.2.

## 2026-07-17 — Spanish accent coverage widened for USA + LATAM market
- **Context:** the initial plan bootstrapped clean `es` speech from OpenSLR SLR61 only (Argentinian / Rioplatense accent). The product owner clarified the target market is **USA + LATAM** (a drive-thru voice automation company) — single-dialect coverage would bias the model and its evaluations.
- **Decision:** use the full OpenSLR crowdsourced LatAm series (SLR61 AR, SLR71 CL, SLR72 CO, SLR73 PE, SLR74 PR, SLR75 VE) with balanced hours per accent; add an optional `accent` column to dataset manifests ([03 §1](03-data-spec.md)) so metrics can be sliced per accent; TTS domain corpus prioritizes `es_MX` (Spanish) and `en_US` (English) Piper voices. Known gap: no no-auth Mexican-Spanish read corpus exists — mitigated now via `es_MX` TTS, later via Common Voice enrichment (phase 7) and real field audio; a US deployment requires an `es-mx` eval slice before its gates count.
- **Why:** Mexican/Caribbean accents dominate US Spanish; multi-accent training/eval avoids dialect bias; all chosen sources remain no-auth direct downloads so the automated bootstrap still works.
- **Impact:** phase-0 tasks 0.5/0.6 + pitfalls, 03-data-spec §1, README (intro + §7), 00-overview, CLAUDE.md.

## 2026-07-17 — Plan created
- **Context:** initial planning session from idea.txt + product owner answers.
- **Decision:** bilingual es+en scope; docs in English; hardware dual-path (cloud GPU training, CPU int8 inference default); bootstrap from public datasets (OpenSLR61, LibriSpeech, DEMAND, MUSAN, UrbanSound8K) + Piper TTS domain corpus; noise level codes 05/10/15 mapped to SNR +10/0/−5 dB (level 10 = noise RMS equals speech-active RMS, per product definition; −5 dB chosen over −10 dB for level 15 to avoid WER ceiling effects in diagnostics, with optional level 20 = −10 dB for stress).
- **Why:** answers from the product owner (2026-07-17); no-auth datasets chosen so an automated builder can bootstrap without credentials.
- **Impact:** entire plan/ tree.
