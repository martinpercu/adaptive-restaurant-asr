# Phase 0 — Foundations

**Goal:** a reproducible development environment: repo scaffold, config system, storage abstraction, public datasets downloaded and manifested, CI green.
**Depends on:** nothing. **Estimated effort:** 2–4 days.

## Deliverables

- Git repo initialized, `.gitignore` covering `data/`, `models/` (except `models/registry.json`), `reports/`, caches, and **`.env`** (API keys live there; an `.env.example` with variable names only — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — is committed instead).
- `pyproject.toml` with dependency groups; `Makefile`; `docker/` (CPU Dockerfile, compose with MinIO); CI workflow.
- `src/ars/config.py`, `src/ars/storage.py`, `src/ars/contracts.py` (initial models from [02](../02-architecture.md)/[03](../03-data-spec.md)).
- `configs/default.yaml`, `configs/noise_taxonomy.yaml` (seed content from [01 §3](../01-conventions.md)).
- `scripts/download_datasets.py` + per-dataset manifests under `data/clean/` and raw noise sources staged for phase 2.
- `scripts/tts_corpus.py` — restaurant-order TTS generator (both languages).
- `plan/STATUS.md` updated.

## Tasks

### 0.1 Repo & tooling
`git init`; Python ≥ 3.11 via `uv`; `pyproject.toml` with groups:
- core: `numpy pandas pyarrow pydantic pydantic-settings soundfile torchaudio structlog fastapi uvicorn faster-whisper jiwer rapidfuzz jellyfish pyyaml`
- `train`: `torch transformers peft datasets accelerate ctranslate2`
- `preprocess`: `noisereduce deepfilternet` (+ optional extra `separation`: `demucs`)
- `flywheel`: `prefect anthropic openai`
- dev: `pytest pytest-cov ruff`

Pin majors. Makefile targets: `setup lint test test-acceptance gate download-data fixtures api`. `make gate PHASE=N` runs `pytest tests/acceptance/test_phase{N}_*.py -m acceptance`.

### 0.2 Config system
`ars.config.Settings` (pydantic-settings): loads `configs/default.yaml`, env override prefix `ARS_`. Sections: `paths`, `vad`, `asr`, `preprocess`, `keydetector`, `training`, `judge`, `flywheel`, `eval`. Fail-fast on unknown keys (`extra="forbid"`).

### 0.3 Storage
`Storage` interface: `put(path, bytes)`, `get(path)`, `exists`, `list(prefix)`, `url(path)`. Implement `LocalStorage` (rooted at `settings.paths.data`). Add `S3Storage` (boto3 → MinIO from compose) behind the same interface; smoke-tested only when `ARS_S3_ENDPOINT` is set.

### 0.4 Docker & CI
`docker/Dockerfile.cpu` (python-slim + ffmpeg + libsndfile), `docker-compose.yml` (api + minio). Document (don't build) `Dockerfile.gpu` (CUDA base) for the cloud-GPU path. CI (GitHub Actions): `ruff check`, `pytest -m "not slow and not gpu and not network and not acceptance"` on CPU.

### 0.5 Dataset downloads (no-auth sources only)
`scripts/download_datasets.py` with one subcommand per source, each: download → checksum verify → convert to 16 kHz mono WAV → write manifest per [03 §1](../03-data-spec.md).

| Source | Use | Why this one |
|--------|-----|--------------|
| OpenSLR crowdsourced LatAm Spanish series — SLR61 (AR), SLR71 (CL), SLR72 (CO), SLR73 (PE), SLR74 (PR), SLR75 (VE), all CC-BY-SA | clean `es` speech, multi-accent | direct download, no auth; covers Southern Cone, Andean and Caribbean accents for the USA + LATAM market |
| LibriSpeech `dev-clean` + `test-clean` | clean `en` speech | direct download, standard |
| DEMAND (DKITCHEN, PCAFETER, OMEETING, TMETRO...) | noise seeds (fam. A/C) | real recorded environments |
| MUSAN (noise + babble subsets) | babble (AC/CA), misc | permissive license |
| UrbanSound8K (jackhammer, drilling, engine idling, street) | fam. B seeds | labeled outdoor classes |

Store raw noise under `data/_staging/noise_sources/<source>/` — phase 2 curates it into the bank. Record every file's license. Subset sizes: ≥ 2 h clean speech per language — for `es`, balanced across the six accent sources (roughly equal hours per accent, `accent` column filled per [03 §1](../03-data-spec.md)); ≥ 30 min raw noise per intended subtype.

### 0.6 TTS domain corpus
`scripts/tts_corpus.py`: grammar-based order generator (templates × synthetic bilingual menu from `configs/menu/demo.yaml`, which this task creates: ~40 items with es/en names + aliases). Render with Piper (offline TTS, several es/en voices) → `cl-<lang>-*` utterances with references and `keywords` filled. Include utterances containing every word from the idea.txt confusion tables (they become KER targets). Target: ≥ 500 utterances per language, ≥ 3 voices each; prioritize `es_MX` and other LatAm voices for Spanish and `en_US` voices for English, filling `accent` from each voice's locale.

### 0.7 Fixture generator
`tests/conftest.py` fixtures: seeded sine/chirp/white/pink noise/silence WAV factories per [testing strategy](../testing/test-strategy.md).

## Acceptance tests (`tests/acceptance/test_phase0_*.py`)

1. `test_config_loads_and_rejects_unknown_keys` — Settings loads default.yaml; a bogus key raises.
2. `test_storage_roundtrip` — put/get/exists/list on LocalStorage with tmp root.
3. `test_taxonomy_registry_valid` — every subtype references an existing family; canonical levels map to documented SNRs.
4. `test_dataset_manifests_valid` — every manifest under `data/` parses, schema-validates, all `path`s exist, langs ∈ {es,en}. Skips (with clear message) if datasets not downloaded.
5. `test_tts_corpus_manifest` — ≥ 500 rows/lang, every idea.txt confusion word appears in ≥ 1 utterance's `keywords`, all audio 16 kHz mono.
6. `test_make_targets` — `make lint` and `make test` exit 0 (subprocess).

## Exit checklist
- [ ] CI green on a clean clone (datasets skipped in CI).
- [ ] `make download-data && make gate PHASE=0` passes locally with all datasets present.
- [ ] `STATUS.md` row for phase 0 → done, with date and gate output summary.

## Pitfalls
- Piper voice downloads need `network`-marked tests; TTS generation itself must be a script, not a test.
- LibriSpeech is FLAC — convert, don't symlink. DEMAND channels: use channel 1 only.
- Do not attempt Common Voice here (auth wall); it is an optional phase-7 enrichment.
- Known accent gap: there is no no-auth Mexican-Spanish read corpus, and Mexican/Caribbean accents dominate US Spanish. Interim coverage = `es_MX` TTS voices (task 0.6); Common Voice `es` is the phase-7 enrichment; real field audio is the definitive fix. A US deployment must include an `es-mx` eval slice before its gates count (phase 7).
