# 03 — Data Specifications

Machine-readable schemas. Implement each as a pydantic model in `src/ars/contracts.py` (runtime) or validate-on-load helpers (YAML/Parquet). Any file that violates its schema must fail loudly at load time.

## 1. Dataset manifest (`data/datasets/<dataset_id>/manifest.parquet`)

One row per utterance. Columns:

| Column | Type | Notes |
|--------|------|-------|
| `utterance_id` | str | per conventions §4 |
| `path` | str | relative to dataset dir |
| `lang` | str | `es` / `en` |
| `text` | str | reference transcript (normalized, see §6) |
| `duration_s` | float | |
| `source` | str | `openslr61`, `openslr72`, `librispeech`, `tts-piper`, `prod`, ... |
| `accent` | str? | locale-style accent tag: `es-ar`, `es-cl`, `es-co`, `es-pe`, `es-pr`, `es-ve`, `es-mx`, `en-us`, ...; from source metadata or TTS voice locale; null if unknown |
| `clean_id` | str? | for mixed rows: source clean utterance |
| `noise_subtype` | str? | e.g. `AB`; null = clean |
| `noise_level` | str? | `05`/`10`/`15`/`20` |
| `noise_clip_id` | str? | e.g. `nz-AB-0007` |
| `snr_db_target` | float? | from level mapping |
| `snr_db_achieved` | float? | measured post-mix; `abs(achieved-target) ≤ 0.5` |
| `mix_seed` | int? | |
| `keywords` | list[str] | domain terms present in `text` (for KER) |

Plus `data/datasets/<dataset_id>/dataset.json`: `{dataset_id, created_at, generator, generator_version, config_hash, seed, row_count, langs}`.

## 2. Noise bank manifest (`data/noise_bank/manifest.parquet`)

| Column | Type | Notes |
|--------|------|-------|
| `clip_id` | str | `nz-<SUBTYPE>-<seq>` |
| `subtype` | str | must exist in taxonomy registry |
| `path` | str | |
| `duration_s` | float | ≥ 3.0 |
| `source` | str | `demand-DKITCHEN`, `musan`, `urbansound8k`, `own-recording` |
| `license` | str | e.g. `CC-BY-4.0`; **required**, block ingestion if unknown |
| `split` | str | `train` / `eval` — a source clip never appears in both (leakage guard) |

## 3. Sensitivity matrix output (`reports/sensitivity/<run_id>/`)

- `matrix.parquet`: one row per `(model_version, lang, noise_subtype, noise_level)` — plus one `clean` row per lang — with columns `wer`, `cer`, `ker`, `hallucination_rate`, `n_utts`, `avg_logprob_mean`.
- `ndi.json`:

```json
{
  "run_id": "run-20260717T120000Z-1.0.0",
  "model_version": "1.0.0",
  "baseline": {"es": {"wer": 0.142}, "en": {"wer": 0.098}},
  "weights": {"d_wer": 0.5, "d_ker": 0.4, "hallucination": 0.1},
  "ranking": [
    {"subtype": "AC", "lang": "es", "ndi": 0.87,
     "per_level": {"05": 0.31, "10": 0.79, "15": 1.52}},
    {"subtype": "AB", "lang": "es", "ndi": 0.55, "per_level": {}}
  ]
}
```

NDI per cell = `0.5·ΔWER_rel + 0.4·ΔKER_rel + 0.1·hallucination_rate`, where `Δx_rel = (x_noisy − x_clean) / max(x_clean, 0.01)`. Subtype NDI = mean over canonical levels. Weights live in config, defaults as shown.

- `heatmap-{lang}.png`: subtypes × levels, cell = WER.

## 4. Confusion rule (`configs/rules/rules-<lang>.yaml`)

```yaml
- id: es-0001
  lang: es
  wrong: "bocina"          # surface form the ASR outputs
  right: "cocina"          # replacement
  scope: word              # word | phrase
  context_any: ["de la", "en la", "a la"]   # optional: fire only if any appears within ±3 tokens
  context_none: []         # optional: never fire if any appears
  status: active           # candidate | approved | active | retired
  provenance:
    source: seed           # seed | mined
    evidence_count: 12     # times observed in mined data (0 for seed)
    added: "2026-07-17"
  notes: "seseo; 'cocina' vs 'bocina' under extractor noise"
```

Engine semantics: match on **normalized** tokens (§6); at most one rule fires per token span; `lexicon` corrections run before rule corrections and win conflicts; a fired rule never re-triggers on its own output.

## 5. Model registry entry (`models/registry.json`)

```json
{
  "entries": [
    {
      "version": "1.3.0",
      "stage": "production",
      "base_model": "whisper-medium",
      "adapter": "lora-2026w29",
      "ct2_path": "models/ct2/1.3.0",
      "languages": ["es", "en"],
      "gates": {
        "noisy_wer_rel_improvement": {"es": 0.18, "en": 0.16, "required": 0.15},
        "clean_wer_rel_regression": {"es": 0.004, "en": 0.011, "max": 0.02},
        "keyword_recall_drop": {"es": 0.002, "en": 0.0, "max": 0.01}
      },
      "sensitivity_run": "run-...",
      "promoted_at": "2026-08-01T09:00:00Z",
      "promoted_by": "flywheel"
    }
  ]
}
```

## 6. Text normalization (used for WER/KER and keydetector matching)

Single implementation `ars.eval.normalize(text, lang)`: lowercase → strip punctuation (keep intra-word apostrophes in `en`) → collapse whitespace → number words left as-is (do **not** digit-normalize in v1) → strip diacritics **only for matching keys**, never in displayed text. WER/CER via `jiwer` on normalized text.

**KER (Keyword Error Rate)**: for each reference keyword, it is *recovered* if the hypothesis contains a token (or n-gram for phrases) with either exact normalized match or equal phonetic key (§7). `KER = 1 − recovered / total_keywords`.

**Hallucination rate**: fraction of utterances where hypothesis is non-empty but reference is empty, or repetition guard fired.

## 7. Phonetic keys (`ars.keydetector.phonetics`)

- `en`: double metaphone (primary key) via `jellyfish`.
- `es`: rule-based folding `es_key(word)`, applied in order: lowercase & strip diacritics (keep `ñ`→`n` last) → `v`→`b` → drop `h` (but keep `ch`, folded `ch`→`x`) → `ll`→`y` → `z`→`s`, `c(e|i)`→`s(e|i)`, remaining `c`→`k`, `qu`→`k`, `k` stays → `g(e|i)`→`j(e|i)` → `x`→`ks` → `w`→`u` → collapse doubled letters. Examples (must be unit-tested): `bocina→bosina`, `cocina→kosina`, `hielo→yelo`? No — `hielo→ielo`; `cielo→sielo`; `vaso→baso`; `bazo→baso` (collision intended: seseo homophones share keys).

## 8. LLM-judge contract (`ars.judge`)

Request: `{transcript, lang, menu_items: [str], pos_ticket: [str] | null, asr_confidence, audio_meta}`.
Response (the LLM must return exactly this JSON; validate with pydantic, retry once on invalid):

```json
{
  "verdict": "correct | minor_errors | wrong | hallucination",
  "corrected_reference": "string or null",
  "confusion_candidates": [{"heard": "bocina", "intended": "cocina"}],
  "order_core_match": true,
  "confidence": 0.86
}
```

**The provider is pluggable**: `JudgeClient` is an interface with one implementation per provider, selected by config (`judge.provider: anthropic | openai`, `judge.model`). Supported models:

| Provider | Model | Role |
|----------|-------|------|
| Anthropic (default) | `claude-sonnet-5` | **reference judge** — highest quality; all calibration is measured against it |
| OpenAI | `gpt-4o-mini` | low-cost alternate |
| OpenAI | `gpt-4.1-nano` | lowest-cost alternate |

Rules for every implementation:

- Enforce the JSON shape with the provider's **structured outputs** (Anthropic: `output_config.format` json_schema — and do **not** set `temperature`/`top_p`/`top_k`, Sonnet 5 rejects non-default sampling params with a 400; OpenAI: `response_format` json_schema with `strict: true`, `temperature: 0`). Never prompt-only JSON. Always validate with pydantic, one retry on invalid.
- The weekly cycle submits judgments through the configured provider's **batch API** (both offer ~50% batch pricing; results return well within the cycle's time budget).
- API keys from env only: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (loaded from `.env`, which is gitignored — never committed, never logged).
- **Calibration gate**: a cheaper model may become the production judge only after `python -m ars.judge.calibrate` shows ≥ 90% `verdict` agreement with `claude-sonnet-5` on a labeled set of ≥ 100 items (report in `reports/judge/`). Until then, non-reference models run in comparison mode only.

The client interface ships a `MockJudge` for tests — no test may hit the network, for any provider.

## 9. Telemetry line (JSONL, `data/telemetry/YYYY-MM-DD.jsonl`)

```json
{"ts": "...Z", "trace_id": "...", "store_id": "...", "duration_s": 4.2,
 "speech_ratio": 0.74, "noise_pred": "AB", "noise_confidence": 0.81,
 "chain_applied": ["deepfilternet"], "language": "es",
 "avg_logprob": -0.31, "guard_flags": [], "rules_fired": ["es-0001"],
 "latency_ms": {"vad": 12, "preprocess": 210, "asr": 1900, "keydetector": 3, "total": 2140},
 "model_version": "1.3.0"}
```

## 10. SQLite tables (`data/db/ars.db`)

```sql
utterances(utterance_id PK, path, lang, store_id, captured_at, duration_s, speech_ratio, meta_json)
transcriptions(id PK, utterance_id FK, model_version, raw_text, final_text, avg_logprob, guard_flags_json, created_at)
corrections(id PK, transcription_id FK, rule_id, before, after, confidence)
judge_verdicts(id PK, transcription_id FK, verdict, corrected_reference, confusion_json, order_core_match, confidence, created_at)
review_queue(id PK, transcription_id FK, reason, status /*pending|accepted|rejected*/, reviewer_note, resolved_at)
metric_runs(run_id PK, model_version, dataset_id, metrics_json, created_at)
```
