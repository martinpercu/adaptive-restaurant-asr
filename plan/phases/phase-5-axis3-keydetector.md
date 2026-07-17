# Phase 5 — Axis 3: Keydetector (Post-ASR Correction)

**Goal:** the "after the model" axis: a deterministic correction engine that exploits the restaurant prior. Two mechanisms, in order: (1) **menu lexicon matching** — fuzzy + phonetic recovery of menu terms; (2) **confusion pair rules** — curated replacements for known phonetic confusions. Errors mined by the flywheel become *rules*, not training data: a rule ships in minutes with golden tests, no GPU involved.
**Depends on:** phase 1 gate (phases 2–4 not required at runtime; phase 2 mixer is used to build the eval set). Can run in parallel with phase 4 if staffing allows, but gate order in STATUS.md stays sequential. **Estimated effort:** 5–7 days.

## Deliverables

- `src/ars/keydetector/phonetics.py` — `en` double metaphone + `es_key` per [03 §7](../03-data-spec.md).
- `src/ars/keydetector/lexicon.py` — menu lexicon index + matcher.
- `src/ars/keydetector/rules.py` — confusion rule engine + YAML loader with lifecycle.
- `src/ars/keydetector/pipeline.py` — orchestration, `mode: replace | log_only`.
- `configs/rules/rules-es.yaml`, `rules-en.yaml` — seeded from the idea.txt confusion tables.
- `tests/golden/` framework + one golden pair per seeded rule.
- Dataset `eval-confusion-<lang>-v1` + KER report with/without keydetector.

## Tasks

### 5.1 Phonetic keys
Implement [03 §7](../03-data-spec.md) exactly, with the documented examples as unit tests. Note the coverage split: some seseo homophone pairs collide by key (`vaso`/`bazo` → `baso`, intended) and are recoverable by the lexicon; others do **not** collide (`hielo→ielo` vs `cielo→sielo`) and can only be fixed by a confusion rule. Encode the expected behavior *per pair* in the unit tests — never assume all confusion pairs share keys.

### 5.2 Menu lexicon
Build from `configs/menu/<store>.yaml` (items: `name_es`, `name_en`, `aliases`, `modifiers`). Index: normalized form + phonetic key per token/n-gram (up to trigram for multiword items). Matcher over hypothesis n-grams: exact normalized match (skip) → phonetic-key equality **or** RapidFuzz `ratio ≥ 88` → replace with canonical menu form, `Correction(rule_id="lexicon")`, confidence = scaled similarity. Constraints: never replace across an n-gram already corrected; never fire on stopwords; length guard (`abs(len_diff) ≤ 3` chars) to stop absurd matches.

### 5.3 Rule engine
Load rules per [03 §4](../03-data-spec.md); validate on load (unknown fields, bad status, duplicate ids → hard fail). Firing semantics: normalized-token match on `wrong` → check `context_any`/`context_none` within ±3 tokens → replace preserving original casing pattern. `active` fire in both modes; `approved` only log. Precedence: lexicon first; spans consumed by a correction are locked.

### 5.4 Seed rules
Transcribe idea.txt's confusion tables into rules (`source: seed`). **Direction matters**: the tables list restaurant-word → confusion-word, but a rule's `wrong` field is what the ASR *outputs* — so seed rules map confusion → restaurant word (`tejedor→tenedor`, `bocina→cocina`, `soap→soup`). Status per pair by ambiguity:
- `active` only for low-ambiguity confusions where the `wrong` form is rare in orders: `tejedor→tenedor`, `bocina→cocina` (context_any food terms), `soap→soup` (context_any order phrasing).
- `approved` (log-only) for confusions whose `wrong` form is a common legitimate word (`paso`, `ropa`, `misa`, `class`, `life`, `water`...), always with tight `context_any` gates. Phase 6 promotes them on evidence.

Record the rationale in each rule's `notes`. This conservatism is deliberate: false corrections are worse than missed ones.

### 5.5 Golden test framework
`tests/golden/cases-<lang>.yaml`: each case `{rule_id, input, expected, mode}` — for every non-retired rule ≥ 1 **positive** (correction happens, exact output) and ≥ 1 **negative** (the `wrong` word used legitimately → untouched). A pytest collects the YAML and runs all cases; a rule without both cases fails collection ([01 §8](../01-conventions.md) rule 5). Phase 6's miner auto-generates case skeletons for new rules.

### 5.6 Confusion eval set + measurement
`eval-confusion-<lang>-v1`: TTS utterances embedding each confusion's restaurant word in natural order phrases, mixed with noise at levels 10/15 via the phase-2 mixer (the levels where confusions actually happen). Run pipeline with keydetector off vs on (replace mode):
- **KER improves** ≥ 10% relative per lang on this set.
- **False-correction rate** on `eval-clean-*` general sentences (non-domain): corrections fired / utterances ≤ 0.5%, and none of them changes a correct keyword.

## Acceptance tests (`tests/acceptance/test_phase5_*.py`)

1. `test_phonetic_keys_table` — every documented example + seseo collision expectations.
2. `test_lexicon_recovers_menu_terms` — parametrized corrupted menu terms ("mega chedar blas" → "MegaCheddar Blast"-equivalent from demo menu) recovered; length guard and stopword guard hold.
3. `test_rule_engine_semantics` — context gates, casing preservation, one-rule-per-span, lexicon precedence, approved-vs-active behavior, no re-trigger on own output.
4. `test_rules_files_valid` — schema, unique ids, every rule has golden pair (cross-check with golden YAML).
5. `test_golden_all` — full golden suite green (this *is* the regression net for axis 3).
6. `test_modes` — `log_only`: text unchanged, corrections in trace; `replace`: applied.
7. `test_ker_improvement` (`slow`) — 5.6 thresholds met, both langs, report saved to `reports/keydetector/`.
8. `test_false_correction_rate` — ≤ 0.5% on clean general sets.
9. `test_latency` — keydetector ≤ 20 ms per utterance (pure Python budget; index built at startup).

## Exit checklist
- [ ] `make gate PHASE=5` green; KER before/after table in STATUS.md.
- [ ] Seed rules committed with golden pairs; ambiguous pairs documented as `approved` with promotion criteria (phase 6 flips them on evidence).
- [ ] API serves with `keydetector.mode: replace` on demo menu.

## Pitfalls
- The #1 failure mode is over-correction. Every heuristic here (contexts, length guard, conservative seeding, false-correction gate) exists to prevent it. When in doubt, `approved`/log-only.
- Multiword menu items must be indexed as units; token-by-token matching mangles them.
- Casing preservation matters for the POS cross-check later — "Coca" vs "coca".
