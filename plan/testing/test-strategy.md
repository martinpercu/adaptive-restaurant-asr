# Test Strategy

How testing works across all phases. Phase docs define *what* is asserted; this doc defines *how* tests are organized, run, and gated.

## 1. Layers

| Layer | Location | Marker | Runs in CI | Purpose |
|-------|----------|--------|------------|---------|
| Unit | `tests/unit/` | none | yes | pure functions: mixer math, normalization, phonetic keys, gate logic, schemas |
| Integration | `tests/integration/` | none | yes | module seams: API pipeline with tiny model, DB roundtrips, registry ops |
| Golden | `tests/golden/` | none | yes | keydetector rule cases (positive + negative per rule) — the axis-3 regression net |
| Acceptance | `tests/acceptance/` | `acceptance` | no (local `make gate PHASE=N`) | phase exit gates; may be slow, may need datasets/models |
| Slow/GPU/Network | anywhere | `slow` / `gpu` / `network` | no | real model sizes, training smoke, latency budgets, downloads |

CI command: `pytest -m "not slow and not gpu and not network and not acceptance"`. A phase gate: `pytest tests/acceptance/test_phase{N}_*.py -m acceptance` (plus its `slow` companions run at least once, evidence pasted into STATUS.md).

## 2. Fixture policy

- All audio fixtures **generated at test time** by `tests/conftest.py` factories, seeded (`seed=1337` default): `tone(freq, dur)`, `chirp()`, `white_noise()`, `pink_noise()`, `babble_like()` (sum of AM-modulated band-limited noise), `silence()`, `speechlike()` (formant-ish AM tone bursts — enough for VAD/mixer tests, not for WER tests).
- WER-bearing tests use tiny TTS clips generated once by `make fixtures` into `data/fixtures/` (gitignored) — tests skip with a clear message if absent.
- No committed binary > 100 KB. Golden cases are YAML text.
- Model in CI: `whisper-tiny`. Any test needing small/medium is `slow`.

## 3. Determinism & tolerances

- Every stochastic component takes a seed; tests assert bit-identical reruns where the contract promises it (mixer, dataset builder).
- Tolerances (from [01 §7](../01-conventions.md)): SNR ±0.5 dB; WER asserted to 3 decimals on fixed toy sets; CT2-vs-HF parity ≤ 1.0 absolute WER point; latency assertions only in `slow` tests with 2× headroom over budget.
- Never assert on floating-point equality of audio arrays across platforms except same-process determinism tests.

## 4. Test doubles

- `MockJudge` — scripted verdict sequences from YAML fixtures; **no test may call the Anthropic API** (network marker exists for dataset downloads only).
- Engine spy — wraps `WhisperEngine` counting invocations (for "VAD gate prevented ASR" assertions).
- Fixture DB — `tests/factories.py` builds populated SQLite in tmpdir.
- Simulation seams — the harvester accepts injected (hypothesis, reference) pairs via a documented test-only entry point (phase 6.8); the seam is part of the public test contract, kept stable.

## 5. Gate discipline

- `make gate PHASE=N` fails if any acceptance test for phase N fails **or** if STATUS.md doesn't mark phases < N done.
- Gate evidence (metric tables, report paths) is pasted into STATUS.md by the builder at phase close. Numbers in STATUS.md must match report files (spot-checked by `test_status_consistency` — a tiny test that parses STATUS.md run ids and confirms the referenced report files exist).
- Regression direction: once a phase is closed, its acceptance tests join the permanent suite — later phases must keep them green. A later phase may *supersede* a gate only via a DECISIONS.md entry that names the replacement test.

## 6. What is deliberately NOT tested

- Absolute WER values of public models (they drift with model releases) — gates are always *relative* (vs baseline run ids or vs `none`-chain controls).
- Audio perceptual quality — only ΔWER/ΔKER matter.
- Prefect scheduling itself — flows are tested by direct invocation; cron wiring is smoke-checked manually per runbook.
