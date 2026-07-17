# Phase 6 — The Autonomous Flywheel

**Goal:** close the loop: production traces are harvested, judged, reviewed, and turned into (a) new confusion rules for axis 3, (b) fresh training data + retrain triggers for axis 2, (c) updated noise statistics for axis 1 — with shadow evaluation and gated blue-green promotion. Human involvement shrinks to a review queue. The whole cycle must run end-to-end against **simulated** production traffic before it ever sees real audio.
**Depends on:** phases 1–5 gates. **Estimated effort:** 8–12 days.

## Deliverables

- `src/ars/flywheel/harvester.py`, `judge/` client, `flywheel/review.py` (CLI), `flywheel/pair_miner.py`, `flywheel/shadow.py`, `flywheel/promote.py`, `flywheel/flows.py` (Prefect), `flywheel/simulate.py`.
- `models/registry.json` lifecycle fully automated (candidate → shadow → production, rollback).
- One complete simulated weekly cycle with evidence report `reports/flywheel/<cycle_id>/CYCLE.md`.

## Tasks

### 6.1 Harvester
Weekly window query over telemetry + DB: select utterances where any of — `avg_logprob < -0.6`; guard flags present; keydetector fired ≥ 2 rules; judge previously marked `wrong`; random 2% sample (control). Cap per cycle (config, default 500). Output: work list in DB.

### 6.2 LLM-as-a-Judge
Client per [03 §8](../03-data-spec.md): pluggable provider (`claude-sonnet-5` reference default; OpenAI `gpt-4o-mini` / `gpt-4.1-nano` low-cost alternates), structured outputs, pydantic-validated, one retry; the weekly cycle submits all judgments as one batch job on the configured provider (~50% price). Switching the production judge to a cheaper model requires the calibration gate from 03 §8. `MockJudge` for all tests. Judge receives transcript + store menu + POS ticket when available (`meta.pos_order_id` joined against a `pos_tickets` table — build the table + a CSV import CLI; in simulation the tickets are synthetic). Routing on verdict:
- `correct` + `order_core_match` → auto-labeled reference, goes to training pool.
- `minor_errors` with `corrected_reference` + `confidence ≥ 0.8` → training pool + confusion candidates to miner.
- else → `review_queue`.

### 6.3 Review queue CLI
`python -m ars.flywheel.review`: iterate pending items — play/point to audio path, show raw/final/judge suggestion — accept (with corrected text) / reject / skip. Writes resolution + promotes accepted references to the labeled pool. Must be usable over SSH (pure terminal).

### 6.4 Pair miner
Over labeled pairs (hypothesis, reference): `jiwer` alignment → substitution ops → aggregate by (normalized wrong, normalized right, lang) with context snapshots. Candidate rule when: `evidence_count ≥ 5`, estimated precision ≥ 0.9 (wrong→right consistent, reverse rare), `wrong` not a menu term. Emits rule with `status: candidate` + auto-generated golden case skeletons (positive from real contexts; negative template requiring human completion). Promotion path: candidate → (human review in queue) → approved (log-only in prod) → after ≥ 2 weeks with fired-correctly evidence and no conflicts → active (automatic flip by the cycle, logged). Existing `approved` seed rules from phase 5 ride the same evidence path.

### 6.5 Retrain trigger + shadow + promotion
- Trigger conditions (any): labeled pool grew ≥ N hours (default 5); NDI re-run shows a subtype's damage ↑ ≥ 20% vs last cycle; 4 weeks since last train. Calls the phase-4 pipeline unchanged (same gates).
- **Shadow:** `flywheel.shadow` runs the candidate CT2 model in parallel with production on live traffic (or simulation): same input, both outputs logged, only production's returned. Runs until ≥ 500 utterances or 7 days.
- **Promotion gates:** shadow WER proxy (judge-scored sample of ≥ 100) candidate ≤ production, candidate mean `avg_logprob` not worse by > 0.1, latency within budget, phase-4 regression suite green. Pass → atomic registry flip (blue-green: keep previous entry, `stage: previous`); fail → candidate archived with report. `python -m ars.registry rollback` restores `previous` instantly.

### 6.6 Noise stats loop (axis 1 feed)
Aggregate per-store classifier outputs weekly → `reports/noise_profile/<store>.json` (subtype distribution, trend). If a store's dominant subtype has no mapped mitigation or its share doubled, flag in cycle report → next sensitivity/effectiveness re-run prioritizes it.

### 6.7 Orchestration
Prefect flow `weekly_cycle`: harvest → judge → (wait for review SLA or timeout 48 h) → mine → maybe-retrain → shadow (async) → promote-or-archive → write `CYCLE.md` (all decisions + evidence links). Schedule: weekly cron + `make cycle` manual trigger. Every step idempotent and resumable (state in DB, keyed by `cycle_id`).

### 6.8 Simulation harness (the phase's proving ground)
`python -m ars.flywheel.simulate --days 7 --seed 1337`: generates synthetic "production" traffic through the real API: matrix-mixed audios + **injected known errors** — (a) a novel confusion pair *not* in the rules (e.g. TTS utterances where reference says "servilleta" but we corrupt audio so ASR outputs "silueta"; practical injection: bypass audio and inject hypothesis/reference pairs at the harvester boundary via a documented test seam), (b) a noise-shift for one store (CB share ×3), (c) synthetic POS tickets with 90% match rate. Then runs `weekly_cycle` with `MockJudge` configured to behave per a fixture script.

## Acceptance tests (`tests/acceptance/test_phase6_*.py`)

1. `test_harvester_selection` — fixture DB → exactly the specced selection + cap + control sample.
2. `test_judge_client_contract` — MockJudge: valid JSON path, invalid-then-retry path, hard-fail path; routing table per 6.2; provider selection: config switches implementation and each client builds the correct request shape (mocked transport, both providers).
3. `test_pair_miner_finds_injected_pair` — labeled fixtures containing 7× "silueta"→"servilleta" → candidate rule emitted with correct fields + golden skeletons; 4× → no rule; menu-term wrong → suppressed.
4. `test_rule_lifecycle_automation` — simulated weeks of evidence → approved→active flip happens (and doesn't without evidence).
5. `test_shadow_and_promotion_logic` — table-driven gate matrix (each gate failing alone → no promote); registry flip atomicity (crash mid-flip leaves valid file); rollback restores previous.
6. `test_cycle_idempotent_resume` — kill the flow after mining, rerun same `cycle_id` → completes without duplicating work.
7. (`slow`, the phase gate) `test_full_simulated_cycle` — run 6.8 end-to-end: asserts (a) injected confusion pair mined as candidate, (b) noise-shift flagged in cycle report, (c) retrain triggered iff pool threshold crossed (configure the sim so it is), (d) shadow report produced, (e) promotion decision matches the gates given the sim's metrics, (f) `CYCLE.md` complete.

## Exit checklist
- [ ] `make gate PHASE=6` green; `CYCLE.md` of the simulated cycle linked in STATUS.md.
- [ ] Judge cost estimate documented (tokens/cycle × price) in CYCLE.md.
- [ ] Runbook stubs exist: rollback model, retire bad rule, pause flywheel (`flywheel.enabled: false`).

## Pitfalls
- The judge is a labeler, not an oracle: never auto-promote a rule or model on judge evidence alone without the review/consistency thresholds above.
- POS tickets validate the *order core*, not verbatim speech — a matching ticket with differing transcript is still `minor_errors`, not `correct`.
- Shadow must never add latency to the production path: candidate inference is async off the request thread.
- All Prefect tasks need explicit timeouts; a hung judge call must not wedge the weekly cycle.
