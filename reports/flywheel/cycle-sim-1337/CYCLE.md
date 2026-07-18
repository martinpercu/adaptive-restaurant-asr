# Flywheel cycle cycle-sim-1337

- created: 2026-07-18T16:55:25Z
- harvested: 65 utterances
- judge routing: {'training': 15, 'training+miner': 47, 'review': 0}
- mined candidate rules: ['mined-cycle-sim-1337-000']
- noise-shift flags: [{'store': 'store-A', 'subtype': 'CB', 'share': 0.6}]
- retrain triggered: True (pool 6.0 h)
- promotion decision: promoted

## Judge cost estimate
- batch API, ~50% price; ~1 call/harvested item × input+output tokens (see judge.provider/model). Fill actuals from the batch job receipt.

## Runbook
- rollback model: `python -m ars.registry rollback`
- retire a bad rule: set its `status: retired` in configs/rules and redeploy
- pause the flywheel: `flywheel.enabled: false` in configs/default.yaml
