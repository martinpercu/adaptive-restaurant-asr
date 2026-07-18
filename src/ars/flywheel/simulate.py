"""Simulation harness (plan/phases/phase-6 §6.8) — the phase's proving ground.

Generates synthetic "production" traffic + injected known errors, then runs one weekly_cycle
with a scripted MockJudge (no network). Injects: (a) a novel confusion pair NOT in the rules
(silueta→servilleta) via the harvester/miner test seam, (b) a noise-shift for one store
(CB share ×3), (c) synthetic POS tickets at 90% match. Everything is deterministic in `seed`.

    python -m ars.flywheel.simulate --days 7 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ars.config import DEFAULT_CONFIG, Settings
from ars.db import connect, insert_pos_ticket, insert_transcription, insert_utterance
from ars.flywheel.flows import weekly_cycle
from ars.judge import MockJudge
from ars.registry import ModelRegistry, RegistryEntry

INJECTED_PAIR = ("me da una silueta", "me da una servilleta", "es")


def _seed_registry(path: Path) -> None:
    reg = ModelRegistry()
    reg.add(RegistryEntry(version="0.1.0", stage="production", base_model="whisper-small"))
    reg.add(
        RegistryEntry(
            version="0.2.0", stage="candidate", base_model="whisper-small", adapter="lora-sim"
        )
    )
    reg.save(path)


def _populate(conn, seed: int, n: int = 200) -> None:
    rng = random.Random(seed)
    for i in range(n):
        uid = f"sim-{i:04d}"
        store = "store-A" if i % 2 == 0 else "store-B"
        insert_utterance(
            conn,
            utterance_id=uid,
            path=f"sim/{uid}.wav",
            lang="es",
            store_id=store,
            captured_at=None,
            duration_s=4.0,
            speech_ratio=0.8,
            meta={"pos_order_id": f"ord-{i}"},
        )
        low = rng.random() < 0.25
        guard = ["repetition_truncated"] if rng.random() < 0.1 else []
        tid = insert_transcription(
            conn,
            utterance_id=uid,
            model_version="0.1.0",
            raw_text="quiero una silueta" if low else "quiero un cafe",
            final_text="quiero una silueta" if low else "quiero un cafe",
            avg_logprob=-0.9 if low else -0.3,
            guard_flags=guard,
        )
        # synthetic POS ticket, 90% matching the order core
        matches = rng.random() < 0.9
        insert_pos_ticket(conn, f"ord-{i}", store, ["cafe"] if matches else ["te"])
        _ = tid


def simulate(
    settings: Settings, days: int = 7, seed: int = 1337, cycle_id: str | None = None
) -> dict:
    cycle_id = cycle_id or f"cycle-sim-{seed}"
    models = Path(settings.paths.models)
    models.mkdir(parents=True, exist_ok=True)
    _seed_registry(models / "registry.json")
    conn = connect(settings.paths.db)
    _populate(conn, seed)

    # MockJudge: label the low-confidence "silueta" utterances as minor_errors with a
    # corrected reference (feeds the miner); everything else correct.
    def script(request):
        t = request.get("transcript", "")
        if "silueta" in t:
            return {
                "verdict": "minor_errors",
                "corrected_reference": t.replace("silueta", "servilleta"),
                "confusion_candidates": [{"heard": "silueta", "intended": "servilleta"}],
                "order_core_match": True,
                "confidence": 0.9,
            }
        return {
            "verdict": "correct",
            "corrected_reference": None,
            "confusion_candidates": [],
            "order_core_match": True,
            "confidence": 0.95,
        }

    judge = MockJudge(script=script)

    # injected novel pair (test seam) x7 so the miner clears its evidence>=5 bar
    injected = [INJECTED_PAIR] * 7
    noise_profiles = {
        "store-A": {"dominant": "CB", "dominant_share": 0.6, "prev_dominant_share": 0.2}
    }
    shadow_metrics = {
        "prod_wer": 0.30,
        "cand_wer": 0.24,
        "prod_logprob": -0.35,
        "cand_logprob": -0.33,
        "cand_latency_ms": 2200,
    }

    cycle = weekly_cycle(
        conn,
        settings,
        judge,
        cycle_id,
        menu_terms={"es": {"cafe", "café", "sopa"}},
        injected_pairs=injected,
        pool_hours=6.0,
        candidate_version="0.2.0",
        shadow_metrics=shadow_metrics,
        regression_ok=True,
        noise_profiles=noise_profiles,
    )
    conn.close()
    _ = days
    return cycle


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS flywheel simulation")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    settings = Settings.load(args.config)
    cycle = simulate(settings, args.days, args.seed)
    print(json.dumps(cycle, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
