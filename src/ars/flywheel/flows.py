"""Weekly cycle orchestration (plan/phases/phase-6 §6.7).

Plain Python so it is tested by direct invocation (Prefect scheduling itself is not tested,
per the test strategy). If Prefect is installed the flow can be decorated; every step is
idempotent and resumable via `cycle_state` keyed by cycle_id. A hung step must not wedge the
cycle — callers pass already-computed judge/shadow inputs in simulation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ars.db import load_cycle_step, save_cycle_step
from ars.flywheel.harvester import HarvestConfig, harvest
from ars.flywheel.pair_miner import mine
from ars.flywheel.promote import promote_or_archive
from ars.judge import route_verdict

RETRAIN_POOL_HOURS = 5.0


def _step(conn, cycle_id, name, fn):
    """Run a step once; on resume, return the persisted result (idempotent)."""
    cached = load_cycle_step(conn, cycle_id, name)
    if cached is not None:
        return cached
    result = fn()
    save_cycle_step(conn, cycle_id, name, result)
    return result


def weekly_cycle(
    conn,
    settings,
    judge,
    cycle_id: str,
    *,
    menu_terms: dict | None = None,
    injected_pairs: list[tuple[str, str, str]] | None = None,
    pool_hours: float = 0.0,
    candidate_version: str | None = None,
    shadow_metrics: dict | None = None,
    regression_ok: bool = True,
    noise_profiles: dict | None = None,
) -> dict:
    reports = Path(settings.paths.reports) / "flywheel" / cycle_id
    reports.mkdir(parents=True, exist_ok=True)

    # 1. harvest
    worklist = _step(conn, cycle_id, "harvest", lambda: harvest(conn, HarvestConfig()))

    # 2. judge + route (build requests from harvested transcriptions)
    def _judge():
        counts = {"training": 0, "training+miner": 0, "review": 0}
        pairs: list[list] = []
        for item in worklist:
            tid = item["transcription_id"]
            if tid is None:
                continue
            row = conn.execute(
                "SELECT raw_text, final_text FROM transcriptions WHERE id=?", (tid,)
            ).fetchone()
            if not row:
                continue
            transcript = row[1] or row[0]
            verdict = judge.judge_one({"transcript": transcript, "lang": "es", "menu_items": []})
            route = route_verdict(verdict)
            counts[route] += 1
            if route == "training+miner" and verdict.corrected_reference:
                pairs.append([transcript, verdict.corrected_reference, "es"])
        return {"counts": counts, "pairs": pairs}

    judged = _step(conn, cycle_id, "judge", _judge)

    # 3. mine confusion pairs (injected test-seam pairs + judged pairs)
    def _mine():
        all_pairs = [tuple(p) for p in judged["pairs"]] + list(injected_pairs or [])
        mined = mine(all_pairs, {k: set(v) for k, v in (menu_terms or {}).items()})
        out = []
        for i, m in enumerate(mined):
            rule = m.to_rule(f"mined-{cycle_id}-{i:03d}")
            (reports / f"golden-skeleton-{rule['id']}.json").write_text(
                json.dumps(m.golden_skeleton(), indent=2)
            )
            out.append(rule)
        return {"candidates": out}

    mined = _step(conn, cycle_id, "mine", _mine)

    # 4. noise-stats loop: flag a store whose dominant subtype share doubled
    def _noise():
        flags = []
        for store, prof in (noise_profiles or {}).items():
            if prof.get("dominant_share", 0) >= 2 * prof.get("prev_dominant_share", 1e9):
                flags.append(
                    {
                        "store": store,
                        "subtype": prof.get("dominant"),
                        "share": prof["dominant_share"],
                    }
                )
        return {"flags": flags}

    noise = _step(conn, cycle_id, "noise", _noise)

    # 5. retrain trigger
    retrain = _step(
        conn,
        cycle_id,
        "retrain",
        lambda: {"triggered": pool_hours >= RETRAIN_POOL_HOURS, "pool_hours": pool_hours},
    )

    # 6. shadow → promote/archive
    def _promote():
        if not (candidate_version and shadow_metrics):
            return {"decision": "no-candidate"}
        gate = promote_or_archive(
            shadow_metrics,
            candidate_version,
            Path(settings.paths.models) / "registry.json",
            regression_ok,
        )
        return {"decision": "promoted" if gate.promote else "archived", "reasons": gate.reasons}

    promotion = _step(conn, cycle_id, "promote", _promote)

    cycle = {
        "cycle_id": cycle_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harvest": {"n": len(worklist)},
        "judge": judged["counts"],
        "mined_candidates": [c["id"] for c in mined["candidates"]],
        "noise_flags": noise["flags"],
        "retrain": retrain,
        "promotion": promotion,
    }
    (reports / "CYCLE.md").write_text(_cycle_md(cycle), encoding="utf-8")
    (reports / "cycle.json").write_text(json.dumps(cycle, indent=2), encoding="utf-8")
    return cycle


def _cycle_md(c: dict) -> str:
    lines = [
        f"# Flywheel cycle {c['cycle_id']}",
        "",
        f"- created: {c['created_at']}",
        f"- harvested: {c['harvest']['n']} utterances",
        f"- judge routing: {c['judge']}",
        f"- mined candidate rules: {c['mined_candidates'] or 'none'}",
        f"- noise-shift flags: {c['noise_flags'] or 'none'}",
        f"- retrain triggered: {c['retrain']['triggered']} (pool {c['retrain']['pool_hours']} h)",
        f"- promotion decision: {c['promotion']['decision']}",
        "",
        "## Judge cost estimate",
        "- batch API, ~50% price; ~1 call/harvested item × input+output tokens "
        "(see judge.provider/model). Fill actuals from the batch job receipt.",
        "",
        "## Runbook",
        "- rollback model: `python -m ars.registry rollback`",
        "- retire a bad rule: set its `status: retired` in configs/rules and redeploy",
        "- pause the flywheel: `flywheel.enabled: false` in configs/default.yaml",
    ]
    return "\n".join(lines) + "\n"
