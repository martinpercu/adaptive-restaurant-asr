"""Low-confidence harvester (plan/phases/phase-6 §6.1).

Selects the weekly work list from the operational DB: utterances with low avg_logprob,
guard flags, many keydetector rules fired, or a prior `wrong` judge verdict — plus a small
random control sample. Deterministic given (rows, seed). Capped per cycle.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass


@dataclass
class HarvestConfig:
    low_logprob: float = -0.6
    min_rules: int = 2
    control_frac: float = 0.02
    cap: int = 500


def harvest(
    conn: sqlite3.Connection, cfg: HarvestConfig | None = None, seed: int = 1337
) -> list[dict]:
    cfg = cfg or HarvestConfig()
    corr_counts = dict(
        conn.execute(
            "SELECT transcription_id, COUNT(*) FROM corrections GROUP BY transcription_id"
        ).fetchall()
    )
    verdicts = dict(conn.execute("SELECT transcription_id, verdict FROM judge_verdicts").fetchall())
    rows = conn.execute(
        "SELECT id, utterance_id, avg_logprob, guard_flags_json FROM transcriptions"
    ).fetchall()

    selected: list[dict] = []
    remaining: list[str] = []
    for tid, uid, avg_logprob, guard_json in rows:
        reasons = []
        if avg_logprob is not None and avg_logprob < cfg.low_logprob:
            reasons.append("low_logprob")
        if guard_json and json.loads(guard_json):
            reasons.append("guard_flags")
        if corr_counts.get(tid, 0) >= cfg.min_rules:
            reasons.append("rules_fired")
        if verdicts.get(tid) == "wrong":
            reasons.append("judge_wrong")
        if reasons:
            selected.append({"transcription_id": tid, "utterance_id": uid, "reasons": reasons})
        else:
            remaining.append(uid)

    # 2% random control sample from the not-otherwise-selected pool
    rng = random.Random(seed)
    n_control = int(round(len(remaining) * cfg.control_frac))
    for uid in rng.sample(remaining, min(n_control, len(remaining))):
        selected.append({"transcription_id": None, "utterance_id": uid, "reasons": ["control"]})

    # deterministic order, then cap
    selected.sort(key=lambda s: (s["utterance_id"], ",".join(s["reasons"])))
    return selected[: cfg.cap]
