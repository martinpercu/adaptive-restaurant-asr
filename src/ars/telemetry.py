"""Request telemetry — one JSONL line per request (plan/03-data-spec.md §9).

Privacy (phase 7): never log raw audio or full customer transcripts at INFO. This
writer records decisions/metrics only; transcript text is not part of the line.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def telemetry_line(
    *,
    trace_id: str,
    store_id: str | None,
    duration_s: float,
    speech_ratio: float,
    noise_pred: str | None,
    noise_confidence: float,
    chain_applied: list[str],
    language: str,
    avg_logprob: float,
    guard_flags: list[str],
    rules_fired: list[str],
    latency_ms: dict[str, float],
    model_version: str | None,
) -> dict:
    return {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "trace_id": trace_id,
        "store_id": store_id,
        "duration_s": round(duration_s, 3),
        "speech_ratio": round(speech_ratio, 3),
        "noise_pred": noise_pred,
        "noise_confidence": round(noise_confidence, 3),
        "chain_applied": chain_applied,
        "language": language,
        "avg_logprob": round(avg_logprob, 3),
        "guard_flags": guard_flags,
        "rules_fired": rules_fired,
        "latency_ms": {k: round(v, 1) for k, v in latency_ms.items()},
        "model_version": model_version,
    }


def write_telemetry(line: dict, telemetry_dir: str | Path) -> Path:
    day = line["ts"][:10]
    path = Path(telemetry_dir) / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path
