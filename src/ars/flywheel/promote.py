"""Shadow-gated promotion (plan/phases/phase-6 §6.5). Gate logic + atomic registry flip."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ars.registry import ModelRegistry

MAX_LOGPROB_WORSE = 0.1


@dataclass
class PromotionGate:
    promote: bool
    reasons: list[str]


def promotion_gate(
    shadow: dict, latency_budget_ms: float = 3000.0, regression_ok: bool = True
) -> PromotionGate:
    """shadow: {prod_wer, cand_wer, prod_logprob, cand_logprob, cand_latency_ms}. Each gate
    failing alone blocks promotion (§6.5)."""
    reasons: list[str] = []
    if shadow["cand_wer"] > shadow["prod_wer"]:
        reasons.append(f"cand WER {shadow['cand_wer']:.3f} > prod {shadow['prod_wer']:.3f}")
    if shadow["cand_logprob"] < shadow["prod_logprob"] - MAX_LOGPROB_WORSE:
        reasons.append("cand avg_logprob worse by > 0.1")
    if shadow.get("cand_latency_ms", 0) > latency_budget_ms:
        reasons.append("cand latency over budget")
    if not regression_ok:
        reasons.append("regression suite failed")
    return PromotionGate(promote=not reasons, reasons=reasons)


def atomic_save(registry: ModelRegistry, path: str | Path) -> None:
    """Write the registry via a temp file + atomic rename (a crash mid-flip leaves the old
    valid file intact)."""
    path = Path(path)
    payload = registry.model_dump(mode="json")
    import json  # noqa: PLC0415

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def promote_or_archive(
    shadow: dict, version: str, registry_path: str | Path, regression_ok: bool = True
) -> PromotionGate:
    gate = promotion_gate(shadow, regression_ok=regression_ok)
    reg = ModelRegistry.load(registry_path)
    if gate.promote:
        reg.promote(version, promoted_by="flywheel")
        atomic_save(reg, registry_path)
    else:
        entry = reg.get(version)
        if entry is not None:
            entry.stage = "retired"  # archived candidate
            atomic_save(reg, registry_path)
    return gate
