"""Drift monitors (plan/phases/phase-7 §7.2). Pure functions + alert emission.

Weekly checks: input drift (PSI of avg_logprob vs trailing reference), noise drift (subtype
share ×2), rule storm (fire rate ×3 vs mean → possible over-correction), quality alarm
(judge wrong+hallucination share). An alert is a structured log line (+ optional webhook).
"""

from __future__ import annotations

import numpy as np


def psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Population Stability Index between two samples. 0 = identical; > 0.2 = significant."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=bins)
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r = r / max(r.sum(), 1)
    c = c / max(c.sum(), 1)
    eps = 1e-6
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def check_drift(
    signals: dict,
    cfg,
) -> list[dict]:
    """signals may include: logprob_reference/logprob_current (lists); noise_shares
    {store: {subtype: (prev, cur)}}; rule_fire_rate {rule: (week, mean)}; judge_shares
    {wrong, hallucination, total}. Returns structured alerts (empty = healthy)."""
    alerts: list[dict] = []

    if "logprob_reference" in signals and "logprob_current" in signals:
        p = psi(signals["logprob_reference"], signals["logprob_current"])
        if p > cfg.psi_input_drift:
            alerts.append(
                {"type": "input_drift", "psi": round(p, 4), "threshold": cfg.psi_input_drift}
            )

    for store, subs in signals.get("noise_shares", {}).items():
        for subtype, (prev, cur) in subs.items():
            if prev > 0 and cur >= cfg.noise_share_multiplier * prev:
                alerts.append(
                    {
                        "type": "noise_drift",
                        "store": store,
                        "subtype": subtype,
                        "prev": prev,
                        "cur": cur,
                    }
                )

    for rule, (week, mean) in signals.get("rule_fire_rate", {}).items():
        if mean > 0 and week >= cfg.rule_storm_multiplier * mean:
            alerts.append({"type": "rule_storm", "rule": rule, "week": week, "mean": mean})

    js = signals.get("judge_shares")
    if js and js.get("total", 0) > 0:
        bad = (js.get("wrong", 0) + js.get("hallucination", 0)) / js["total"]
        if bad > cfg.judge_quality_max:
            alerts.append(
                {
                    "type": "quality_alarm",
                    "bad_share": round(bad, 4),
                    "threshold": cfg.judge_quality_max,
                }
            )

    return alerts


def emit_alerts(alerts: list[dict], webhook_url: str = "", logger=None) -> None:
    """Structured log each alert; optionally POST to a webhook (no vendor lock)."""
    import structlog  # noqa: PLC0415

    log = logger or structlog.get_logger("ars.drift")
    for a in alerts:
        log.warning("drift_alert", **a)
    if webhook_url and alerts:  # pragma: no cover - network
        import json
        import urllib.request

        req = urllib.request.Request(  # noqa: S310
            webhook_url,
            data=json.dumps({"alerts": alerts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
        except Exception:  # noqa: BLE001
            log.error("webhook_failed", url=webhook_url)
