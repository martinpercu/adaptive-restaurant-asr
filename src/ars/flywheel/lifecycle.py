"""Confusion-rule lifecycle automation (plan/phases/phase-6 §6.4).

candidate → (human review) → approved (log-only in prod) → after ≥ 2 weeks of
fired-correctly evidence with no conflicts → active (automatic flip by the cycle, logged).
Pure over (rules, evidence). Never auto-promotes on judge evidence alone (§6 pitfall).
"""

from __future__ import annotations

MIN_WEEKS = 2
MIN_CORRECT_FIRES = 5


def advance_rules(
    rules: list[dict],
    evidence: dict[str, dict],
    min_weeks: int = MIN_WEEKS,
    min_correct: int = MIN_CORRECT_FIRES,
) -> list[tuple[str, str, str]]:
    """Flip eligible `approved` rules to `active`, in place. evidence[rule_id] =
    {weeks_approved, correct_fires, conflicts}. Returns (rule_id, old, new) transitions."""
    flips: list[tuple[str, str, str]] = []
    for rule in rules:
        if rule.get("status") != "approved":
            continue
        ev = evidence.get(rule["id"], {})
        if (
            ev.get("weeks_approved", 0) >= min_weeks
            and ev.get("correct_fires", 0) >= min_correct
            and ev.get("conflicts", 0) == 0
        ):
            rule["status"] = "active"
            flips.append((rule["id"], "approved", "active"))
    return flips
