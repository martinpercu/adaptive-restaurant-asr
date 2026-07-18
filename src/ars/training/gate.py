"""Promotion + regression gate logic (plan/phases/phase-4 §4.4, §4.6). Pure functions.

The gate isolates axis 2: candidate vs baseline on identical eval-noisy sets, preprocessing
and keydetector OFF. All three conditions must hold, per language, to promote.
"""

from __future__ import annotations

from dataclasses import dataclass

# thresholds (03 §5 / §4.6)
MIN_NOISY_WER_REL_IMPROVEMENT = 0.15
MAX_CLEAN_WER_REL_REGRESSION = 0.02
MAX_KEYWORD_RECALL_DROP = 0.01
MAX_GENERIC_WER_REL_REGRESSION = 0.02


def rel_improvement(base: float, cand: float) -> float:
    """(base - cand) / base — positive means candidate is better (lower WER)."""
    return (base - cand) / max(base, 1e-6)


def rel_regression(base: float, cand: float) -> float:
    """(cand - base) / base — positive means candidate is worse."""
    return (cand - base) / max(base, 1e-6)


@dataclass
class GateResult:
    promote: bool
    reasons: list[str]
    per_lang: dict


def promotion_decision(
    base: dict,
    candidate: dict,
    langs=("es", "en"),
    min_noisy_impr=MIN_NOISY_WER_REL_IMPROVEMENT,
    max_clean_reg=MAX_CLEAN_WER_REL_REGRESSION,
    max_kw_drop=MAX_KEYWORD_RECALL_DROP,
) -> GateResult:
    """base/candidate: {lang: {noisy_wer, clean_wer, keyword_recall}}. Promote iff ALL
    conditions hold for BOTH langs (one-lang-passes -> reject)."""
    reasons: list[str] = []
    per_lang: dict = {}
    ok = True
    for lang in langs:
        b, c = base[lang], candidate[lang]
        noisy_impr = rel_improvement(b["noisy_wer"], c["noisy_wer"])
        clean_reg = rel_regression(b["clean_wer"], c["clean_wer"])
        kw_drop = b["keyword_recall"] - c["keyword_recall"]
        pass_noisy = noisy_impr >= min_noisy_impr
        pass_clean = clean_reg <= max_clean_reg
        pass_kw = kw_drop <= max_kw_drop
        per_lang[lang] = {
            "noisy_wer_rel_improvement": round(noisy_impr, 4),
            "clean_wer_rel_regression": round(clean_reg, 4),
            "keyword_recall_drop": round(kw_drop, 4),
            "pass": pass_noisy and pass_clean and pass_kw,
        }
        if not pass_noisy:
            reasons.append(f"{lang}: noisy improvement {noisy_impr:.1%} < {min_noisy_impr:.0%}")
        if not pass_clean:
            reasons.append(f"{lang}: clean regression {clean_reg:.1%} > {max_clean_reg:.0%}")
        if not pass_kw:
            reasons.append(f"{lang}: keyword recall drop {kw_drop:.3f} > {max_kw_drop}")
        ok = ok and per_lang[lang]["pass"]
    return GateResult(promote=ok, reasons=reasons, per_lang=per_lang)


def regression_gate(
    base: dict,
    candidate: dict,
    langs=("es", "en"),
    max_kw_drop=MAX_KEYWORD_RECALL_DROP,
    max_generic_reg=MAX_GENERIC_WER_REL_REGRESSION,
) -> GateResult:
    """Anti-forgetting (§4.4): keyword recall drop <= 1% abs; generic clean WER reg <= 2% rel."""
    reasons: list[str] = []
    per_lang: dict = {}
    ok = True
    for lang in langs:
        b, c = base[lang], candidate[lang]
        kw_drop = b["keyword_recall"] - c["keyword_recall"]
        generic_reg = rel_regression(b["generic_wer"], c["generic_wer"])
        p = kw_drop <= max_kw_drop and generic_reg <= max_generic_reg
        per_lang[lang] = {
            "keyword_recall_drop": round(kw_drop, 4),
            "generic_wer_rel_regression": round(generic_reg, 4),
            "pass": p,
        }
        if kw_drop > max_kw_drop:
            reasons.append(f"{lang}: keyword recall drop {kw_drop:.3f} > {max_kw_drop}")
        if generic_reg > max_generic_reg:
            reasons.append(
                f"{lang}: generic WER regression {generic_reg:.1%} > {max_generic_reg:.0%}"
            )
        ok = ok and p
    return GateResult(promote=ok, reasons=reasons, per_lang=per_lang)
