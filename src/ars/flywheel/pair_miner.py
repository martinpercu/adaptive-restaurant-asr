"""Confusion-pair miner (plan/phases/phase-6 §6.4).

Over labeled (hypothesis, reference) pairs: jiwer word alignment → substitution ops →
aggregate by (normalized wrong, normalized right, lang). A candidate rule is emitted when
evidence ≥ 5, estimated precision ≥ 0.9 (wrong→right consistent, reverse rare), and `wrong`
is not a menu term. Mined errors become *rules* (with golden skeletons), never training data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import jiwer

from ars.eval.normalize import normalize, tokens

MIN_EVIDENCE = 5
MIN_PRECISION = 0.9


@dataclass
class MinedRule:
    lang: str
    wrong: str
    right: str
    evidence_count: int
    precision: float
    contexts: list[str] = field(default_factory=list)

    def to_rule(self, rule_id: str) -> dict:
        return {
            "id": rule_id,
            "lang": self.lang,
            "wrong": self.wrong,
            "right": self.right,
            "scope": "word",
            "context_any": _context_terms(self.contexts, self.wrong, self.lang),
            "context_none": [],
            "status": "candidate",
            "provenance": {"source": "mined", "evidence_count": self.evidence_count, "added": ""},
            "notes": f"mined: precision={self.precision:.2f} over {self.evidence_count} obs",
        }

    def golden_skeleton(self) -> list[dict]:
        pos = self.contexts[0] if self.contexts else f"the {self.wrong}"
        return [
            {
                "rule_id": "PENDING",
                "input": pos,
                "expected": "PENDING",
                "mode": "log_only",
                "fires": True,
                "_note": "positive from real context",
            },
            {
                "rule_id": "PENDING",
                "input": f"TODO legit use of {self.wrong!r}",
                "expected": "TODO same",
                "mode": "log_only",
                "fires": False,
                "_note": "negative — human must complete",
            },
        ]


def _context_terms(contexts: list[str], wrong: str, lang: str) -> list[str]:
    """Frequent neighbour tokens across observed contexts (light gate for the candidate)."""
    counter: Counter[str] = Counter()
    for ctx in contexts:
        for t in tokens(ctx, lang):
            if t != normalize(wrong, lang):
                counter[t] += 1
    return [t for t, _ in counter.most_common(4)]


def mine(
    pairs: list[tuple[str, str, str]],
    menu_terms: dict[str, set[str]] | None = None,
    min_evidence: int = MIN_EVIDENCE,
    min_precision: float = MIN_PRECISION,
) -> list[MinedRule]:
    menu_terms = menu_terms or {}
    sub_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    wrong_totals: dict[tuple[str, str], int] = defaultdict(int)
    contexts: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for hyp, ref, lang in pairs:
        ref_toks = tokens(ref, lang)
        hyp_toks = tokens(hyp, lang)
        if not ref_toks or not hyp_toks:
            continue
        out = jiwer.process_words([" ".join(ref_toks)], [" ".join(hyp_toks)])
        for chunk in out.alignments[0]:
            if chunk.type != "substitute":
                continue
            # single-word substitutions only (the confusion-rule unit)
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                r_i, h_i = chunk.ref_start_idx + k, chunk.hyp_start_idx + k
                if h_i >= len(hyp_toks) or r_i >= len(ref_toks):
                    continue
                wrong, right = hyp_toks[h_i], ref_toks[r_i]
                if wrong == right:
                    continue
                key = (lang, wrong, right)
                sub_counts[key] += 1
                wrong_totals[(lang, wrong)] += 1
                contexts[key].append(hyp)

    mined: list[MinedRule] = []
    for (lang, wrong, right), count in sub_counts.items():
        if count < min_evidence:
            continue
        precision = count / max(wrong_totals[(lang, wrong)], 1)
        if precision < min_precision:
            continue
        if wrong in menu_terms.get(lang, set()):
            continue  # never mine a menu term as the wrong form
        mined.append(
            MinedRule(
                lang, wrong, right, count, round(precision, 3), contexts[(lang, wrong, right)][:5]
            )
        )
    mined.sort(key=lambda m: m.evidence_count, reverse=True)
    return mined
