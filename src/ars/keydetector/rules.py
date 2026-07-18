"""Confusion-rule engine (plan/phases/phase-5 §5.3, plan/03-data-spec.md §4).

Loads curated confusion rules with a lifecycle (candidate→approved→active→retired),
validates them hard on load, and fires them on normalized tokens with context gates.
`active` rules fire in replace mode; `approved` fire only in log_only. Over-correction is
worse than under-correction — every gate here exists to prevent it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ars.contracts import Correction
from ars.eval.normalize import fold_diacritics, normalize


def _fold(token: str, lang: str) -> str:
    """Match key: normalized + diacritics folded (es ASR may or may not emit accents)."""
    return fold_diacritics(normalize(token, lang))


Status = Literal["candidate", "approved", "active", "retired"]


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = "seed"
    evidence_count: int = 0
    added: str = ""


class ConfusionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    lang: str
    wrong: str  # surface form the ASR outputs
    right: str  # replacement
    scope: Literal["word", "phrase"] = "word"
    context_any: list[str] = Field(default_factory=list)
    context_none: list[str] = Field(default_factory=list)
    status: Status = "candidate"
    provenance: Provenance = Field(default_factory=Provenance)
    notes: str = ""


def load_rules(path: str | Path) -> list[ConfusionRule]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    rules = [ConfusionRule.model_validate(r) for r in raw]
    ids = [r.id for r in rules]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate rule ids in {path}: {sorted(dupes)}")
    return rules


def _norm_tokens(tokens: list[str], lang: str) -> list[str]:
    return [_fold(t, lang) for t in tokens]


def _in_window(term: str, window: list[str], lang: str) -> bool:
    """Whole-token (or contiguous-token phrase) presence within the context window."""
    parts = _fold(term, lang).split()
    if len(parts) == 1:
        return parts[0] in window  # token membership, not substring
    joined = " ".join(window)
    return " ".join(parts) in joined


def _context_ok(rule: ConfusionRule, norm_tokens: list[str], i: int, lang: str) -> bool:
    if not rule.context_any and not rule.context_none:
        return True
    window = norm_tokens[max(0, i - 3) : i + 4]
    if rule.context_none and any(_in_window(c, window, lang) for c in rule.context_none):
        return False
    if rule.context_any:
        return any(_in_window(c, window, lang) for c in rule.context_any)
    return True


class RuleEngine:
    """Fires confusion rules on a token stream. `active_only` = replace mode."""

    def __init__(self, rules: list[ConfusionRule]) -> None:
        self.rules = [r for r in rules if r.status in ("active", "approved")]

    def apply(
        self, orig_tokens: list[str], lang: str, active_only: bool, locked: set[int] | None = None
    ) -> list[tuple[int, int, str, Correction]]:
        """Return (start, end, replacement, Correction) for firing rules. `locked` = spans
        already consumed by the lexicon (precedence). One rule per span; no self re-trigger."""
        locked = set(locked or set())
        norm = _norm_tokens(orig_tokens, lang)
        out: list[tuple[int, int, str, Correction]] = []
        used: set[int] = set(locked)
        for rule in self.rules:
            if rule.lang != lang:
                continue
            if active_only and rule.status != "active":
                continue  # approved rules log but do not replace
            wrong_norm = _fold(rule.wrong, lang).split()
            n = len(wrong_norm)
            for i in range(len(norm) - n + 1):
                if any((i + k) in used for k in range(n)):
                    continue
                if norm[i : i + n] != wrong_norm:
                    continue
                if not _context_ok(rule, norm, i, lang):
                    continue
                replacement = _match_case(orig_tokens[i], rule.right)
                out.append(
                    (
                        i,
                        i + n,
                        replacement,
                        Correction(
                            rule_id=rule.id,
                            span=(i, i + n),
                            before=" ".join(orig_tokens[i : i + n]),
                            after=replacement,
                            confidence=1.0,
                        ),
                    )
                )
                for k in range(n):
                    used.add(i + k)
                break  # one firing location per rule pass is enough for a token span
        return out


def _match_case(source: str, replacement: str) -> str:
    """Preserve the source token's casing pattern on the replacement (POS cross-check needs it)."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement
