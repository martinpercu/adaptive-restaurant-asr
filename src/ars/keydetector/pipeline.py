"""Keydetector orchestration (plan/phases/phase-5 §5.3).

Two mechanisms in order: (1) menu lexicon recovery, (2) confusion-pair rules. Lexicon
runs first and its spans are locked (precedence). `replace` applies active rules;
`log_only` records corrections in the trace but leaves text bit-identical.
"""

from __future__ import annotations

from pathlib import Path

from ars.contracts import Correction
from ars.keydetector.lexicon import LexiconIndex, load_menu
from ars.keydetector.rules import RuleEngine, load_rules


class Keydetector:
    def __init__(self, menu: dict, rules_by_lang: dict[str, list], mode: str = "log_only") -> None:
        self.mode = mode
        self._lex = {lang: LexiconIndex(menu).build(lang) for lang in ("es", "en")}
        self._rules = {lang: RuleEngine(rules_by_lang.get(lang, [])) for lang in ("es", "en")}

    def correct(self, text: str, lang: str) -> tuple[str, list[Correction]]:
        if not text.strip() or lang not in self._lex:
            return text, []
        orig = text.split()
        # (1) lexicon
        lex = self._lex[lang].match(orig, lang)
        locked = {k for (i, j, _r, _c) in lex for k in range(i, j)}
        # (2) rules — active_only in replace mode; approved log only
        active_only = self.mode == "replace"
        rule_hits = self._rules[lang].apply(orig, lang, active_only, locked)

        all_hits = sorted(lex + rule_hits, key=lambda h: h[0])
        corrections = [h[3] for h in all_hits]
        if self.mode != "replace":
            return text, corrections  # log_only: text unchanged
        return _apply(orig, all_hits), corrections

    @classmethod
    def from_settings(cls, settings) -> Keydetector:
        menu = load_menu(settings.keydetector.menu_dir, "demo")
        rules_dir = Path(settings.keydetector.rules_dir)
        rules = {}
        for lang in ("es", "en"):
            p = rules_dir / f"rules-{lang}.yaml"
            rules[lang] = load_rules(p) if p.exists() else []
        return cls(menu, rules, mode=settings.keydetector.mode)


def _apply(orig_tokens: list[str], hits: list[tuple[int, int, str, Correction]]) -> str:
    toks: list[str | None] = list(orig_tokens)
    for i, j, replacement, _c in hits:
        toks[i] = replacement
        for k in range(i + 1, j):
            toks[k] = None
    return " ".join(t for t in toks if t is not None)
