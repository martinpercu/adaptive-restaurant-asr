"""Menu lexicon matcher (plan/phases/phase-5 §5.2).

Indexes menu terms (names + aliases + service terms) by normalized form and phonetic
key, multiword items as units (up to trigram). Over a hypothesis's n-grams: exact match
skips; phonetic-key equality OR RapidFuzz ratio >= 88 recovers the canonical menu form.
Guards against over-correction: stopword skip, length guard, one correction per span.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from rapidfuzz import fuzz

from ars.contracts import Correction
from ars.eval.normalize import normalize, tokens
from ars.keydetector.phonetics import phonetic_key

# Single tokens need a high bar (short words collide easily); multiword items tolerate a
# single typo at a slightly lower ratio. Phonetic swaps are the context-gated rules' job.
FUZZY_MIN = 92
FUZZY_MIN_MULTI = 90
MAX_LEN_DIFF = 3
STOPWORDS = {
    "es": {
        "de",
        "la",
        "el",
        "un",
        "una",
        "y",
        "con",
        "por",
        "para",
        "me",
        "da",
        "das",
        "que",
        "los",
        "las",
        "en",
        "al",
        "lo",
        "su",
        "es",
        "no",
        "si",
        "a",
        "o",
    },
    "en": {
        "the",
        "a",
        "an",
        "and",
        "of",
        "to",
        "i",
        "me",
        "can",
        "get",
        "please",
        "with",
        "for",
        "is",
        "it",
        "my",
        "you",
        "do",
        "have",
        "want",
    },
}


class LexiconIndex:
    def __init__(self, menu: dict) -> None:
        self.by_norm: dict[str, str] = {}  # normalized term -> canonical
        self.by_key: dict[str, str] = {}  # phonetic-key seq -> canonical
        self.terms_by_n: dict[int, list[tuple[str, str, str]]] = {1: [], 2: [], 3: []}
        self._lang = None
        self._menu = menu

    def build(self, lang: str) -> LexiconIndex:
        self._lang = lang
        for term, canonical in self._iter_terms(lang):
            toks = tokens(term, lang)
            n = len(toks)
            if n == 0 or n > 3:
                continue
            norm = " ".join(toks)
            key = " ".join(phonetic_key(t, lang) for t in toks)
            self.by_norm.setdefault(norm, canonical)
            self.by_key.setdefault(key, canonical)
            self.terms_by_n[n].append((norm, key, canonical))
        return self

    def _iter_terms(self, lang: str):
        for item in self._menu.get("items", []):
            canonical = item["name"][lang]
            yield canonical, canonical
            for alias in item.get("aliases", {}).get(lang, []):
                yield alias, canonical
        for svc in self._menu.get("service_terms", []):
            yield svc[lang], svc[lang]

    def match(
        self, orig_tokens: list[str], lang: str, locked: set[int] | None = None
    ) -> list[tuple[int, int, str, Correction]]:
        used: set[int] = set(locked or set())
        norm = [normalize(t, lang) for t in orig_tokens]
        keys = [phonetic_key(t, lang) for t in norm]
        out: list[tuple[int, int, str, Correction]] = []
        for n in (3, 2, 1):
            for i in range(len(orig_tokens) - n + 1):
                if any((i + k) in used for k in range(n)):
                    continue
                ngram_norm = " ".join(norm[i : i + n])
                ngram_key = " ".join(keys[i : i + n])
                if ngram_norm in self.by_norm:
                    for k in range(n):  # already a correct menu term — lock, no correction
                        used.add(i + k)
                    continue
                if n == 1 and norm[i] in STOPWORDS.get(lang, set()):
                    continue
                canonical = self._recover(ngram_norm, ngram_key, n, lang)
                if canonical is None:
                    continue
                conf = self._confidence(ngram_norm, canonical, lang)
                before = " ".join(orig_tokens[i : i + n])
                out.append(
                    (
                        i,
                        i + n,
                        canonical,
                        Correction(
                            rule_id="lexicon",
                            span=(i, i + n),
                            before=before,
                            after=canonical,
                            confidence=conf,
                        ),
                    )
                )
                for k in range(n):
                    used.add(i + k)
        return out

    def _recover(self, ngram_norm, ngram_key, n, lang):
        # Lexicon recovery is strong-fuzzy only (>= FUZZY_MIN). Phonetic-key equality
        # over-corrects common words/bigrams (bell->bill, "had to"->tea) on general text —
        # known phonetic confusions are handled by the context-gated confusion rules instead.
        threshold = FUZZY_MIN if n == 1 else FUZZY_MIN_MULTI
        best, best_ratio = None, 0.0
        for term_norm, _key, canonical in self.terms_by_n[n]:
            if abs(len(term_norm) - len(ngram_norm)) > MAX_LEN_DIFF:
                continue
            r = fuzz.ratio(ngram_norm, term_norm)
            if r >= threshold and r > best_ratio:
                best, best_ratio = canonical, r
        return best

    def _confidence(self, ngram_norm, canonical, lang) -> float:
        return round(fuzz.ratio(ngram_norm, normalize(canonical, lang)) / 100.0, 3)


def load_menu(menu_dir: str | Path, store_id: str = "demo") -> dict:
    return yaml.safe_load((Path(menu_dir) / f"{store_id}.yaml").read_text(encoding="utf-8"))
