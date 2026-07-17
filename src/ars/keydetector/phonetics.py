"""Phonetic keys for matching (plan/03-data-spec.md §7).

Two homophone-folding key functions, used by KER (phase 1 eval) and the confusion
engine (phase 5):
  - `es_key`: rule-based Spanish folding (seseo etc.) — deterministic.
  - `en_key`: metaphone via jellyfish.
Words with the same key are treated as phonetically equal.
"""

from __future__ import annotations

import re

import jellyfish

from ars.eval.normalize import fold_diacritics

_DOUBLED_RE = re.compile(r"(.)\1+")
_CE_CI_RE = re.compile(r"c(?=[ei])")
_GE_GI_RE = re.compile(r"g(?=[ei])")


def es_key(word: str) -> str:
    """Fold a Spanish word to a seseo-aware phonetic key (03 §7).

    Ordered so homophones collide: bocina->bosina, cocina->kosina, vaso->baso,
    bazo->baso, hielo->ielo, cielo->sielo.
    """
    s = word.lower()
    # strip diacritics but preserve ñ (folded to n only at the end)
    s = s.replace("ñ", "\x00")
    s = fold_diacritics(s)
    s = s.replace("\x00", "ñ")

    s = s.replace("v", "b")
    s = s.replace("x", "ks")  # orthographic x before the ch-fold claims the 'x' key
    s = s.replace("ch", "x")  # ch folds to a distinct 'x' key
    s = s.replace("h", "")  # silent h dropped (ch already handled)
    s = s.replace("ll", "y")
    s = s.replace("z", "s")
    s = _CE_CI_RE.sub("s", s)  # c before e/i -> s
    s = s.replace("qu", "k")
    s = s.replace("c", "k")  # remaining c -> k
    s = _GE_GI_RE.sub("j", s)  # g before e/i -> j
    s = s.replace("w", "u")
    s = s.replace("ñ", "n")
    s = _DOUBLED_RE.sub(r"\1", s)  # collapse doubled letters
    return s


def en_key(word: str) -> str:
    """English phonetic key: metaphone (03 §7). Falls back to the folded word."""
    key = jellyfish.metaphone(word.lower())
    return key or fold_diacritics(word.lower())


def phonetic_key(word: str, lang: str) -> str:
    return es_key(word) if lang == "es" else en_key(word)
