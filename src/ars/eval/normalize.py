"""Text normalization for WER/KER and keydetector matching (plan/03-data-spec.md §6).

Single implementation, imported everywhere metrics or matching happen:
  lowercase -> strip punctuation (keep intra-word apostrophes in `en`)
  -> collapse whitespace -> number words left as-is (no digit normalization in v1).
Diacritics are stripped **only for matching keys** (`fold_diacritics`), never in
displayed text.
"""

from __future__ import annotations

import re
import unicodedata

# Punctuation to remove. For `en` we protect intra-word apostrophes (don't -> don't).
_APOS = "'’"  # straight + typographic apostrophe
_WS_RE = re.compile(r"\s+")


def _strip_punct(text: str, lang: str) -> str:
    out_chars: list[str] = []
    for i, ch in enumerate(text):
        if ch.isalnum() or ch.isspace():
            out_chars.append(ch)
        elif lang == "en" and ch in _APOS:
            prev_alnum = i > 0 and text[i - 1].isalnum()
            next_alnum = i + 1 < len(text) and text[i + 1].isalnum()
            out_chars.append("'" if (prev_alnum and next_alnum) else " ")
        else:
            out_chars.append(" ")
    return "".join(out_chars)


def normalize(text: str, lang: str) -> str:
    """Normalized surface form used for WER/CER/KER (diacritics preserved)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _strip_punct(text, lang)
    return _WS_RE.sub(" ", text).strip()


def tokens(text: str, lang: str) -> list[str]:
    norm = normalize(text, lang)
    return norm.split() if norm else []


def fold_diacritics(text: str) -> str:
    """Strip accents for matching keys only (keep ñ as a distinct letter -> n handled by keys)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))
