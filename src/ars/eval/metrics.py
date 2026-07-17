"""WER / CER / KER / hallucination metrics (plan/03-data-spec.md §6).

WER/CER via jiwer (>=3 `process_words`/`process_characters`) on normalized text —
one wrapper, not scattered calls (phase-1 pitfall). KER uses exact-normalized or
equal-phonetic-key matching per keyword.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jiwer

from ars.eval.normalize import normalize, tokens
from ars.keydetector.phonetics import phonetic_key


@dataclass
class UttRecord:
    ref: str
    hyp: str
    lang: str
    keywords: list[str] = field(default_factory=list)
    guard_fired: bool = False
    avg_logprob: float = 0.0


def wer_cer(refs: list[str], hyps: list[str], lang: str) -> tuple[float | None, float | None]:
    """Corpus WER and CER on normalized text. None when there is no non-empty reference."""
    pairs = [(normalize(r, lang), normalize(h, lang)) for r, h in zip(refs, hyps, strict=True)]
    pairs = [(r, h) for r, h in pairs if r]  # jiwer errors on empty references
    if not pairs:
        return None, None
    r_list = [r for r, _ in pairs]
    h_list = [h for _, h in pairs]
    wer = jiwer.process_words(r_list, h_list).wer
    cer = jiwer.process_characters(r_list, h_list).cer
    return round(wer, 3), round(cer, 3)


def keyword_recovered(keyword: str, hyp_tokens: list[str], hyp_keys: list[str], lang: str) -> bool:
    kw_tokens = tokens(keyword, lang)
    if not kw_tokens:
        return False
    n = len(kw_tokens)
    kw_keys = [phonetic_key(t, lang) for t in kw_tokens]
    for i in range(len(hyp_tokens) - n + 1):
        if hyp_tokens[i : i + n] == kw_tokens or hyp_keys[i : i + n] == kw_keys:
            return True
    return False


def ker(records: list[UttRecord], lang: str) -> tuple[float | None, int]:
    """Keyword Error Rate = 1 - recovered/total over all reference keywords."""
    total = 0
    recovered = 0
    for rec in records:
        if not rec.keywords:
            continue
        hyp_tokens = tokens(rec.hyp, lang)
        hyp_keys = [phonetic_key(t, lang) for t in hyp_tokens]
        for kw in rec.keywords:
            total += 1
            if keyword_recovered(kw, hyp_tokens, hyp_keys, lang):
                recovered += 1
    if total == 0:
        return None, 0
    return round(1.0 - recovered / total, 3), total


def hallucination_rate(records: list[UttRecord], lang: str) -> float:
    """Fraction with non-empty hyp on empty ref, or where the repetition guard fired."""
    if not records:
        return 0.0
    hits = 0
    for rec in records:
        ref_empty = not normalize(rec.ref, lang)
        hyp_nonempty = bool(normalize(rec.hyp, lang))
        if (ref_empty and hyp_nonempty) or rec.guard_fired:
            hits += 1
    return round(hits / len(records), 3)


def compute_metrics(records: list[UttRecord], lang: str) -> dict:
    """Per-language metric bundle matching the sensitivity/eval schema (03 §3)."""
    wer, cer = wer_cer([r.ref for r in records], [r.hyp for r in records], lang)
    k, n_kw = ker(records, lang)
    logprobs = [r.avg_logprob for r in records]
    return {
        "wer": wer,
        "cer": cer,
        "ker": k,
        "hallucination_rate": hallucination_rate(records, lang),
        "n_utts": len(records),
        "n_keywords": n_kw,
        "avg_logprob_mean": round(sum(logprobs) / len(logprobs), 3) if logprobs else None,
    }
