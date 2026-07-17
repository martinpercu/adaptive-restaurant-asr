"""Evaluation: text normalization, WER/CER/KER, hallucination rate, reports (phases 1+)."""

from ars.eval.normalize import fold_diacritics, normalize, tokens

__all__ = ["normalize", "tokens", "fold_diacritics"]
