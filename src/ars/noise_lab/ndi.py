"""Noise Damage Index (plan/03-data-spec.md §3).

NDI per cell = w_wer·ΔWER_rel + w_ker·ΔKER_rel + w_h·hallucination_rate,
  Δx_rel = (x_noisy − x_clean) / max(x_clean, 0.01).
Subtype NDI = mean over canonical levels. The ranking steers phases 3/4/6.
Pure over a `matrix` DataFrame (one clean row per lang + one row per cell).
"""

from __future__ import annotations

import pandas as pd

CANONICAL_LEVELS = ("05", "10", "15")
DEFAULT_WEIGHTS = {"d_wer": 0.5, "d_ker": 0.4, "hallucination": 0.1}


def delta_rel(x: float, x_clean: float) -> float:
    return (x - x_clean) / max(x_clean, 0.01)


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def cell_ndi(
    wer: float,
    ker: float | None,
    hallucination_rate: float,
    clean_wer: float,
    clean_ker: float | None,
    weights: dict,
) -> float:
    d_wer = delta_rel(wer, clean_wer)
    if ker is None or clean_ker is None:
        d_ker = 0.0
    else:
        d_ker = delta_rel(ker, clean_ker)
    return (
        weights["d_wer"] * d_wer
        + weights["d_ker"] * d_ker
        + weights["hallucination"] * (hallucination_rate or 0.0)
    )


def compute_ndi(
    matrix: pd.DataFrame,
    weights: dict | None = None,
    canonical_levels: tuple[str, ...] = CANONICAL_LEVELS,
) -> dict:
    """Build the ndi.json body (03 §3): baseline per lang + ranking sorted by NDI desc."""
    weights = weights or DEFAULT_WEIGHTS
    baseline: dict[str, dict] = {}
    ranking: list[dict] = []

    for lang in sorted(matrix["lang"].unique()):
        sub = matrix[matrix["lang"] == lang]
        clean = sub[sub["noise_subtype"].isna()]
        if clean.empty:
            raise ValueError(f"matrix missing clean row for lang {lang!r}")
        clean_wer = _num(clean["wer"].iloc[0]) or 0.0
        clean_ker = _num(clean["ker"].iloc[0])
        baseline[lang] = {"wer": round(clean_wer, 4)}

        cells = sub[sub["noise_subtype"].notna()]
        for subtype in sorted(cells["noise_subtype"].unique()):
            per_level: dict[str, float] = {}
            for level in canonical_levels:
                row = cells[(cells["noise_subtype"] == subtype) & (cells["noise_level"] == level)]
                if row.empty:
                    continue
                r = row.iloc[0]
                val = cell_ndi(
                    _num(r["wer"]) or 0.0,
                    _num(r["ker"]),
                    _num(r["hallucination_rate"]) or 0.0,
                    clean_wer,
                    clean_ker,
                    weights,
                )
                per_level[level] = round(val, 4)
            if not per_level:
                continue
            subtype_ndi = sum(per_level.values()) / len(per_level)
            ranking.append(
                {
                    "subtype": subtype,
                    "lang": lang,
                    "ndi": round(subtype_ndi, 4),
                    "per_level": per_level,
                }
            )

    ranking.sort(key=lambda e: e["ndi"], reverse=True)
    return {"weights": dict(weights), "baseline": baseline, "ranking": ranking}


def top_subtypes(ndi: dict, lang: str, n: int = 3) -> list[str]:
    ranked = [e for e in ndi["ranking"] if e["lang"] == lang]
    return [e["subtype"] for e in ranked[:n]]
