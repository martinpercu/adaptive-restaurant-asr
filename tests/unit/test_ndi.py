from __future__ import annotations

import pandas as pd
import pytest

from ars.noise_lab.ndi import DEFAULT_WEIGHTS, cell_ndi, compute_ndi, top_subtypes


def _matrix() -> pd.DataFrame:
    rows = [
        {
            "lang": "es",
            "noise_subtype": None,
            "noise_level": None,
            "wer": 0.10,
            "ker": 0.20,
            "hallucination_rate": 0.0,
        },
        # AC — heavy damage, grows with level
        {
            "lang": "es",
            "noise_subtype": "AC",
            "noise_level": "05",
            "wer": 0.13,
            "ker": 0.25,
            "hallucination_rate": 0.0,
        },
        {
            "lang": "es",
            "noise_subtype": "AC",
            "noise_level": "10",
            "wer": 0.20,
            "ker": 0.40,
            "hallucination_rate": 0.05,
        },
        {
            "lang": "es",
            "noise_subtype": "AC",
            "noise_level": "15",
            "wer": 0.35,
            "ker": 0.60,
            "hallucination_rate": 0.10,
        },
        # AB — mild
        {
            "lang": "es",
            "noise_subtype": "AB",
            "noise_level": "05",
            "wer": 0.11,
            "ker": 0.21,
            "hallucination_rate": 0.0,
        },
        {
            "lang": "es",
            "noise_subtype": "AB",
            "noise_level": "10",
            "wer": 0.12,
            "ker": 0.22,
            "hallucination_rate": 0.0,
        },
        {
            "lang": "es",
            "noise_subtype": "AB",
            "noise_level": "15",
            "wer": 0.14,
            "ker": 0.24,
            "hallucination_rate": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_cell_ndi_hand_value():
    # AC/15: dwer=(0.35-0.10)/0.10=2.5, dker=(0.60-0.20)/0.20=2.0, halluc=0.10
    # ndi = 0.5*2.5 + 0.4*2.0 + 0.1*0.10 = 1.25 + 0.8 + 0.01 = 2.06
    assert cell_ndi(0.35, 0.60, 0.10, 0.10, 0.20, DEFAULT_WEIGHTS) == pytest.approx(2.06)


def test_ndi_ranking_order_and_values():
    ndi = compute_ndi(_matrix())
    assert ndi["baseline"]["es"]["wer"] == 0.10
    ranked = ndi["ranking"]
    assert [e["subtype"] for e in ranked] == ["AC", "AB"]  # AC dominates
    ac = ranked[0]
    assert ac["per_level"] == {"05": 0.25, "10": 0.905, "15": 2.06}
    assert ac["ndi"] == pytest.approx((0.25 + 0.905 + 2.06) / 3, abs=1e-3)


def test_top_subtypes():
    ndi = compute_ndi(_matrix())
    assert top_subtypes(ndi, "es", n=1) == ["AC"]


def test_missing_clean_row_raises():
    df = _matrix()
    df = df[df["noise_subtype"].notna()]
    with pytest.raises(ValueError):
        compute_ndi(df)


def test_ker_none_falls_back_to_wer_only():
    rows = [
        {
            "lang": "en",
            "noise_subtype": None,
            "noise_level": None,
            "wer": 0.10,
            "ker": None,
            "hallucination_rate": 0.0,
        },
        {
            "lang": "en",
            "noise_subtype": "BA",
            "noise_level": "10",
            "wer": 0.20,
            "ker": None,
            "hallucination_rate": 0.0,
        },
    ]
    ndi = compute_ndi(pd.DataFrame(rows))
    # dwer=1.0 -> ndi cell = 0.5*1.0 = 0.5 (ker term dropped)
    assert ndi["ranking"][0]["per_level"]["10"] == 0.5
