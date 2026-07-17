"""Phase 2 exit gate (plan/phases/phase-2-noise-lab.md). Run via `make gate PHASE=2`."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from tests.conftest import babble_like, pink_noise, speechlike, white_noise

from ars.contracts import NoiseBankRow
from ars.noise_lab.mixer import mix, rms
from ars.noise_lab.ndi import compute_ndi
from ars.noise_lab.taxonomy import load_taxonomy

pytestmark = pytest.mark.acceptance
SR = 16000


# 1 -------------------------------------------------------------------------- #
@pytest.mark.parametrize("snr", [10.0, 0.0, -5.0, -10.0])
@pytest.mark.parametrize("noise", ["white", "pink", "babble"])
def test_mixer_snr_accuracy(snr, noise):
    clean = speechlike(dur=2.0, amp=0.1)
    nz = {"white": white_noise, "pink": pink_noise, "babble": babble_like}[noise](dur=2.0, amp=0.1)
    res = mix(clean, nz, snr, rms(clean), seed=1337)
    assert abs(res.achieved_snr_db - snr) <= 0.5


# 2 -------------------------------------------------------------------------- #
def test_mixer_determinism():
    clean, nz = speechlike(dur=1.0, amp=0.1), white_noise(dur=3.0, amp=0.1)
    a = mix(clean, nz, 0.0, rms(clean), seed=1337)
    b = mix(clean, nz, 0.0, rms(clean), seed=1337)
    c = mix(clean, nz, 0.0, rms(clean), seed=99)
    assert np.array_equal(a.mixed, b.mixed)
    assert a.noise_offset != c.noise_offset


# 3 -------------------------------------------------------------------------- #
def test_mixer_clipping_guard():
    clean, nz = speechlike(dur=1.0, amp=0.8), white_noise(dur=1.0, amp=0.8)
    res = mix(clean, nz, -10.0, rms(clean), seed=1337)
    assert res.peak_scaled and np.max(np.abs(res.mixed)) <= 0.99 + 1e-6
    assert abs(res.achieved_snr_db - (-10.0)) <= 0.5


# 6 -------------------------------------------------------------------------- #
def test_ndi_computation():
    rows = [
        {
            "lang": "es",
            "noise_subtype": None,
            "noise_level": None,
            "wer": 0.10,
            "ker": 0.20,
            "hallucination_rate": 0.0,
        },
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
    ndi = compute_ndi(pd.DataFrame(rows))
    assert [e["subtype"] for e in ndi["ranking"]] == ["AC", "AB"]
    assert ndi["ranking"][0]["per_level"] == {"05": 0.25, "10": 0.905, "15": 2.06}


# 4 -------------------------------------------------------------------------- #
def test_noise_bank_manifest(repo_root):
    manifest = repo_root / "data" / "noise_bank" / "manifest.parquet"
    if not manifest.exists():
        pytest.skip("no noise bank; run `python -m ars.noise_lab.curate`")
    tax = load_taxonomy(repo_root / "configs" / "noise_taxonomy.yaml")
    df = pd.read_parquet(manifest)
    for rec in df.to_dict(orient="records"):
        row = NoiseBankRow.model_validate(rec)
        assert row.subtype in tax.subtypes
        assert row.license  # required, non-empty
        assert (manifest.parent / row.path).exists()
    # leakage: no source recording spans both splits
    spans = df.groupby("recording_id")["split"].nunique()
    assert (spans == 1).all(), f"recordings in both splits: {spans[spans > 1].index.tolist()}"
    # minimum clips per subtype
    for st, g in df.groupby("subtype"):
        assert (g["split"] == "train").sum() >= 10, f"{st}: <10 train clips"
        assert (g["split"] == "eval").sum() >= 4, f"{st}: <4 eval clips"


# 5 -------------------------------------------------------------------------- #
@pytest.mark.parametrize("lang", ["es", "en"])
def test_matrix_corpus_complete(repo_root, lang):
    d = repo_root / "data" / "datasets" / f"eval-matrix-{lang}-v1"
    manifest = d / "manifest.parquet"
    if not manifest.exists():
        pytest.skip(f"no eval-matrix-{lang}; run `python -m ars.noise_lab.build_corpus`")
    info = json.loads((d / "dataset.json").read_text())
    df = pd.read_parquet(manifest)
    clean_n = info["clean_per_lang"]
    # clean rows
    assert (df["noise_subtype"].isna()).sum() == clean_n
    # exactly clean_n rows per (subtype, level)
    for st in info["subtypes"]:
        for lv in info["levels"]:
            cell = df[(df["noise_subtype"] == st) & (df["noise_level"] == lv)]
            assert len(cell) == clean_n, f"{st}/{lv}: {len(cell)} != {clean_n}"
    # files exist; SNR achieved within tolerance for 100% of mixed rows
    mixed = df[df["noise_subtype"].notna()]
    for rec in mixed.to_dict(orient="records"):
        assert (d / rec["path"]).exists()
        if not math.isinf(rec["snr_db_achieved"]):
            assert abs(rec["snr_db_achieved"] - rec["snr_db_target"]) <= 0.5


# 7 -------------------------------------------------------------------------- #
def test_sensitivity_report_valid(repo_root):
    runs = (
        sorted((repo_root / "reports" / "sensitivity").glob("run-*"))
        if (repo_root / "reports" / "sensitivity").exists()
        else []
    )
    if not runs:
        pytest.skip("no sensitivity report; run `python -m ars.noise_lab.sensitivity`")
    latest = runs[-1]
    ndi = json.loads((latest / "ndi.json").read_text())
    assert ndi["model_version"] and ndi["ranking"]
    matrix = pd.read_parquet(latest / "matrix.parquet")
    tax = load_taxonomy(repo_root / "configs" / "noise_taxonomy.yaml")
    for lang in ("es", "en"):
        sub = matrix[(matrix["lang"] == lang) & (matrix["noise_subtype"].notna())]
        if sub.empty:
            continue
        # full canonical grid over curated subtypes
        for st in sub["noise_subtype"].unique():
            assert st in tax.subtypes
            levels = set(sub[sub["noise_subtype"] == st]["noise_level"])
            assert {"05", "10", "15"} <= levels, f"{lang}/{st} missing levels"
        assert (latest / f"heatmap-{lang}.png").exists()


# 8 (monotonicity — informational, never fails) ------------------------------ #
def test_matrix_monotonicity_informational(repo_root):
    sd = repo_root / "reports" / "sensitivity"
    runs = sorted(sd.glob("run-*")) if sd.exists() else []
    if not runs:
        pytest.skip("no sensitivity report")
    assert (runs[-1] / "ANALYSIS.md").exists()  # records any monotonicity warnings
