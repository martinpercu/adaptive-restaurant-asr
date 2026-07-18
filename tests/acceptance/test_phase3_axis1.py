"""Phase 3 exit gate (plan/phases/phase-3-axis1-preprocessing.md). `make gate PHASE=3`."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from tests.conftest import silence, speechlike, white_noise

from ars.preprocess.classifier import CLEAN, decide
from ars.preprocess.denoisers import available_chains, get_denoiser
from ars.preprocess.policy import PolicyPreprocessor, generate_policy

pytestmark = pytest.mark.acceptance
SR = 16000


# 3 -------------------------------------------------------------------------- #
def test_classifier_confidence_gate():
    classes = ["AB", "BB", CLEAN]
    # confident noisy prediction -> subtype
    hi = decide(np.array([5.0, 0.0, 0.0]), classes, min_confidence=0.6)
    assert hi.subtype == "AB" and hi.confidence > 0.6
    # flat logits -> low confidence -> None
    lo = decide(np.array([0.1, 0.0, 0.0]), classes, min_confidence=0.6)
    assert lo.subtype is None
    # CLEAN class wins -> None regardless of confidence
    cl = decide(np.array([0.0, 0.0, 9.0]), classes, min_confidence=0.6)
    assert cl.subtype is None


# 4 -------------------------------------------------------------------------- #
@pytest.mark.parametrize("chain", available_chains())
def test_denoiser_interface(chain):
    d = get_denoiser(chain)
    x = speechlike(dur=1.0, amp=0.2) + white_noise(dur=1.0, amp=0.1)
    out = d.process(x, SR)
    assert out.shape == x.shape and out.dtype == np.float32
    # silence in -> ~silence out
    s_out = d.process(silence(0.5), SR)
    assert np.max(np.abs(s_out)) < 1e-3
    # deterministic
    assert np.array_equal(d.process(x, SR), d.process(x, SR))


# 5 -------------------------------------------------------------------------- #
def _eff(rows):
    return pd.DataFrame(rows)


def test_policy_helps_both_langs_mapped():
    rows = []
    for lang in ("es", "en"):
        rows += [
            {
                "lang": lang,
                "subtype": "AB",
                "level": "10",
                "chain": "none",
                "wer": 0.50,
                "ker": 0.6,
                "latency_ms": 0,
            },
            {
                "lang": lang,
                "subtype": "AB",
                "level": "10",
                "chain": "spectral_gate",
                "wer": 0.40,
                "ker": 0.5,
                "latency_ms": 100,
            },
        ]
    assert generate_policy(_eff(rows))["AB"] == "spectral_gate"  # 20% rel, both langs


def test_policy_helps_one_lang_only_none():
    rows = [
        {
            "lang": "es",
            "subtype": "AB",
            "level": "10",
            "chain": "none",
            "wer": 0.50,
            "ker": 0.6,
            "latency_ms": 0,
        },
        {
            "lang": "es",
            "subtype": "AB",
            "level": "10",
            "chain": "spectral_gate",
            "wer": 0.40,
            "ker": 0.5,
            "latency_ms": 100,
        },
        {
            "lang": "en",
            "subtype": "AB",
            "level": "10",
            "chain": "none",
            "wer": 0.50,
            "ker": 0.6,
            "latency_ms": 0,
        },
        {
            "lang": "en",
            "subtype": "AB",
            "level": "10",
            "chain": "spectral_gate",
            "wer": 0.50,
            "ker": 0.6,
            "latency_ms": 100,
        },  # no help
    ]
    assert generate_policy(_eff(rows))["AB"] == "none"


def test_policy_over_latency_none():
    rows = []
    for lang in ("es", "en"):
        rows += [
            {
                "lang": lang,
                "subtype": "BB",
                "level": "10",
                "chain": "none",
                "wer": 0.5,
                "ker": 0.6,
                "latency_ms": 0,
            },
            {
                "lang": lang,
                "subtype": "BB",
                "level": "10",
                "chain": "deepfilternet",
                "wer": 0.3,
                "ker": 0.4,
                "latency_ms": 900,
            },
        ]
    assert generate_policy(_eff(rows))["BB"] == "none"  # helps but blows 400 ms budget


def test_policy_ker_worsens_none():
    rows = []
    for lang in ("es", "en"):
        rows += [
            {
                "lang": lang,
                "subtype": "CA",
                "level": "10",
                "chain": "none",
                "wer": 0.5,
                "ker": 0.5,
                "latency_ms": 0,
            },
            {
                "lang": lang,
                "subtype": "CA",
                "level": "10",
                "chain": "spectral_gate",
                "wer": 0.4,
                "ker": 0.7,
                "latency_ms": 100,
            },
        ]
    assert generate_policy(_eff(rows))["CA"] == "none"  # WER better but KER worse


# 7 -------------------------------------------------------------------------- #
class _FakeClassifier:
    def __init__(self, subtype):
        self._subtype = subtype

    def predict(self, audio, sr=SR):
        from ars.preprocess.classifier import NoisePrediction

        return NoisePrediction(self._subtype, 0.9)


class _DoubleDenoiser:
    chain_id = "double"

    def process(self, audio, sr=SR):
        return (audio * 2).astype(np.float32)


def test_pipeline_modes():
    audio = speechlike(dur=0.5, amp=0.2)
    policy = {"AB": "double"}

    off = PolicyPreprocessor(_FakeClassifier("AB"), policy, mode="off")
    out, rep = off.process(audio, SR)
    assert np.array_equal(out, audio) and rep.noise_pred is None

    log = PolicyPreprocessor(_FakeClassifier("AB"), policy, mode="log_only")
    log._cache["double"] = _DoubleDenoiser()
    out, rep = log.process(audio, SR)
    assert np.array_equal(out, audio)  # log_only: audio bit-identical
    assert rep.noise_pred == "AB" and rep.chain_applied == []

    act = PolicyPreprocessor(_FakeClassifier("AB"), policy, mode="active")
    act._cache["double"] = _DoubleDenoiser()
    out, rep = act.process(audio, SR)
    assert np.array_equal(out, audio * 2) and rep.chain_applied == ["double"]


# 1 -------------------------------------------------------------------------- #
def test_classifier_eval_report(repo_root):
    report = repo_root / "models" / "noise_classifier" / "latest" / "eval.json"
    if not report.exists():
        pytest.skip("no classifier eval; run `python -m ars.preprocess.train_classifier`")
    r = json.loads(report.read_text())
    assert r["subtype_macro_f1"] >= 0.80, f"subtype F1 {r['subtype_macro_f1']} < 0.80"
    assert r["family_macro_f1"] >= 0.90, f"family F1 {r['family_macro_f1']} < 0.90"
    assert r["clean_precision"] >= 0.95, f"clean precision {r['clean_precision']} < 0.95"


# 2 -------------------------------------------------------------------------- #
def test_classifier_no_window_leakage(repo_root):
    bank = repo_root / "data" / "noise_bank" / "manifest.parquet"
    if not bank.exists():
        pytest.skip("no noise bank")
    df = pd.read_parquet(bank)
    train_recs = set(df[df["split"] == "train"]["recording_id"])
    eval_recs = set(df[df["split"] == "eval"]["recording_id"])
    assert not (train_recs & eval_recs), "source recordings shared across classifier splits"


# 6 -------------------------------------------------------------------------- #
def test_policy_file_is_generated(repo_root):
    path = repo_root / "configs" / "mitigation_policy.yaml"
    if not path.exists():
        pytest.skip("no policy; run `python -m ars.preprocess.gen_policy`")
    doc = yaml.safe_load(path.read_text())
    assert doc["_meta"]["run_id"]
    bank = pd.read_parquet(repo_root / "data" / "noise_bank" / "manifest.parquet")
    for st in bank["subtype"].unique():
        assert st in doc["policy"], f"subtype {st} not mapped in policy"


# 8 (slow, local) ------------------------------------------------------------ #
def _top3(repo_root) -> list[str]:
    runs = sorted((repo_root / "reports" / "sensitivity").glob("run-*"))
    if runs:
        ndi = json.loads((runs[-1] / "ndi.json").read_text())
        es = [e["subtype"] for e in ndi["ranking"] if e["lang"] == "es"][:3]
        if es:
            return es
    return ["BB", "CA", "BC"]


@pytest.mark.slow
def test_effectiveness_improvement(repo_root):
    reports = (
        sorted((repo_root / "reports" / "mitigation").glob("run-*"))
        if (repo_root / "reports" / "mitigation").exists()
        else []
    )
    policy_path = repo_root / "configs" / "mitigation_policy.yaml"
    if not reports or not policy_path.exists():
        pytest.skip("no effectiveness report/policy; run the phase-3 heavy path")

    df = pd.read_parquet(reports[-1] / "effectiveness.parquet")
    policy = yaml.safe_load(policy_path.read_text())["policy"]
    top3 = _top3(repo_root)

    for lang in ("es", "en"):
        none_wer, active_wer = [], []
        for st in top3:
            chain = policy.get(st, "none")
            cells = df[(df["lang"] == lang) & (df["subtype"] == st)]
            if cells.empty:
                continue
            none_wer.append(cells[cells["chain"] == "none"]["wer"].mean())
            active_wer.append(cells[cells["chain"] == chain]["wer"].mean())
        nw, aw = float(np.nanmean(none_wer)), float(np.nanmean(active_wer))
        rel = (nw - aw) / max(nw, 1e-6)
        assert rel >= 0.05, (
            f"{lang}: top-3 WER improvement {rel:.1%} < 5% (none={nw:.3f} active={aw:.3f})"
        )

    # clean regression <= 1% relative
    clean = df[df["subtype"] == "__clean__"]
    for lang in ("es", "en"):
        nw = clean[(clean["lang"] == lang) & (clean["chain"] == "none")]["wer"].mean()
        for chain in set(policy.values()) - {"none"}:
            cw = clean[(clean["lang"] == lang) & (clean["chain"] == chain)]["wer"].mean()
            if not np.isnan(cw):
                assert (cw - nw) / max(nw, 1e-6) <= 0.01, f"{lang}/{chain} harms clean > 1%"
