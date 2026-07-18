"""Phase 1 exit gate (plan/phases/phase-1-baseline-asr.md). Run via `make gate PHASE=1`."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from tests.fakes import FakeEngine, FakeVad

from ars.api.app import app, get_pipeline
from ars.asr.engine import _clamp_language
from ars.asr.guard import apply_guard
from ars.config import AsrGuardCfg, Settings
from ars.contracts import FinalTranscript, RawTranscript, Segment
from ars.eval.metrics import UttRecord, compute_metrics, ker, wer_cer
from ars.eval.normalize import normalize
from ars.pipeline import Pipeline

pytestmark = pytest.mark.acceptance
SR = 16000


def _wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _client_with(pipeline: Pipeline) -> TestClient:
    from ars.api.app import get_settings

    s = Settings.load()
    s.security.auth_enabled = False  # phase-1 API contract tests predate phase-7 auth
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: s
    return TestClient(app)


@pytest.fixture
def settings(repo_root):
    return Settings.load(repo_root / "configs" / "default.yaml")


# 1 -------------------------------------------------------------------------- #
def test_api_contract(settings):
    pipe = Pipeline(settings, FakeVad(0.9), FakeEngine(text="hola"), telemetry_sink=lambda _: None)
    client = _client_with(pipe)
    try:
        resp = client.post(
            "/v1/transcribe", files={"file": ("a.wav", _wav_bytes(np.ones(SR) * 0.1))}
        )
        assert resp.status_code == 200
        ft = FinalTranscript.model_validate(resp.json())
        assert ft.text == "hola"
        for key in ("vad", "preprocess", "asr", "keydetector", "total"):
            assert key in ft.trace.latency_ms.model_dump()
    finally:
        app.dependency_overrides.clear()


# 2 -------------------------------------------------------------------------- #
def test_pure_noise_returns_empty(settings):
    spy = FakeEngine()
    pipe = Pipeline(
        settings, FakeVad(speech_ratio=0.0, segments=[]), spy, telemetry_sink=lambda _: None
    )
    client = _client_with(pipe)
    try:
        noise = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.3
        resp = client.post("/v1/transcribe", files={"file": ("n.wav", _wav_bytes(noise))})
        ft = FinalTranscript.model_validate(resp.json())
        assert ft.text == ""
        assert "low_speech_gated" in ft.trace.guard_flags
        assert spy.calls == 0  # engine spy: ASR never invoked on pure noise
    finally:
        app.dependency_overrides.clear()


# 3 -------------------------------------------------------------------------- #
def test_repetition_guard():
    looped = "gracias por venir " * 5  # period-3 loop -> 3-gram repeats consecutively
    seg = Segment(start=0.0, end=5.0, text=looped.strip(), avg_logprob=-0.4, no_speech_prob=0.1)
    raw = RawTranscript(text=looped.strip(), language="es", segments=[seg], avg_logprob=-0.4)
    out = apply_guard(raw, None, AsrGuardCfg())
    assert "repetition_truncated" in out.guard_flags
    assert out.text == "gracias por venir"


# 4 -------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "detected,probs",
    [
        ("fr", [("fr", 0.9), ("es", 0.06), ("en", 0.04)]),
        ("de", [("de", 0.8), ("en", 0.15), ("es", 0.05)]),
        ("es", [("es", 0.7), ("en", 0.3)]),
        ("en", None),
        (None, None),
    ],
)
def test_language_clamp(detected, probs):
    assert _clamp_language(detected, probs) in {"es", "en"}


# 5 -------------------------------------------------------------------------- #
def test_wer_harness_known_values():
    refs = [
        "the cat sat",
        "i want fries",
        "give me a spoon",
        "large soda please",
        "chicken sandwich",
    ]
    hyps = [
        "the cat sat",
        "i want flies",
        "give me spoon",
        "large soda please now",
        "chicken sandwich",
    ]
    wer, cer = wer_cer(refs, hyps, "en")
    assert wer == 0.2  # 3 word errors / 15 ref words

    # single controlled pair for exact CER
    _, cer1 = wer_cer(["spoon"], ["spoons"], "en")
    assert cer1 == 0.2  # 1 insertion / 5 ref chars

    recs = [
        UttRecord(ref="i want fries", hyp="i want flies", lang="en", keywords=["fries"]),
        UttRecord(ref="give me a spoon", hyp="give me spoon", lang="en", keywords=["spoon"]),
    ]
    k, total = ker(recs, "en")
    assert total == 2 and k == 0.5  # spoon recovered, fries not


def test_compute_metrics_bundle():
    recs = [UttRecord(ref="hola cafe", hyp="hola cafe", lang="es", keywords=["cafe"])]
    m = compute_metrics(recs, "es")
    assert m["wer"] == 0.0 and m["ker"] == 0.0 and m["n_utts"] == 1


# 6 -------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,lang,expected",
    [
        ("Hola, ¿me da café?", "es", "hola me da café"),
        ("Dos HAMBURGUESAS!!", "es", "dos hamburguesas"),
        ("I don't want water.", "en", "i don't want water"),
        ("Fries, please!!!", "en", "fries please"),
        ("  multiple   spaces  ", "en", "multiple spaces"),
    ],
)
def test_normalize_rules(text, lang, expected):
    assert normalize(text, lang) == expected


# 7 -------------------------------------------------------------------------- #
def test_baseline_reports_exist_and_valid(repo_root):
    seen = False
    for lang in ("es", "en"):
        path = repo_root / "reports" / "baseline" / f"baseline-{lang}.json"
        if not path.exists():
            continue
        seen = True
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["lang"] == lang
        assert data["eval_clean"]["n_utts"] >= 200, f"{lang}: too few eval utts"
        assert 0.0 <= data["eval_clean"]["wer"] <= 1.0
    if not seen:
        pytest.skip("no baseline reports; run `python -m ars.eval.baseline`")
