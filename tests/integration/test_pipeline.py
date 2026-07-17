"""Pipeline orchestration with fakes — no real model (runs in CI)."""

from __future__ import annotations

import numpy as np
import pytest
from tests.fakes import FakeEngine, FakeVad

from ars.config import Settings
from ars.contracts import Correction
from ars.pipeline import Pipeline

SR = 16000


class FakeCorrector:
    def correct(self, text, lang):
        if "gafé" in text:
            return text.replace("gafé", "café"), [
                Correction(
                    rule_id="es-0001", span=(0, 4), before="gafé", after="café", confidence=0.9
                )
            ]
        return text, []


@pytest.fixture
def settings(repo_root):
    return Settings.load(repo_root / "configs" / "default.yaml")


def test_normal_path_returns_final_transcript(settings):
    sink = []
    engine = FakeEngine(text="hola me da un café", language="es")
    pipe = Pipeline(settings, FakeVad(0.9), engine, telemetry_sink=sink.append)
    result = pipe.transcribe(np.ones(SR, np.float32) * 0.1)
    assert result.text == "hola me da un café"
    assert result.raw_text == result.text  # log_only + passthrough
    assert engine.calls == 1
    for key in ("vad", "preprocess", "asr", "keydetector", "total"):
        assert hasattr(result.trace.latency_ms, key)
    assert len(sink) == 1  # telemetry emitted


def test_vad_gate_blocks_engine(settings):
    engine = FakeEngine()
    pipe = Pipeline(
        settings, FakeVad(speech_ratio=0.0, segments=[]), engine, telemetry_sink=lambda _: None
    )
    result = pipe.transcribe(np.zeros(SR, np.float32))
    assert result.text == ""
    assert "low_speech_gated" in result.trace.guard_flags
    assert engine.calls == 0  # engine spy: never invoked on near-silence


def test_log_only_keeps_raw_text(settings):
    settings.keydetector.mode = "log_only"
    engine = FakeEngine(text="me da un gafé")
    pipe = Pipeline(
        settings, FakeVad(0.9), engine, keydetector=FakeCorrector(), telemetry_sink=lambda _: None
    )
    result = pipe.transcribe(np.ones(SR, np.float32) * 0.1)
    assert result.text == "me da un gafé"  # unchanged in log_only
    assert [c.rule_id for c in result.corrections] == ["es-0001"]  # but recorded
    assert result.trace.rules_fired == ["es-0001"]


def test_replace_applies_correction(settings):
    settings.keydetector.mode = "replace"
    engine = FakeEngine(text="me da un gafé")
    pipe = Pipeline(
        settings, FakeVad(0.9), engine, keydetector=FakeCorrector(), telemetry_sink=lambda _: None
    )
    result = pipe.transcribe(np.ones(SR, np.float32) * 0.1)
    assert result.text == "me da un café"
    assert result.raw_text == "me da un gafé"
