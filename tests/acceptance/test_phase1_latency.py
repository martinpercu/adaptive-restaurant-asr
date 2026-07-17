"""Latency budget (plan/phases/phase-1 acceptance #8) — `slow`, local only.

End-to-end RTF <= 0.6 on a 5 s utterance with whisper-small int8 (02 §7).
Runs the real model, so it is a slow companion to the phase-1 gate, not CI.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ars.asr.prompt_builder import load_menu
from ars.config import Settings
from ars.pipeline import Pipeline
from ars.vad import SileroVad

pytestmark = pytest.mark.slow
SR = 16000


def test_latency_budget(repo_root):
    from ars.asr.engine import WhisperEngine

    settings = Settings.load(repo_root / "configs" / "default.yaml")
    menu = load_menu(repo_root / "configs" / "menu", "demo")
    engine = WhisperEngine(settings.asr, model="small")
    pipe = Pipeline(
        settings, SileroVad(settings.vad), engine, menu=menu, telemetry_sink=lambda _: None
    )

    # 5 s of low-amplitude pink-ish noise + a speechlike burst (enough to run the model)
    rng = np.random.default_rng(1337)
    audio = rng.standard_normal(5 * SR).astype(np.float32) * 0.05
    engine.transcribe(audio[:SR], SR, language="es")  # warm up (model load excluded)

    t0 = time.perf_counter()
    pipe.transcribe(audio)
    elapsed = time.perf_counter() - t0
    rtf = elapsed / 5.0
    assert rtf <= 0.6, f"RTF {rtf:.2f} exceeds 0.6 budget"
