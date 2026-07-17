"""Real Silero VAD (bundled model, offline) + speech_active_rms math."""

from __future__ import annotations

import numpy as np

from ars.vad import SileroVad, speech_active_rms


def test_speech_active_rms_known_value():
    audio = np.concatenate([np.full(16000, 0.5, dtype=np.float32), np.zeros(16000, np.float32)])
    # RMS over [0,1)s (the 0.5-filled second) == 0.5
    assert abs(speech_active_rms(audio, [(0.0, 1.0)], 16000) - 0.5) < 1e-6
    # whole file RMS is lower (half is silence)
    assert speech_active_rms(audio, [(0.0, 2.0)], 16000) < 0.5


def test_speech_active_rms_empty_segments():
    assert speech_active_rms(np.ones(1000, np.float32), [], 16000) == 0.0


def test_silence_has_no_speech():
    vad = SileroVad()
    result = vad.detect(np.zeros(16000 * 2, dtype=np.float32), 16000)
    assert result.speech_ratio == 0.0
    assert result.segments == []
