from __future__ import annotations

import math

import numpy as np
import pytest
from tests.conftest import babble_like, pink_noise, speechlike, white_noise

from ars.noise_lab.mixer import mix, rms

SR = 16000


def _recovered_snr(clean, mixed, speech_rms):
    added = mixed - clean  # exact when no peak scaling
    return 20.0 * math.log10(speech_rms / rms(added))


@pytest.mark.parametrize("snr", [10.0, 0.0, -5.0, -10.0])
@pytest.mark.parametrize("noise_name", ["white", "pink", "babble"])
def test_mixer_snr_accuracy(snr, noise_name):
    clean = speechlike(dur=2.0, amp=0.1)
    noise = {
        "white": white_noise(dur=2.0, amp=0.1),
        "pink": pink_noise(dur=2.0, amp=0.1),
        "babble": babble_like(dur=2.0, amp=0.1),
    }[noise_name]
    speech_rms = rms(clean)
    res = mix(clean, noise, snr, speech_rms, seed=1337)
    assert not res.peak_scaled  # low amplitudes -> no clipping for this check
    assert abs(res.achieved_snr_db - snr) <= 0.5
    assert abs(_recovered_snr(clean, res.mixed, speech_rms) - snr) <= 0.5


def test_mixer_determinism_same_seed():
    clean = speechlike(dur=1.0, amp=0.1)
    noise = white_noise(dur=3.0, amp=0.1)  # longer -> offset can vary
    a = mix(clean, noise, 0.0, rms(clean), seed=1337)
    b = mix(clean, noise, 0.0, rms(clean), seed=1337)
    assert np.array_equal(a.mixed, b.mixed)
    assert a.noise_offset == b.noise_offset


def test_mixer_different_seed_different_window():
    clean = speechlike(dur=1.0, amp=0.1)
    noise = white_noise(dur=3.0, amp=0.1)
    a = mix(clean, noise, 0.0, rms(clean), seed=1)
    b = mix(clean, noise, 0.0, rms(clean), seed=2)
    assert a.noise_offset != b.noise_offset
    assert not np.array_equal(a.mixed, b.mixed)


def test_mixer_clipping_guard():
    # loud speech + very low SNR -> would exceed 1.0; guard must scale to <= 0.99.
    clean = speechlike(dur=1.0, amp=0.8)
    noise = white_noise(dur=1.0, amp=0.8)
    res = mix(clean, noise, -10.0, rms(clean), seed=1337)
    assert res.peak_scaled
    assert np.max(np.abs(res.mixed)) <= 0.99 + 1e-6
    # SNR is preserved through whole-mix scaling.
    assert abs(res.achieved_snr_db - (-10.0)) <= 0.5


def test_mixer_short_noise_loop_padded():
    clean = speechlike(dur=3.0, amp=0.1)
    noise = white_noise(dur=1.0, amp=0.1)  # shorter than clean -> tiled
    res = mix(clean, noise, 0.0, rms(clean), seed=1337)
    assert len(res.mixed) == len(clean)
    assert abs(res.achieved_snr_db) <= 0.5


def test_silent_noise_returns_clean():
    clean = speechlike(dur=1.0, amp=0.1)
    silent = np.zeros(SR, dtype=np.float32)
    res = mix(clean, silent, 0.0, rms(clean), seed=1337)
    assert np.array_equal(res.mixed, clean)
    assert math.isinf(res.achieved_snr_db)
