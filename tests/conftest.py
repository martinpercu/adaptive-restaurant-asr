"""Shared test fixtures (plan/testing/test-strategy.md §2).

All audio fixtures are GENERATED here, seeded (seed=1337 default) — no committed
binaries. Factories return float32 mono arrays in [-1, 1] at 16 kHz.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 16000
SEED = 1337


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(SEED if seed is None else seed)


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR), dtype=np.float32) / SR


def tone(freq: float = 440.0, dur: float = 1.0, amp: float = 0.5) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * freq * _t(dur))).astype(np.float32)


def chirp(f0: float = 200.0, f1: float = 4000.0, dur: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = _t(dur)
    k = (f1 - f0) / max(dur, 1e-9)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    return (amp * np.sin(phase)).astype(np.float32)


def white_noise(dur: float = 1.0, amp: float = 0.3, seed: int | None = None) -> np.ndarray:
    x = _rng(seed).standard_normal(int(dur * SR)).astype(np.float32)
    return (amp * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def pink_noise(dur: float = 1.0, amp: float = 0.3, seed: int | None = None) -> np.ndarray:
    n = int(dur * SR)
    white = _rng(seed).standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs)
    x = np.fft.irfft(spectrum, n).astype(np.float32)
    return (amp * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def babble_like(dur: float = 1.0, amp: float = 0.3, seed: int | None = None) -> np.ndarray:
    """Sum of AM-modulated band-limited noise — a crude babble surrogate for VAD/mixer tests."""
    rng = _rng(seed)
    t = _t(dur)
    out = np.zeros_like(t)
    for _ in range(6):
        base = rng.standard_normal(len(t))
        center = rng.uniform(300, 3000)
        carrier = np.sin(2 * np.pi * center * t)
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 8) * t)
        out += base * carrier * mod
    out = out.astype(np.float32)
    return (amp * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def silence(dur: float = 1.0) -> np.ndarray:
    return np.zeros(int(dur * SR), dtype=np.float32)


def speechlike(dur: float = 1.0, amp: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Formant-ish AM tone bursts — enough for VAD/mixer tests, NOT for WER tests."""
    rng = _rng(seed)
    t = _t(dur)
    formants = [500, 1500, 2500]
    sig = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(formants))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 6) * t)  # syllable-rate envelope
    sig = (sig * env).astype(np.float32)
    return (amp * sig / (np.max(np.abs(sig)) + 1e-9)).astype(np.float32)


@pytest.fixture
def audio_factories() -> dict[str, Callable[..., np.ndarray]]:
    return {
        "tone": tone,
        "chirp": chirp,
        "white_noise": white_noise,
        "pink_noise": pink_noise,
        "babble_like": babble_like,
        "silence": silence,
        "speechlike": speechlike,
    }


@pytest.fixture
def write_wav(tmp_path: Path) -> Callable[[np.ndarray, str], Path]:
    def _write(audio: np.ndarray, name: str = "clip.wav", sr: int = SR) -> Path:
        path = tmp_path / name
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return path

    return _write


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
