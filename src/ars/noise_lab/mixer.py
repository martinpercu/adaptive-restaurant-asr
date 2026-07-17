"""SNR-accurate mixer (plan/phases/phase-2 §2.2, plan/01-conventions.md §3.2).

`mix` is a pure, deterministic function: identical `(clean, noise, snr_db, speech_rms,
seed)` give bit-identical output. SNR is defined against the clean utterance's
**speech-active RMS** (the caller computes it once with the production VAD and passes
it in — whole-file RMS is the #1 way this phase goes wrong).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SR = 16000
_EPS = 1e-9


@dataclass
class MixResult:
    mixed: np.ndarray
    achieved_snr_db: float
    gain: float
    peak_scaled: bool
    noise_offset: int


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _tile_with_crossfade(noise: np.ndarray, target_len: int, crossfade: int) -> np.ndarray:
    """Loop the noise up to >= target_len with an equal-power crossfade at each seam.

    Crossfading avoids the click transients a hard loop introduces (those behave like
    AA clatter and would contaminate other subtypes — phase-2 pitfall).
    """
    if len(noise) >= target_len:
        return noise
    cf = int(min(crossfade, len(noise) // 2))
    if cf <= 0:
        reps = target_len // len(noise) + 1
        return np.tile(noise, reps)[:target_len]
    fade_out = np.cos(np.linspace(0, np.pi / 2, cf)) ** 2
    fade_in = np.sin(np.linspace(0, np.pi / 2, cf)) ** 2
    out = noise.copy()
    while len(out) < target_len:
        head = out[:-cf]
        seam = out[-cf:] * fade_out + noise[:cf] * fade_in
        out = np.concatenate([head, seam, noise[cf:]])
    return out[:target_len].astype(np.float32)


def _noise_window(noise: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if len(noise) < n:
        noise = _tile_with_crossfade(noise, n, int(0.05 * SR))
    if len(noise) > n:
        offset = int(rng.integers(0, len(noise) - n + 1))
    else:
        offset = 0
    return np.ascontiguousarray(noise[offset : offset + n], dtype=np.float32), offset


def mix(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    speech_rms: float,
    seed: int = 1337,
) -> MixResult:
    """Mix `noise` into `clean` at `snr_db` (against `speech_rms`). Deterministic in `seed`."""
    clean = np.ascontiguousarray(clean, dtype=np.float32)
    n = len(clean)
    rng = np.random.default_rng(seed)
    window, offset = _noise_window(np.ascontiguousarray(noise, dtype=np.float32), n, rng)

    noise_rms = rms(window)
    if noise_rms < _EPS or speech_rms < _EPS:
        return MixResult(clean.copy(), math.inf, 0.0, False, offset)

    gain = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    mixed = clean + gain * window

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    peak_scaled = False
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)  # scale WHOLE mix -> preserves SNR
        peak_scaled = True

    # Achieved SNR from the components actually used (invariant to peak scaling).
    achieved = 20.0 * math.log10(speech_rms / (gain * noise_rms))
    return MixResult(mixed.astype(np.float32), achieved, gain, peak_scaled, offset)
