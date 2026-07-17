"""Shared audio I/O for scripts: convert any source clip to canonical 16 kHz mono WAV.

Canonical format (plan/01-conventions.md §1): WAV, PCM 16-bit written from float32
[-1,1], 16 kHz, mono. Reads via soundfile (handles FLAC/WAV/OGG); resample via
torchaudio. Kept out of `src/ars` because it is a build-time utility, not runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_mono_float32(src: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)  # downmix channels -> mono
    return mono.astype(np.float32), sr


def resample(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio
    import torch
    import torchaudio.functional as AF  # noqa: PLC0415 (heavy import, lazy)

    tensor = torch.from_numpy(audio).unsqueeze(0)
    out = AF.resample(tensor, sr, target_sr).squeeze(0).numpy()
    return out.astype(np.float32)


def to_wav_16k_mono(src: str | Path, dst: str | Path) -> float:
    """Convert `src` to canonical WAV at `dst`. Returns duration in seconds."""
    audio, sr = load_mono_float32(src)
    audio = resample(audio, sr, TARGET_SR)
    audio = np.clip(audio, -1.0, 1.0)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
    return len(audio) / TARGET_SR


def write_wav(dst: str | Path, audio: np.ndarray, sr: int = TARGET_SR) -> float:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), np.clip(audio, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")
    return len(audio) / sr
