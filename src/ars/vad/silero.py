"""Silero VAD wrapper (plan/phases/phase-1 §1.1).

`detect(audio, sr) -> VadResult` with speech segments, speech ratio, and RMS over
speech-active frames. `speech_active_rms` is the single implementation reused by the
phase-2 SNR mixer (SNR is defined against speech-active RMS, plan/01 §3.2).
"""

from __future__ import annotations

import numpy as np

from ars.config import VadCfg
from ars.contracts import VadResult

SR = 16000


def speech_active_rms(
    audio: np.ndarray, segments: list[tuple[float, float]], sr: int = SR
) -> float:
    """RMS over the samples inside `segments` (seconds). 0.0 if no active samples."""
    if not segments:
        return 0.0
    mask = np.zeros(len(audio), dtype=bool)
    for start, end in segments:
        i0 = max(0, int(round(start * sr)))
        i1 = min(len(audio), int(round(end * sr)))
        if i1 > i0:
            mask[i0:i1] = True
    active = audio[mask]
    if active.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))


class SileroVad:
    """Thin wrapper over silero-vad. The bundled model loads offline (no network)."""

    def __init__(self, cfg: VadCfg | None = None) -> None:
        self.cfg = cfg or VadCfg()
        self._model = None  # lazy-loaded

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad  # noqa: PLC0415 (lazy, heavy)

            self._model = load_silero_vad()
        return self._model

    def detect(self, audio: np.ndarray, sr: int = SR) -> VadResult:
        if sr != SR:
            raise ValueError(f"VAD expects {SR} Hz mono; got {sr}")
        import torch  # noqa: PLC0415
        from silero_vad import get_speech_timestamps  # noqa: PLC0415

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        tensor = torch.from_numpy(audio)
        stamps = get_speech_timestamps(
            tensor,
            self._ensure_model(),
            sampling_rate=SR,
            threshold=self.cfg.threshold,
            min_speech_duration_ms=self.cfg.min_speech_ms,
            min_silence_duration_ms=self.cfg.min_silence_ms,
            return_seconds=True,
        )
        segments = [(float(s["start"]), float(s["end"])) for s in stamps]
        total = len(audio) / SR if len(audio) else 0.0
        speech_dur = sum(e - s for s, e in segments)
        ratio = float(speech_dur / total) if total > 0 else 0.0
        return VadResult(
            segments=segments,
            speech_ratio=min(1.0, ratio),
            speech_rms=speech_active_rms(audio, segments, SR),
        )
