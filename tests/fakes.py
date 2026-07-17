"""Test doubles (plan/testing/test-strategy.md §4): fake engine/VAD, engine spy."""

from __future__ import annotations

import numpy as np

from ars.contracts import RawTranscript, Segment, VadResult

SR = 16000


class FakeEngine:
    """Returns a canned RawTranscript and counts invocations (the engine spy)."""

    def __init__(
        self,
        text: str = "hola me da un café",
        language: str = "es",
        segments: list[Segment] | None = None,
        avg_logprob: float = -0.2,
    ) -> None:
        self.text = text
        self.language = language
        self.segments = segments
        self.avg_logprob = avg_logprob
        self.calls = 0
        self.last_prompt: str | None = None

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = SR,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> RawTranscript:
        self.calls += 1
        self.last_prompt = initial_prompt
        segs = self.segments
        if segs is None:
            segs = [
                Segment(
                    start=0.0,
                    end=len(audio) / SR,
                    text=self.text,
                    avg_logprob=self.avg_logprob,
                    no_speech_prob=0.05,
                )
            ]
        return RawTranscript(
            text=self.text,
            language=self.language,  # type: ignore[arg-type]
            segments=segs,
            avg_logprob=self.avg_logprob,
            guard_flags=[],
        )


class FakeVad:
    """Returns a fixed VadResult regardless of input."""

    def __init__(
        self,
        speech_ratio: float = 0.8,
        segments: list[tuple[float, float]] | None = None,
        speech_rms: float = 0.1,
    ) -> None:
        self.speech_ratio = speech_ratio
        self.segments = segments if segments is not None else [(0.0, 1.0)]
        self.speech_rms = speech_rms

    def detect(self, audio: np.ndarray, sr: int = SR) -> VadResult:
        return VadResult(
            segments=self.segments, speech_ratio=self.speech_ratio, speech_rms=self.speech_rms
        )
