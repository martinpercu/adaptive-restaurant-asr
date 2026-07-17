"""ASR engine (plan/02-architecture.md §4).

faster-whisper (CTranslate2) backend. Decoding defaults are fixed by the plan:
`condition_on_previous_text=False` (mandatory), beam_size, temperature fallback,
no_speech/log_prob thresholds. Detected language is clamped to {es, en} (01 §2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ars.config import AsrCfg
from ars.contracts import Lang, RawTranscript, Segment

SR = 16000
_LANGS = ("es", "en")


@runtime_checkable
class AsrEngine(Protocol):
    """The seam the pipeline depends on. `WhisperEngine` and test fakes implement it."""

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = SR,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> RawTranscript: ...


def _clamp_language(detected: str | None, all_probs: list | None) -> Lang:
    """Pick the higher-probability of {es, en}; never a third language (01 §2)."""
    if all_probs:
        probs = {lang: p for lang, p in all_probs}
        return "es" if probs.get("es", 0.0) >= probs.get("en", 0.0) else "en"
    if detected in _LANGS:
        return detected  # type: ignore[return-value]
    return "es"


class WhisperEngine:
    """faster-whisper engine built from a registry entry (or an explicit model size/path)."""

    def __init__(self, cfg: AsrCfg | None = None, model: str | None = None) -> None:
        self.cfg = cfg or AsrCfg()
        self.model_id = model or self.cfg.model_size
        self._model = None  # lazy — avoids loading a model in unit tests

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # noqa: PLC0415 (heavy, lazy)

            self._model = WhisperModel(
                self.model_id, device=self.cfg.device, compute_type=self.cfg.compute_type
            )
        return self._model

    def _decode_opts(self, initial_prompt: str | None) -> dict:
        return {
            "beam_size": self.cfg.beam_size,
            "temperature": self.cfg.temperature,
            "condition_on_previous_text": self.cfg.condition_on_previous_text,  # False (mandatory)
            "no_speech_threshold": self.cfg.no_speech_threshold,
            "log_prob_threshold": self.cfg.log_prob_threshold,
            "initial_prompt": initial_prompt,
        }

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = SR,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> RawTranscript:
        if sr != SR:
            raise ValueError(f"engine expects {SR} Hz; got {sr} (ingest must resample)")
        model = self._ensure_model()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        opts = self._decode_opts(initial_prompt)

        if language is None:
            segments, info = model.transcribe(audio, language=None, **opts)
            lang = _clamp_language(info.language, getattr(info, "all_language_probs", None))
            if info.language != lang:
                segments, info = model.transcribe(audio, language=lang, **opts)
        else:
            lang = language if language in _LANGS else "es"
            segments, info = model.transcribe(audio, language=lang, **opts)

        seg_list = [
            Segment(
                start=float(s.start),
                end=float(s.end),
                text=s.text.strip(),
                avg_logprob=float(s.avg_logprob),
                no_speech_prob=float(s.no_speech_prob),
            )
            for s in segments
        ]
        text = " ".join(s.text for s in seg_list).strip()
        avg = float(np.mean([s.avg_logprob for s in seg_list])) if seg_list else 0.0
        return RawTranscript(
            text=text, language=lang, segments=seg_list, avg_logprob=avg, guard_flags=[]
        )
