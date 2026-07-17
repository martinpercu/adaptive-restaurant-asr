"""Request pipeline (plan/00-overview inference path, plan/02-architecture.md §3).

VAD gate -> preprocess (AXIS 1) -> ASR engine -> hallucination guard -> keydetector
(AXIS 3) -> FinalTranscript + telemetry. The VAD gate is hallucination defense #1:
near-silent audio never reaches Whisper.

Components are injected so tests can supply fakes/spies (no real model in CI).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from ars.asr.engine import AsrEngine
from ars.asr.guard import apply_guard
from ars.asr.prompt_builder import build_bilingual_prompt
from ars.config import Settings
from ars.contracts import (
    Correction,
    FinalTranscript,
    PreprocessReport,
    StageTiming,
    TranscribeTrace,
    VadResult,
)
from ars.keydetector.passthrough import Corrector, PassthroughKeydetector
from ars.preprocess import PassthroughPreprocessor, Preprocessor
from ars.telemetry import telemetry_line, write_telemetry

SR = 16000


class _VadLike:  # structural: anything with detect(audio, sr) -> VadResult
    def detect(self, audio: np.ndarray, sr: int = SR) -> VadResult: ...


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        vad: _VadLike,
        engine: AsrEngine,
        preprocess: Preprocessor | None = None,
        keydetector: Corrector | None = None,
        menu: dict | None = None,
        model_version: str | None = None,
        telemetry_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.settings = settings
        self.vad = vad
        self.engine = engine
        self.preprocess = preprocess or PassthroughPreprocessor()
        self.keydetector = keydetector or PassthroughKeydetector()
        self.menu = menu or {}
        self.model_version = model_version
        self._prompt = build_bilingual_prompt(self.menu, settings.asr.initial_prompt_max_tokens)
        self._telemetry_sink = telemetry_sink or self._default_telemetry

    # --- telemetry default: JSONL to configured dir ------------------------ #
    def _default_telemetry(self, line: dict) -> None:
        write_telemetry(line, self.settings.ingest.telemetry_dir)

    def _empty(
        self, trace_id: str, vad_result: VadResult, pre: PreprocessReport | None, timings: dict
    ) -> FinalTranscript:
        trace = TranscribeTrace(
            trace_id=trace_id,
            vad=vad_result,
            preprocess=pre,
            guard_flags=["low_speech_gated"],
            rules_fired=[],
            latency_ms=StageTiming(**timings),
            model_version=self.model_version,
        )
        return FinalTranscript(text="", raw_text="", language="es", corrections=[], trace=trace)

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = SR,
        store_id: str | None = None,
        meta: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> FinalTranscript:
        if sr != SR:  # ingest guarantees 16 kHz; never resample here (phase-1 pitfall)
            raise ValueError(f"pipeline expects {SR} Hz; got {sr}")
        trace_id = trace_id or uuid.uuid4().hex
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        duration_s = len(audio) / SR
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        vad_result = self.vad.detect(audio, sr)
        timings["vad"] = (time.perf_counter() - t0) * 1000.0

        # VAD gate — hallucination defense #1
        if vad_result.speech_ratio < self.settings.vad.min_speech_ratio:
            timings["total"] = timings["vad"]
            result = self._empty(trace_id, vad_result, None, timings)
            self._emit(result, duration_s, store_id, PreprocessReport(), avg_logprob=0.0)
            return result

        t0 = time.perf_counter()
        audio2, pre_report = self.preprocess.process(audio, sr)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        raw = self.engine.transcribe(audio2, sr, language=None, initial_prompt=self._prompt)
        timings["asr"] = (time.perf_counter() - t0) * 1000.0

        guarded = apply_guard(raw, vad_result, self.settings.asr.guard)

        t0 = time.perf_counter()
        if self.settings.keydetector.mode == "replace":
            final_text, corrections = self.keydetector.correct(guarded.text, guarded.language)
        else:  # log_only: record corrections in trace but keep raw text
            _, corrections = self.keydetector.correct(guarded.text, guarded.language)
            final_text = guarded.text
        timings["keydetector"] = (time.perf_counter() - t0) * 1000.0
        timings["total"] = sum(
            timings.get(k, 0.0) for k in ("vad", "preprocess", "asr", "keydetector")
        )

        result = self._build(
            trace_id,
            guarded.language,
            guarded.text,
            final_text,
            corrections,
            vad_result,
            pre_report,
            guarded.guard_flags,
            timings,
        )
        self._emit(result, duration_s, store_id, pre_report, guarded.avg_logprob)
        return result

    def _build(
        self,
        trace_id,
        language,
        raw_text,
        final_text,
        corrections: list[Correction],
        vad_result,
        pre_report,
        guard_flags,
        timings,
    ) -> FinalTranscript:
        trace = TranscribeTrace(
            trace_id=trace_id,
            vad=vad_result,
            preprocess=pre_report,
            guard_flags=guard_flags,
            rules_fired=[c.rule_id for c in corrections],
            latency_ms=StageTiming(**timings),
            model_version=self.model_version,
        )
        return FinalTranscript(
            text=final_text,
            raw_text=raw_text,
            language=language,
            corrections=corrections,
            trace=trace,
        )

    def _emit(
        self,
        result: FinalTranscript,
        duration_s: float,
        store_id: str | None,
        pre: PreprocessReport,
        avg_logprob: float,
    ) -> None:
        line = telemetry_line(
            trace_id=result.trace.trace_id,
            store_id=store_id,
            duration_s=duration_s,
            speech_ratio=result.trace.vad.speech_ratio if result.trace.vad else 0.0,
            noise_pred=pre.noise_pred,
            noise_confidence=pre.noise_confidence,
            chain_applied=pre.chain_applied,
            language=result.language,
            avg_logprob=avg_logprob,
            guard_flags=result.trace.guard_flags,
            rules_fired=result.trace.rules_fired,
            latency_ms=result.trace.latency_ms.model_dump(),
            model_version=self.model_version,
        )
        self._telemetry_sink(line)
