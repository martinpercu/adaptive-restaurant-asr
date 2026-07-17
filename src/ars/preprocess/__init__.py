"""Preprocess — AXIS 1: noise classifier + targeted mitigation chains (phase 3).

Phase 1 ships a pass-through that honors the `PreprocessReport` contract so the
pipeline and telemetry are wired end-to-end before the real classifier lands.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import numpy as np

from ars.contracts import PreprocessReport

SR = 16000


@runtime_checkable
class Preprocessor(Protocol):
    def process(self, audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, PreprocessReport]: ...


class PassthroughPreprocessor:
    """No-op: returns audio unchanged, reports clean (noise_pred=None, empty chain)."""

    def process(self, audio: np.ndarray, sr: int = SR) -> tuple[np.ndarray, PreprocessReport]:
        t0 = time.perf_counter()
        report = PreprocessReport(
            noise_pred=None,
            noise_confidence=0.0,
            chain_applied=[],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return audio, report
