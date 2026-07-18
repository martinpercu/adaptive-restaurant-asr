"""Preprocess — AXIS 1: noise classifier + targeted mitigation chains (phase 3).

`PassthroughPreprocessor` is the phase-1 no-op; `PolicyPreprocessor` (phase 3) does the
real classify → policy → chain. `build_preprocessor` picks the right one from settings,
falling back to pass-through when the classifier/policy have not been generated yet.
"""

from __future__ import annotations

import time
from pathlib import Path
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


def build_preprocessor(settings) -> Preprocessor:
    """AXIS 1 for the production pipeline. Falls back to pass-through if unavailable."""
    cfg = settings.preprocess
    if not cfg.enabled or cfg.mode == "off":
        return PassthroughPreprocessor()
    classifier_path = Path(cfg.classifier_path)
    policy_path = Path(cfg.policy_path)
    if not classifier_path.exists() or not policy_path.exists():
        return PassthroughPreprocessor()  # not generated yet (pre phase-3 heavy path)
    from ars.preprocess.classifier import Classifier  # noqa: PLC0415
    from ars.preprocess.policy import PolicyPreprocessor, load_policy  # noqa: PLC0415

    classifier = Classifier.load(str(classifier_path), min_confidence=cfg.min_confidence)
    return PolicyPreprocessor(classifier, load_policy(policy_path), mode=cfg.mode)
