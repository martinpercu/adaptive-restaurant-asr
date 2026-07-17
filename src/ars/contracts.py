"""Typed cross-module contracts (plan/02-architecture.md §2, plan/03-data-spec.md).

Every piece of data that crosses a module boundary is one of these pydantic models.
No ad-hoc dicts between modules (CLAUDE.md hard rule 6).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Lang = Literal["es", "en"]


# --------------------------------------------------------------------------- #
# Online request-path contracts (02 §2)
# --------------------------------------------------------------------------- #
class AudioRecord(BaseModel):
    """An ingested utterance. Audio is always 16 kHz mono PCM after ingest."""

    utterance_id: str
    path: str  # storage-relative
    sample_rate: int = 16000
    duration_s: float
    store_id: str | None = None
    captured_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)  # mic gain, POS order id, daypart...


class VadResult(BaseModel):
    segments: list[tuple[float, float]] = Field(default_factory=list)  # (start_s, end_s)
    speech_ratio: float  # 0..1
    speech_rms: float  # RMS over VAD-active frames of the clean utterance


class PreprocessReport(BaseModel):
    noise_pred: str | None = None  # subtype code e.g. "AB"; None = clean
    noise_confidence: float = 0.0
    chain_applied: list[str] = Field(default_factory=list)  # e.g. ["deepfilternet"]
    latency_ms: float = 0.0


class Segment(BaseModel):
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


class RawTranscript(BaseModel):
    text: str
    language: Lang
    segments: list[Segment] = Field(default_factory=list)
    avg_logprob: float
    guard_flags: list[str] = Field(default_factory=list)  # e.g. ["repetition_truncated"]


class Correction(BaseModel):
    rule_id: str  # confusion rule id or "lexicon"
    span: tuple[int, int]  # char offsets in raw text
    before: str
    after: str
    confidence: float


class StageTiming(BaseModel):
    vad: float = 0.0
    preprocess: float = 0.0
    asr: float = 0.0
    keydetector: float = 0.0
    total: float = 0.0


class TranscribeTrace(BaseModel):
    trace_id: str
    vad: VadResult | None = None
    preprocess: PreprocessReport | None = None
    guard_flags: list[str] = Field(default_factory=list)
    rules_fired: list[str] = Field(default_factory=list)
    latency_ms: StageTiming = Field(default_factory=StageTiming)
    model_version: str | None = None


class FinalTranscript(BaseModel):
    text: str
    raw_text: str
    language: Lang
    corrections: list[Correction] = Field(default_factory=list)
    trace: TranscribeTrace


# --------------------------------------------------------------------------- #
# Data-spec contracts (03)
# --------------------------------------------------------------------------- #
class DatasetManifestRow(BaseModel):
    """One row of data/datasets/<id>/manifest.parquet (03 §1)."""

    utterance_id: str
    path: str  # relative to dataset dir
    lang: Lang
    text: str
    duration_s: float
    source: str
    accent: str | None = None
    clean_id: str | None = None
    noise_subtype: str | None = None
    noise_level: str | None = None
    noise_clip_id: str | None = None
    snr_db_target: float | None = None
    snr_db_achieved: float | None = None
    mix_seed: int | None = None
    keywords: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    """data/datasets/<id>/dataset.json (03 §1)."""

    dataset_id: str
    created_at: datetime
    generator: str
    generator_version: str
    config_hash: str
    seed: int
    row_count: int
    langs: list[Lang]


class NoiseBankRow(BaseModel):
    """One row of data/noise_bank/manifest.parquet (03 §2)."""

    clip_id: str  # nz-<SUBTYPE>-<seq>
    subtype: str
    path: str
    duration_s: float = Field(ge=3.0)
    source: str
    license: str  # required; block ingestion if unknown
    split: Literal["train", "eval"]


class JudgeVerdict(BaseModel):
    """LLM-judge structured output (03 §8)."""

    verdict: Literal["correct", "minor_errors", "wrong", "hallucination"]
    corrected_reference: str | None = None
    confusion_candidates: list[dict[str, str]] = Field(default_factory=list)
    order_core_match: bool
    confidence: float
