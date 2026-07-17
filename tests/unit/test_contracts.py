from __future__ import annotations

import pytest
from pydantic import ValidationError

from ars.contracts import (
    DatasetManifestRow,
    FinalTranscript,
    NoiseBankRow,
    TranscribeTrace,
)


def test_manifest_row_minimal():
    row = DatasetManifestRow(
        utterance_id="cl-es-00001",
        path="openslr61/x.wav",
        lang="es",
        text="me da un café",
        duration_s=1.2,
        source="openslr61",
    )
    assert row.noise_subtype is None
    assert row.keywords == []


def test_manifest_row_bad_lang_rejected():
    with pytest.raises(ValidationError):
        DatasetManifestRow(
            utterance_id="x", path="p", lang="fr", text="t", duration_s=1.0, source="s"
        )


def test_noise_bank_min_duration_enforced():
    with pytest.raises(ValidationError):
        NoiseBankRow(
            clip_id="nz-AB-0001",
            subtype="AB",
            path="p",
            duration_s=1.0,
            source="musan",
            license="CC-BY-4.0",
            split="train",
        )


def test_final_transcript_roundtrips():
    ft = FinalTranscript(
        text="me da un café",
        raw_text="me da un gafé",
        language="es",
        trace=TranscribeTrace(trace_id="t1"),
    )
    dumped = ft.model_dump_json()
    assert FinalTranscript.model_validate_json(dumped).language == "es"
