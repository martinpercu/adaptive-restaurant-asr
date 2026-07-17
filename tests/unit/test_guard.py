from __future__ import annotations

from ars.asr.guard import _truncate_repetition, apply_guard
from ars.config import AsrGuardCfg
from ars.contracts import RawTranscript, Segment, VadResult


def _raw(segments, text=None):
    text = text if text is not None else " ".join(s.text for s in segments)
    return RawTranscript(text=text, language="es", segments=segments, avg_logprob=-0.3)


def test_repetition_truncated_and_flagged():
    text = "gracias por venir " * 4  # period-3 loop -> 3-gram repeats consecutively
    out, fired = _truncate_repetition(text.strip(), ngram=3, max_repeats=3)
    assert fired
    assert out == "gracias por venir"


def test_no_repetition_untouched():
    out, fired = _truncate_repetition("me da un cafe con leche", 3, 3)
    assert not fired
    assert out == "me da un cafe con leche"


def test_apply_guard_repetition():
    seg = Segment(
        start=0.0,
        end=4.0,
        text="gracias por venir gracias por venir gracias por venir gracias por venir",
        avg_logprob=-0.4,
        no_speech_prob=0.1,
    )
    out = apply_guard(_raw([seg]), None, AsrGuardCfg())
    assert "repetition_truncated" in out.guard_flags
    assert out.text == "gracias por venir"


def test_no_speech_segment_dropped():
    good = Segment(start=0.0, end=1.0, text="hola", avg_logprob=-0.3, no_speech_prob=0.1)
    bad = Segment(start=1.0, end=2.0, text="ruido", avg_logprob=-2.0, no_speech_prob=0.95)
    out = apply_guard(_raw([good, bad]), None, AsrGuardCfg())
    assert out.text == "hola"
    assert "low_speech_dropped" in out.guard_flags


def test_vad_span_drop():
    inside = Segment(start=0.0, end=1.0, text="hola", avg_logprob=-0.3, no_speech_prob=0.1)
    outside = Segment(start=5.0, end=6.0, text="inventado", avg_logprob=-0.3, no_speech_prob=0.1)
    vad = VadResult(segments=[(0.0, 2.0)], speech_ratio=0.5, speech_rms=0.1)
    out = apply_guard(_raw([inside, outside]), vad, AsrGuardCfg())
    assert out.text == "hola"
    assert "vad_dropped" in out.guard_flags
