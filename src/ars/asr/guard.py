"""Hallucination guard (plan/02-architecture.md §4).

Post-decode defenses (Whisper hallucination defenses are sacred, CLAUDE.md):
  (a) drop segments with no_speech_prob > max AND avg_logprob < min;
  (b) if a 3-gram repeats >= N times consecutively, truncate at the first repetition;
  (c) drop segments whose time span the VAD marked as non-speech.
Each defense that fires appends its flag.
"""

from __future__ import annotations

from ars.config import AsrGuardCfg
from ars.contracts import RawTranscript, Segment, VadResult


def _overlaps_speech(seg: Segment, vad_segments: list[tuple[float, float]]) -> bool:
    return any(seg.start < end and seg.end > start for start, end in vad_segments)


def _truncate_repetition(text: str, ngram: int, max_repeats: int) -> tuple[str, bool]:
    """Truncate at the first n-gram that repeats >= max_repeats times consecutively."""
    toks = text.split()
    n = ngram
    for i in range(len(toks)):
        block = toks[i : i + n]
        if len(block) < n:
            break
        reps = 1
        while toks[i + reps * n : i + (reps + 1) * n] == block:
            reps += 1
        if reps >= max_repeats:
            return " ".join(toks[: i + n]), True
    return text, False


def apply_guard(
    raw: RawTranscript, vad: VadResult | None, cfg: AsrGuardCfg | None = None
) -> RawTranscript:
    cfg = cfg or AsrGuardCfg()
    flags: list[str] = list(raw.guard_flags)

    kept: list[Segment] = []
    for s in raw.segments:
        if s.no_speech_prob > cfg.no_speech_prob_max and s.avg_logprob < cfg.avg_logprob_min:
            flags.append("low_speech_dropped")
            continue
        kept.append(s)

    if vad is not None and vad.segments:
        surviving: list[Segment] = []
        for s in kept:
            if _overlaps_speech(s, vad.segments):
                surviving.append(s)
            else:
                flags.append("vad_dropped")
        kept = surviving

    text = " ".join(s.text.strip() for s in kept).strip()
    text, truncated = _truncate_repetition(text, cfg.repetition_ngram, cfg.repetition_max_repeats)
    if truncated:
        flags.append("repetition_truncated")

    # de-dup flags, preserve order
    seen: set[str] = set()
    ordered = [f for f in flags if not (f in seen or seen.add(f))]
    return RawTranscript(
        text=text,
        language=raw.language,
        segments=kept,
        avg_logprob=raw.avg_logprob,
        guard_flags=ordered,
    )
