"""Anti-catastrophic-forgetting regression suite (plan/phases/phase-4 §4.4).

Fixed corpus regression-keywords-v1: keyword-bearing utterances (menu items + idea.txt
confusion targets, from the TTS domain corpus) + generic non-domain sentences (clean read
speech). Metric: keyword recall + generic clean WER. Gate lives in `training.gate`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from ars.asr.guard import apply_guard
from ars.config import DEFAULT_CONFIG, Settings
from ars.eval.metrics import UttRecord, keyword_recovered, wer_cer
from ars.eval.normalize import tokens
from ars.keydetector.phonetics import phonetic_key

SR = 16000


def keyword_recall(records: list[UttRecord], lang: str) -> tuple[float, int]:
    """recovered / total over reference keywords (= 1 - KER)."""
    total = recovered = 0
    for r in records:
        if not r.keywords:
            continue
        hyp_tokens = tokens(r.hyp, lang)
        hyp_keys = [phonetic_key(t, lang) for t in hyp_tokens]
        for kw in r.keywords:
            total += 1
            if keyword_recovered(kw, hyp_tokens, hyp_keys, lang):
                recovered += 1
    return (recovered / total if total else 1.0), total


def build_corpus(settings: Settings, generic_per_lang: int = 50) -> Path:
    """regression-keywords-v1 from existing audio (keyword part: TTS; generic: eval-clean)."""
    import os  # noqa: PLC0415

    data = Path(settings.paths.data)
    out = data / "datasets" / "regression-keywords-v1"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for lang in ("es", "en"):
        tts = pd.read_parquet(data / "datasets" / f"tts-{lang}-v1" / "manifest.parquet")
        kw = tts[tts["keywords"].map(lambda k: k is not None and len(k) > 0)].head(120)
        for r in kw.to_dict("records"):
            uid = f"rk-{lang}-{r['utterance_id']}"
            dst = out / "audio" / f"{uid}.wav"
            if not dst.exists():
                os.link(data / "datasets" / f"tts-{lang}-v1" / r["path"], dst)
            rows.append({**_row(uid, lang, r["text"], r["keywords"], "keyword")})
        gen = pd.read_parquet(data / "datasets" / f"eval-clean-{lang}-v1" / "manifest.parquet")
        gen = gen.head(generic_per_lang)
        for r in gen.to_dict("records"):
            uid = f"rg-{lang}-{r['utterance_id']}"
            dst = out / "audio" / f"{uid}.wav"
            if not dst.exists():
                os.link(data / "datasets" / f"eval-clean-{lang}-v1" / r["path"], dst)
            rows.append({**_row(uid, lang, r["text"], [], "generic")})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "manifest.parquet", index=False)
    (out / "dataset.json").write_text(
        json.dumps({"dataset_id": "regression-keywords-v1", "row_count": len(rows)}, indent=2)
    )
    print(f"regression-keywords-v1: {len(rows)} rows -> {out}")
    return out


def _row(uid, lang, text, keywords, part):
    return {
        "utterance_id": uid,
        "path": f"audio/{uid}.wav",
        "lang": lang,
        "text": text,
        "part": part,
        "keywords": list(keywords) if keywords is not None else [],
    }


def run_regression(engine, settings: Settings, corpus_dir: str | Path) -> dict:
    """Return {lang: {keyword_recall, generic_wer}} for one engine."""
    corpus_dir = Path(corpus_dir)
    df = pd.read_parquet(corpus_dir / "manifest.parquet")
    out: dict = {}
    for lang in sorted(df["lang"].unique()):
        sub = df[df["lang"] == lang]
        kw_records, gen_refs, gen_hyps = [], [], []
        for r in sub.to_dict("records"):
            audio, sr = sf.read(str(corpus_dir / r["path"]), dtype="float32", always_2d=True)
            raw = engine.transcribe(audio.mean(axis=1), SR, language=lang)
            guarded = apply_guard(raw, None, settings.asr.guard)
            if r["part"] == "keyword":
                kws = list(r["keywords"]) if r["keywords"] is not None else []
                kw_records.append(
                    UttRecord(ref=r["text"], hyp=guarded.text, lang=lang, keywords=kws)
                )
            else:
                gen_refs.append(r["text"])
                gen_hyps.append(guarded.text)
        recall, _ = keyword_recall(kw_records, lang)
        gen_wer, _ = wer_cer(gen_refs, gen_hyps, lang) if gen_refs else (None, None)
        out[lang] = {"keyword_recall": round(recall, 4), "generic_wer": gen_wer}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS regression suite")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--build", action="store_true")
    args = p.parse_args(argv)
    settings = Settings.load(args.config)
    if args.build:
        build_corpus(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
