"""Build held-out clean eval sets eval-clean-{es,en}-v1 (plan/phases/phase-1 §1.5).

Splits data/clean/<lang> by speaker/voice (seed 1337), hardlinks the held-out audio
into the dataset dir so manifest paths resolve from their own directory (keeps the
phase-0 manifest contract green), and writes manifest.parquet + dataset.json.

    python -m scripts.build_eval_sets --eval-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ars.config import DEFAULT_CONFIG, Settings

GENERATOR_VERSION = "0.1.0"


def speaker_of(path: str) -> str:
    """Derive a speaker/voice id from a clean-utterance relative path."""
    stem = Path(path).stem
    if "-" in stem and stem.split("-")[0].isdigit():  # librispeech: 1272-128104-0000
        return stem.split("-")[0]
    return "_".join(stem.split("_")[:2])  # openslr: prf_02484_...


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_lang(lang: str, settings: Settings, eval_frac: float, min_utts: int) -> Path | None:
    clean_dir = Path(settings.paths.data) / "clean" / lang
    manifest = clean_dir / "manifest.parquet"
    if not manifest.exists():
        print(f"[{lang}] no clean manifest; skipping")
        return None
    df = pd.read_parquet(manifest)
    df["speaker"] = df["path"].map(speaker_of)

    speakers = sorted(df["speaker"].unique())
    rng = random.Random(settings.seed)
    rng.shuffle(speakers)
    n_eval = max(1, round(len(speakers) * eval_frac))
    eval_speakers = set(speakers[:n_eval])
    eval_df = df[df["speaker"].isin(eval_speakers)].reset_index(drop=True)

    # top up to min_utts by adding whole speakers if the split came out small
    i = n_eval
    while len(eval_df) < min_utts and i < len(speakers):
        eval_speakers.add(speakers[i])
        eval_df = df[df["speaker"].isin(eval_speakers)].reset_index(drop=True)
        i += 1

    out_dir = Path(settings.paths.data) / "datasets" / f"eval-clean-{lang}-v1"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for row in eval_df.to_dict(orient="records"):
        uid = row["utterance_id"]
        src = clean_dir / row["path"]
        dst = audio_dir / f"{uid}.wav"
        if not dst.exists():
            os.link(src, dst)  # hardlink (same filesystem) — no duplication
        rows.append(
            {
                "utterance_id": uid,
                "path": f"audio/{uid}.wav",
                "lang": lang,
                "text": row["text"],
                "duration_s": row["duration_s"],
                "source": row["source"],
                "accent": row.get("accent"),
                "clean_id": None,
                "noise_subtype": None,
                "noise_level": None,
                "noise_clip_id": None,
                "snr_db_target": None,
                "snr_db_achieved": None,
                "mix_seed": None,
                "keywords": [],
            }
        )

    pd.DataFrame(rows).to_parquet(out_dir / "manifest.parquet", index=False)
    info = {
        "dataset_id": f"eval-clean-{lang}-v1",
        "created_at": _now(),
        "generator": "scripts.build_eval_sets",
        "generator_version": GENERATOR_VERSION,
        "config_hash": "",
        "seed": settings.seed,
        "row_count": len(rows),
        "langs": [lang],
        "eval_speakers": sorted(eval_speakers),
    }
    (out_dir / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[{lang}] {len(rows)} held-out utts from {len(eval_speakers)} speakers -> {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build held-out clean eval sets")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--min-utts", type=int, default=250)
    parser.add_argument("--langs", nargs="+", default=["es", "en"], choices=["es", "en"])
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    for lang in args.langs:
        build_lang(lang, settings, args.eval_frac, args.min_utts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
