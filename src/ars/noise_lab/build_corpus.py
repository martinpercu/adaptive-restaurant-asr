"""Matrix corpus builder (plan/phases/phase-2 §2.3).

Selects clean, keyword-bearing utterances per lang and crosses them with every noise
subtype × canonical level using **eval-split** bank clips (round-robin), producing the
eval-matrix datasets. SNR is set against speech-active RMS (production VAD). Mixing is
deterministic per (clean_id, clip_id, level).

    python -m ars.noise_lab.build_corpus --config configs/default.yaml \
        --langs es en --clean-per-lang 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ars.config import DEFAULT_CONFIG, Settings
from ars.noise_lab.mixer import mix
from ars.noise_lab.taxonomy import load_taxonomy
from ars.vad import SileroVad

SR = 16000
GENERATOR_VERSION = "0.1.0"


def _seed(base: int, clean_id: str, clip_id: str, level: str) -> int:
    h = hashlib.sha256(f"{base}|{clean_id}|{clip_id}|{level}".encode()).hexdigest()
    return int(h[:8], 16)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz")
    return audio.mean(axis=1).astype(np.float32)


def _clean_row(uid: str, path: str, lang: str, text: str, dur: float, keywords) -> dict:
    return {
        "utterance_id": uid,
        "path": path,
        "lang": lang,
        "text": text,
        "duration_s": round(dur, 3),
        "source": "eval-matrix",
        "accent": None,
        "clean_id": None,
        "noise_subtype": None,
        "noise_level": None,
        "noise_clip_id": None,
        "snr_db_target": None,
        "snr_db_achieved": None,
        "mix_seed": None,
        "keywords": list(keywords) if keywords is not None else [],
    }


def build_lang(lang: str, settings: Settings, clean_per_lang: int, levels: list[str]) -> Path:
    data = Path(settings.paths.data)
    tax = load_taxonomy(settings.paths.noise_taxonomy)
    level_snr = {lv: tax.levels[lv].snr_db for lv in levels}

    # clean base: keyword-bearing utterances from the TTS domain corpus (held-out, clean)
    tts_dir = data / "datasets" / f"tts-{lang}-v1"
    clean_df = pd.read_parquet(tts_dir / "manifest.parquet")
    clean_df = clean_df.sample(n=min(clean_per_lang, len(clean_df)), random_state=settings.seed)
    clean_df = clean_df.reset_index(drop=True)

    # eval-split noise clips grouped by subtype
    bank = pd.read_parquet(data / "noise_bank" / "manifest.parquet")
    bank_eval = bank[bank["split"] == "eval"]
    subtypes = sorted(bank_eval["subtype"].unique())
    clips_by_subtype = {
        st: bank_eval[bank_eval["subtype"] == st].reset_index(drop=True) for st in subtypes
    }
    rr_index = dict.fromkeys(subtypes, 0)  # round-robin cursor per subtype

    out_dir = data / "datasets" / f"eval-matrix-{lang}-v1"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    vad = SileroVad(settings.vad)
    rows: list[dict] = []
    for c in clean_df.to_dict(orient="records"):
        clean_id = c["utterance_id"]
        clean_audio = _load(tts_dir / c["path"])
        # clean row (copy audio into the matrix dataset)
        clean_rel = f"audio/{clean_id}.wav"
        shutil.copyfile(tts_dir / c["path"], out_dir / clean_rel)
        rows.append(
            _clean_row(clean_id, clean_rel, lang, c["text"], len(clean_audio) / SR, c["keywords"])
        )
        # speech-active RMS via the production VAD (cached per clean utterance)
        vr = vad.detect(clean_audio, SR)
        s_rms = vr.speech_rms if vr.speech_rms > 0 else float(np.sqrt(np.mean(clean_audio**2)))

        for st in subtypes:
            clips = clips_by_subtype[st]
            for level in levels:
                clip = clips.iloc[rr_index[st] % len(clips)]
                rr_index[st] += 1
                noise = _load(data / "noise_bank" / clip["path"])
                seed = _seed(settings.seed, clean_id, clip["clip_id"], level)
                res = mix(clean_audio, noise, level_snr[level], s_rms, seed)
                mixed_id = f"{clean_id}__noise-{st}-{level}"
                rel = f"audio/{mixed_id}.wav"
                sf.write(str(out_dir / rel), np.clip(res.mixed, -1, 1), SR, subtype="PCM_16")
                rows.append(
                    {
                        "utterance_id": mixed_id,
                        "path": rel,
                        "lang": lang,
                        "text": c["text"],
                        "duration_s": round(len(res.mixed) / SR, 3),
                        "source": "eval-matrix",
                        "accent": None,
                        "clean_id": clean_id,
                        "noise_subtype": st,
                        "noise_level": level,
                        "noise_clip_id": clip["clip_id"],
                        "snr_db_target": level_snr[level],
                        "snr_db_achieved": round(res.achieved_snr_db, 3),
                        "mix_seed": seed,
                        "keywords": list(c["keywords"]) if c["keywords"] is not None else [],
                    }
                )

    pd.DataFrame(rows).to_parquet(out_dir / "manifest.parquet", index=False)
    info = {
        "dataset_id": f"eval-matrix-{lang}-v1",
        "created_at": _now(),
        "generator": "ars.noise_lab.build_corpus",
        "generator_version": GENERATOR_VERSION,
        "config_hash": "",
        "seed": settings.seed,
        "row_count": len(rows),
        "langs": [lang],
        "subtypes": subtypes,
        "levels": levels,
        "clean_per_lang": len(clean_df),
    }
    (out_dir / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(
        f"[{lang}] {len(clean_df)} clean x {len(subtypes)} subtypes x {len(levels)} levels "
        f"= {len(rows)} rows -> {out_dir}"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS noise matrix corpus builder")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--langs", nargs="+", default=["es", "en"], choices=["es", "en"])
    parser.add_argument("--clean-per-lang", type=int, default=60)
    parser.add_argument("--levels", nargs="+", default=["05", "10", "15"])
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    for lang in args.langs:
        build_lang(lang, settings, args.clean_per_lang, args.levels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
