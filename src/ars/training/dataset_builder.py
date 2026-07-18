"""Damage-weighted noisy training set (plan/phases/phase-4 §4.1, §4.2).

Sampling is weighted by the phase-2 NDI: subtype ~ softmax(NDI / T), so the model
trains hardest on the noises that hurt it most. Levels draw a *continuous* SNR from
per-band ranges (avoids overfitting to three exact ratios). 30% clean anchors against
catastrophic forgetting. Residual-damage subtypes (no phase-3 mitigation) get an NDI
boost. Only train-split noise clips are used. Deterministic given (config, NDI, seed).

    python -m ars.training.dataset_builder build --size-hours 20 \
        --ndi reports/sensitivity/<run>/ndi.json
    python -m ars.training.dataset_builder eval --per-cell 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ars.config import DEFAULT_CONFIG, Settings
from ars.noise_lab.mixer import mix
from ars.vad import SileroVad

SR = 16000
GENERATOR_VERSION = "0.1.0"
# continuous SNR bands per level (dB) — §4.1
SNR_BANDS = {"05": (8.0, 15.0), "10": (-2.0, 2.0), "15": (-7.0, -3.0)}
CANONICAL_LEVELS = ("05", "10", "15")


def ndi_by_subtype(ndi_json: dict, lang: str) -> dict[str, float]:
    return {e["subtype"]: e["ndi"] for e in ndi_json["ranking"] if e["lang"] == lang}


def softmax_weights(
    ndi: dict[str, float],
    subtypes: list[str],
    temperature: float,
    residual: set[str] | None = None,
    boost: float = 1.5,
) -> dict[str, float]:
    """subtype -> sampling probability = softmax(NDI/T), residual subtypes boosted ×boost."""
    residual = residual or set()
    vals = np.array(
        [ndi.get(s, 0.0) * (boost if s in residual else 1.0) for s in subtypes], dtype=float
    )
    z = vals / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    p = e / e.sum()
    return dict(zip(subtypes, p, strict=True))


def _load(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz")
    return a.mean(axis=1).astype(np.float32)


def _kw(v) -> list:
    """Coerce a keywords cell (may be a numpy array from parquet) to a list."""
    return list(v) if v is not None else []


def _config_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _clean_pool(settings: Settings, lang: str, exclude: set[str]) -> pd.DataFrame:
    """Clean utts (TTS domain + read speech) minus held-out ids, as a train pool."""
    data = Path(settings.paths.data)
    frames = []
    tts = data / "datasets" / f"tts-{lang}-v1" / "manifest.parquet"
    if tts.exists():
        df = pd.read_parquet(tts)
        df["root"] = str(tts.parent)
        frames.append(df)
    clean = data / "clean" / lang / "manifest.parquet"
    if clean.exists():
        df = pd.read_parquet(clean)
        df["root"] = str(clean.parent)
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    return pool[~pool["utterance_id"].isin(exclude)].reset_index(drop=True)


def _reserved_eval_ids(settings: Settings, lang: str) -> set[str]:
    """Ids already committed to eval sets (eval-clean, eval-matrix) — never train on them."""
    data = Path(settings.paths.data)
    ids: set[str] = set()
    for name in (f"eval-clean-{lang}-v1", f"eval-matrix-{lang}-v1"):
        m = data / "datasets" / name / "manifest.parquet"
        if m.exists():
            df = pd.read_parquet(m)
            ids |= set(df["utterance_id"])
            ids |= set(df["clean_id"].dropna())
    return ids


def _mix_row(clean_audio, clean_row, subtype, snr, noise_path, s_rms, seed, out_dir, uid):
    noise = _load(Path(noise_path))
    res = mix(clean_audio, noise, snr, s_rms, seed)
    rel = f"audio/{uid}.wav"
    sf.write(str(out_dir / rel), np.clip(res.mixed, -1, 1), SR, subtype="PCM_16")
    return {
        "utterance_id": uid,
        "path": rel,
        "lang": clean_row["lang"],
        "text": clean_row["text"],
        "duration_s": round(len(res.mixed) / SR, 3),
        "source": "train-noisy",
        "accent": clean_row.get("accent"),
        "clean_id": clean_row["utterance_id"],
        "noise_subtype": subtype,
        "noise_level": None,
        "noise_clip_id": Path(noise_path).stem,
        "snr_db_target": round(snr, 2),
        "snr_db_achieved": round(res.achieved_snr_db, 3),
        "mix_seed": seed,
        "keywords": _kw(clean_row.get("keywords")),
    }


def build_train(
    settings,
    size_hours,
    ndi_file,
    seed,
    clean_frac=0.3,
    temperature=0.5,
    residual_boost=1.5,
    dataset_id="train-noisy-xx-v1",
):
    data = Path(settings.paths.data)
    ndi_json = json.loads(Path(ndi_file).read_text())
    bank = pd.read_parquet(data / "noise_bank" / "manifest.parquet")
    bank_train = bank[bank["split"] == "train"]
    subtypes = sorted(bank_train["subtype"].unique())
    # residual = subtypes the phase-3 policy left as `none`
    residual = _policy_residual(settings, subtypes)

    out_dir = data / "datasets" / dataset_id
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    vad = SileroVad(settings.vad)
    rng = random.Random(seed)

    target_s = size_hours * 3600.0
    rows: list[dict] = []
    total_s = 0.0
    seq = 0
    # weights per lang
    weights = {
        lang: softmax_weights(
            ndi_by_subtype(ndi_json, lang), subtypes, temperature, residual, residual_boost
        )
        for lang in ("es", "en")
    }
    pools = {
        lang: _clean_pool(settings, lang, _reserved_eval_ids(settings, lang))
        for lang in ("es", "en")
    }
    clips_by_st = {
        st: bank_train[bank_train["subtype"] == st].reset_index(drop=True) for st in subtypes
    }

    while total_s < target_s:
        lang = "es" if rng.random() < 0.5 else "en"
        pool = pools[lang]
        crow = pool.iloc[rng.randrange(len(pool))].to_dict()
        clean_audio = _load(Path(crow["root"]) / crow["path"])
        uid = f"tn-{lang}-{seq:06d}"
        if rng.random() < clean_frac:
            rel = f"audio/{uid}.wav"
            sf.write(str(out_dir / rel), np.clip(clean_audio, -1, 1), SR, subtype="PCM_16")
            rows.append(
                {
                    "utterance_id": uid,
                    "path": rel,
                    "lang": lang,
                    "text": crow["text"],
                    "duration_s": round(len(clean_audio) / SR, 3),
                    "source": "train-noisy",
                    "accent": crow.get("accent"),
                    "clean_id": crow["utterance_id"],
                    "noise_subtype": None,
                    "noise_level": None,
                    "noise_clip_id": None,
                    "snr_db_target": None,
                    "snr_db_achieved": None,
                    "mix_seed": None,
                    "keywords": _kw(crow.get("keywords")),
                }
            )
        else:
            w = weights[lang]
            st = rng.choices(subtypes, weights=[w[s] for s in subtypes])[0]
            level = rng.choice(CANONICAL_LEVELS)
            lo, hi = SNR_BANDS[level]
            snr = rng.uniform(lo, hi)
            clips = clips_by_st[st]
            noise_path = data / "noise_bank" / clips.iloc[rng.randrange(len(clips))]["path"]
            vr = vad.detect(clean_audio, SR)
            s_rms = vr.speech_rms if vr.speech_rms > 0 else float(np.sqrt(np.mean(clean_audio**2)))
            rows.append(
                _mix_row(
                    clean_audio,
                    crow,
                    st,
                    snr,
                    noise_path,
                    s_rms,
                    rng.randrange(1 << 30),
                    out_dir,
                    uid,
                )
            )
        total_s += rows[-1]["duration_s"]
        seq += 1

    _write(
        out_dir,
        rows,
        dataset_id,
        settings,
        seed,
        {
            "size_hours": size_hours,
            "clean_frac": clean_frac,
            "temperature": temperature,
            "residual_boost": residual_boost,
            "residual": sorted(residual),
            "ndi_file": ndi_file,
        },
    )
    return out_dir


def build_eval_noisy(settings, per_cell=40, seed=1337):
    """Frozen noisy eval sets: disjoint clean holdout × eval-split noise × 7 subtypes × 3 levels."""
    data = Path(settings.paths.data)
    bank = pd.read_parquet(data / "noise_bank" / "manifest.parquet")
    bank_eval = bank[bank["split"] == "eval"]
    subtypes = sorted(bank_eval["subtype"].unique())
    vad = SileroVad(settings.vad)
    out_dirs = []
    for lang in ("es", "en"):
        rng = random.Random(seed + (0 if lang == "es" else 1))
        # holdout: reuse eval-clean utts (already disjoint from train), by their tts/clean source
        pool = _eval_clean_pool(settings, lang, per_cell)
        out_dir = data / "datasets" / f"eval-noisy-{lang}-v1"
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
        clips_by_st = {
            st: bank_eval[bank_eval["subtype"] == st].reset_index(drop=True) for st in subtypes
        }
        rr = dict.fromkeys(subtypes, 0)
        rows = []
        for st in subtypes:
            for level in CANONICAL_LEVELS:
                lo, hi = SNR_BANDS[level]
                for i in range(per_cell):
                    crow = pool[i % len(pool)]
                    clean_audio = crow["audio"]
                    clips = clips_by_st[st]
                    clip = clips.iloc[rr[st] % len(clips)]
                    rr[st] += 1
                    snr = rng.uniform(lo, hi)
                    uid = f"en-{lang}-{st}-{level}-{i:03d}"
                    noise = _load(data / "noise_bank" / clip["path"])
                    res = mix(clean_audio, noise, snr, crow["s_rms"], rng.randrange(1 << 30))
                    rel = f"audio/{uid}.wav"
                    sf.write(str(out_dir / rel), np.clip(res.mixed, -1, 1), SR, subtype="PCM_16")
                    rows.append(
                        {
                            "utterance_id": uid,
                            "path": rel,
                            "lang": lang,
                            "text": crow["text"],
                            "duration_s": round(len(res.mixed) / SR, 3),
                            "source": "eval-noisy",
                            "accent": None,
                            "clean_id": crow["id"],
                            "noise_subtype": st,
                            "noise_level": level,
                            "noise_clip_id": clip["clip_id"],
                            "snr_db_target": round(snr, 2),
                            "snr_db_achieved": round(res.achieved_snr_db, 3),
                            "mix_seed": None,
                            "keywords": _kw(crow["keywords"]),
                        }
                    )
        _write(
            out_dir,
            rows,
            f"eval-noisy-{lang}-v1",
            settings,
            seed,
            {"per_cell": per_cell, "subtypes": subtypes},
        )
        _ = vad
        out_dirs.append(out_dir)
    return out_dirs


def _eval_clean_pool(settings, lang, need):
    data = Path(settings.paths.data)
    d = data / "datasets" / f"eval-clean-{lang}-v1"
    df = pd.read_parquet(d / "manifest.parquet").head(max(need, 40))
    vad = SileroVad(settings.vad)
    pool = []
    for r in df.to_dict("records"):
        audio = _load(d / r["path"])
        vr = vad.detect(audio, SR)
        s_rms = vr.speech_rms if vr.speech_rms > 0 else float(np.sqrt(np.mean(audio**2)))
        pool.append(
            {
                "id": r["utterance_id"],
                "text": r["text"],
                "audio": audio,
                "s_rms": s_rms,
                "keywords": _kw(r.get("keywords")),
            }
        )
    return pool


def _policy_residual(settings, subtypes) -> set[str]:
    path = Path(settings.preprocess.policy_path)
    if not path.exists():
        return set(subtypes)
    import yaml  # noqa: PLC0415

    policy = yaml.safe_load(path.read_text())["policy"]
    return {s for s in subtypes if policy.get(s, "none") == "none"}


def _write(out_dir: Path, rows, dataset_id, settings, seed, extra):
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "manifest.parquet", index=False)
    info = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "ars.training.dataset_builder",
        "generator_version": GENERATOR_VERSION,
        "config_hash": _config_hash({"seed": seed, **extra}),
        "seed": seed,
        "row_count": len(rows),
        "langs": sorted(df["lang"].unique()),
        **extra,
    }
    (out_dir / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    n_clean = int(df["noise_subtype"].isna().sum())
    print(
        f"{dataset_id}: {len(rows)} rows ({n_clean} clean, {len(rows) - n_clean} mixed), "
        f"{df['duration_s'].sum() / 3600:.2f} h -> {out_dir}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS training dataset builder")
    p.add_argument("mode", choices=["build", "eval"])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--size-hours", type=float, default=20.0)
    p.add_argument("--ndi", default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--per-cell", type=int, default=40)
    p.add_argument("--dataset-id", default="train-noisy-xx-v1")
    args = p.parse_args(argv)
    settings = Settings.load(args.config)
    if args.mode == "build":
        ndi = args.ndi or _latest_ndi(settings)
        build_train(settings, args.size_hours, ndi, args.seed, dataset_id=args.dataset_id)
    else:
        build_eval_noisy(settings, args.per_cell, args.seed)
    return 0


def _latest_ndi(settings) -> str:
    runs = sorted((Path(settings.paths.reports) / "sensitivity").glob("run-*"))
    if not runs:
        raise SystemExit("no sensitivity NDI report found; run phase 2 first")
    return str(runs[-1] / "ndi.json")


if __name__ == "__main__":
    raise SystemExit(main())
