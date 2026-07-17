"""Noise-bank curation (plan/phases/phase-2 §2.1).

Turns staged source audio into a labeled noise bank: clips of ~10-30 s, canonical
16 kHz mono, with clip_id / subtype / license / split. Split is by SOURCE RECORDING
(never by clip) — the leakage guard. DEMAND single recordings are split by time block
(train early / eval late) so no clip spans both splits. BC car-cabin clips are built as
composites (meeting speech over an engine bed). Driven by configs/noise_curation.yaml.

    python -m scripts... no — module CLI:
    python -m ars.noise_lab.curate --config configs/default.yaml            # all subtypes
    python -m ars.noise_lab.curate --config configs/default.yaml --subtype AB
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ars.config import DEFAULT_CONFIG, Settings
from ars.noise_lab.mixer import rms

SR = 16000


@dataclass
class Clip:
    audio: np.ndarray
    recording_id: str
    split: str  # train | eval


def _load(path: str | Path) -> np.ndarray:
    import soundfile as sf  # noqa: PLC0415

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != SR:
        import torch  # noqa: PLC0415
        import torchaudio.functional as AF  # noqa: PLC0415

        audio = AF.resample(torch.from_numpy(audio).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return audio.astype(np.float32)


def _cut(stream: np.ndarray, clip_len: int, hop: int, min_len: int) -> list[np.ndarray]:
    out = []
    i = 0
    while i + min_len <= len(stream):
        clip = stream[i : i + clip_len]
        if len(clip) >= min_len:
            out.append(np.ascontiguousarray(clip))
        i += hop
    return out


# --------------------------------------------------------------------------- #
# Source adapters -> list[Clip]
# --------------------------------------------------------------------------- #
def _demand_stream(staging: Path, env: str, channel: int) -> np.ndarray:
    path = staging / "noise_sources" / "demand" / "extracted" / env / f"ch{channel:02d}.wav"
    if not path.exists():
        raise FileNotFoundError(f"DEMAND source missing: {path}")
    return _load(path)


def _demand_clips(
    cfg: dict, staging: Path, clip_len: int, hop: int, min_len: int, eval_frac: float
) -> list[Clip]:
    stream = _demand_stream(staging, cfg["env"], cfg.get("channel", 1))
    split_at = int(len(stream) * (1 - eval_frac))
    clips: list[Clip] = []
    for arr in _cut(stream[:split_at], clip_len, hop, min_len):
        clips.append(Clip(arr, f"{cfg['env']}-train", "train"))
    for arr in _cut(stream[split_at:], clip_len, hop, min_len):
        clips.append(Clip(arr, f"{cfg['env']}-eval", "eval"))
    return clips


def _us8k_root(staging: Path) -> Path:
    return staging / "noise_sources" / "urbansound8k" / "extracted" / "UrbanSound8K"


def _us8k_streams_by_recording(staging: Path, classes: list[str]) -> dict[str, np.ndarray]:
    """Concatenate all slices sharing an fsID (one freesound recording) into one stream."""
    root = _us8k_root(staging)
    meta = root / "metadata" / "UrbanSound8K.csv"
    if not meta.exists():
        raise FileNotFoundError(f"UrbanSound8K missing: {meta} (run download-data urbansound8k)")
    by_fs: dict[str, list[tuple[float, Path]]] = {}
    with meta.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["class"] in classes:
                p = root / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
                by_fs.setdefault(f"{r['class']}-{r['fsID']}", []).append((float(r["start"]), p))
    streams: dict[str, np.ndarray] = {}
    for fs, slices in by_fs.items():
        parts = [_load(p) for _, p in sorted(slices)]
        if parts:
            streams[fs] = np.concatenate(parts).astype(np.float32)
    return streams


def _us8k_clips(
    cfg: dict, staging: Path, clip_len: int, hop: int, min_len: int, eval_frac: float, seed: int
) -> list[Clip]:
    streams = _us8k_streams_by_recording(staging, cfg["classes"])
    recordings = sorted(streams)
    rng = random.Random(seed)
    rng.shuffle(recordings)
    n_eval = max(1, round(len(recordings) * eval_frac))
    eval_set = set(recordings[:n_eval])
    clips: list[Clip] = []
    for rec in recordings:
        split = "eval" if rec in eval_set else "train"
        for arr in _cut(streams[rec], clip_len, hop, min_len):
            clips.append(Clip(arr, rec, split))
    return clips


def _composite_clips(
    cfg: dict, staging: Path, clip_len: int, hop: int, min_len: int, eval_frac: float, seed: int
) -> list[Clip]:
    """BC car-cabin: sparse meeting speech layered over an engine bed at rel_db."""
    speech = _demand_clips(cfg["speech"], staging, clip_len, hop, min_len, eval_frac)
    bed = _us8k_clips(cfg["bed"], staging, clip_len, hop, min_len, eval_frac, seed)
    rel = float(cfg.get("rel_db", -3.0))
    rng = random.Random(seed + 1)
    out: list[Clip] = []
    for split in ("train", "eval"):
        sp = [c for c in speech if c.split == split]
        bd = [c for c in bed if c.split == split]
        if not sp or not bd:
            continue
        for i, b in enumerate(bd):
            s = sp[i % len(sp)].audio
            n = len(b.audio)
            s = np.resize(s, n) if len(s) < n else s[:n]
            s_rms, b_rms = rms(s), rms(b.audio)
            if s_rms < 1e-9 or b_rms < 1e-9:
                continue
            gain = (b_rms * (10.0 ** (rel / 20.0))) / s_rms
            comp = b.audio + gain * s
            peak = float(np.max(np.abs(comp)))
            if peak > 0.9:
                comp = comp * (0.9 / peak)
            out.append(Clip(comp.astype(np.float32), f"BC-{b.recording_id}", split))
        _ = rng  # seed reserved for future randomized pairing
    return out


def curate_subtype(subtype: str, cfg: dict, params: dict, staging: Path, seed: int) -> list[Clip]:
    clip_len = int(params["clip_len_s"] * SR)
    hop = int(params["hop_s"] * SR)
    min_len = int(3.0 * SR)  # NoiseBankRow requires duration >= 3 s
    ef = params["eval_frac"]
    kind = cfg["kind"]
    if kind == "demand":
        return _demand_clips(cfg, staging, clip_len, hop, min_len, ef)
    if kind == "urbansound8k":
        return _us8k_clips(cfg, staging, clip_len, hop, min_len, ef, seed)
    if kind == "composite":
        return _composite_clips(cfg, staging, clip_len, hop, min_len, ef, seed)
    raise ValueError(f"unknown curation kind: {kind}")


def write_bank(subtypes: dict[str, list[Clip]], curation: dict, out_dir: Path) -> pd.DataFrame:
    import soundfile as sf  # noqa: PLC0415

    rows: list[dict] = []
    for subtype, clips in subtypes.items():
        lic = curation["subtypes"][subtype]["license"]
        sub_dir = out_dir / subtype
        sub_dir.mkdir(parents=True, exist_ok=True)
        for seq, clip in enumerate(clips):
            clip_id = f"nz-{subtype}-{seq:04d}"
            rel = f"{subtype}/{clip_id}.wav"
            sf.write(str(out_dir / rel), np.clip(clip.audio, -1, 1), SR, subtype="PCM_16")
            rows.append(
                {
                    "clip_id": clip_id,
                    "subtype": subtype,
                    "path": rel,
                    "duration_s": round(len(clip.audio) / SR, 3),
                    "source": _source_tag(curation["subtypes"][subtype]),
                    "license": lic,
                    "split": clip.split,
                    "recording_id": clip.recording_id,
                }
            )
    df = pd.DataFrame(rows)
    manifest = out_dir / "manifest.parquet"
    if manifest.exists():
        prev = pd.read_parquet(manifest)
        prev = prev[~prev["subtype"].isin(df["subtype"].unique())]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(manifest, index=False)
    return df


def _source_tag(cfg: dict) -> str:
    if cfg["kind"] == "demand":
        return f"demand-{cfg['env']}"
    if cfg["kind"] == "urbansound8k":
        return "urbansound8k"
    return "composite-bc"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS noise-bank curation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--subtype", default=None, help="curate only this subtype")
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    curation = yaml.safe_load(
        (Path(settings.paths.configs) / "noise_curation.yaml").read_text(encoding="utf-8")
    )
    staging = Path(settings.paths.data) / "_staging"
    out_dir = Path(settings.paths.data) / "noise_bank"

    wanted = [args.subtype] if args.subtype else list(curation["subtypes"])
    built: dict[str, list[Clip]] = {}
    for st in wanted:
        try:
            clips = curate_subtype(st, curation["subtypes"][st], curation, staging, settings.seed)
        except FileNotFoundError as exc:
            print(f"  SKIP {st}: {exc}")
            continue
        n_train = sum(c.split == "train" for c in clips)
        n_eval = sum(c.split == "eval" for c in clips)
        if n_train < curation["min_train"] or n_eval < curation["min_eval"]:
            print(f"  WARN {st}: only {n_train} train / {n_eval} eval clips (< minimum)")
        built[st] = clips
        print(f"  {st}: {n_train} train + {n_eval} eval clips")

    if built:
        # merge with existing manifest for subtypes not rebuilt this run
        df = write_bank(built, curation, out_dir)
        n_sub = df["subtype"].nunique()
        print(f"noise bank -> {out_dir} ({len(df)} clips across {n_sub} subtypes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
