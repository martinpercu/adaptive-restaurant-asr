"""Public dataset bootstrap (plan/phases/phase-0 §0.5).

Idempotent: each source downloads -> checksum-verifies (when a checksum is known)
-> converts to 16 kHz mono WAV -> writes a manifest per plan/03-data-spec.md §1.
Clean-speech manifests land under `data/clean/<lang>/`; raw noise is staged under
`data/_staging/noise_sources/<source>/` with a license file, for phase 2 to curate.

No-auth direct downloads only (see plan/phases/phase-0 §0.5 table). Run:
    python -m scripts.download_datasets all
    python -m scripts.download_datasets openslr61
    python -m scripts.download_datasets librispeech --limit-hours 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ars.config import DEFAULT_CONFIG, Settings
from scripts.audio_io import to_wav_16k_mono

GENERATOR_VERSION = "0.1.0"


# --------------------------------------------------------------------------- #
# Source registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpeechSource:
    name: str
    lang: str
    accent: str
    license: str
    urls: list[str]  # zip/tar archives, each with a line_index.tsv + audio


@dataclass(frozen=True)
class NoiseSource:
    name: str
    license: str
    urls: list[str]
    intended_subtypes: list[str] = field(default_factory=list)


_OPENSLR = "https://www.openslr.org/resources"

SPEECH_SOURCES: dict[str, SpeechSource] = {
    "openslr61": SpeechSource(
        "openslr61",
        "es",
        "es-ar",
        "CC-BY-SA-4.0",
        [f"{_OPENSLR}/61/es_ar_female.zip", f"{_OPENSLR}/61/es_ar_male.zip"],
    ),
    "openslr71": SpeechSource(
        "openslr71",
        "es",
        "es-cl",
        "CC-BY-SA-4.0",
        [f"{_OPENSLR}/71/es_cl_female.zip", f"{_OPENSLR}/71/es_cl_male.zip"],
    ),
    "openslr72": SpeechSource(
        "openslr72",
        "es",
        "es-co",
        "CC-BY-SA-4.0",
        [f"{_OPENSLR}/72/es_co_female.zip", f"{_OPENSLR}/72/es_co_male.zip"],
    ),
    "openslr73": SpeechSource(
        "openslr73",
        "es",
        "es-pe",
        "CC-BY-SA-4.0",
        [f"{_OPENSLR}/73/es_pe_female.zip", f"{_OPENSLR}/73/es_pe_male.zip"],
    ),
    "openslr74": SpeechSource(
        "openslr74",
        "es",
        "es-pr",
        "CC-BY-SA-4.0",
        [f"{_OPENSLR}/74/es_pr_female.zip"],
    ),
    "openslr75": SpeechSource(
        "openslr75",
        "es",
        "es-ve",
        "CC-BY-SA-4.0",
        [f"{_OPENSLR}/75/es_ve_female.zip", f"{_OPENSLR}/75/es_ve_male.zip"],
    ),
    "librispeech": SpeechSource(
        "librispeech",
        "en",
        "en-us",
        "CC-BY-4.0",
        [f"{_OPENSLR}/12/dev-clean.tar.gz", f"{_OPENSLR}/12/test-clean.tar.gz"],
    ),
}

NOISE_SOURCES: dict[str, NoiseSource] = {
    "demand": NoiseSource(
        "demand",
        "CC-BY-SA-3.0",
        [
            "https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip",
            "https://zenodo.org/record/1227121/files/PCAFETER_16k.zip",
            "https://zenodo.org/record/1227121/files/OMEETING_16k.zip",
            "https://zenodo.org/record/1227121/files/TMETRO_16k.zip",
        ],
        intended_subtypes=["AA", "AB", "CA", "BA"],
    ),
    "musan": NoiseSource(
        "musan",
        "CC-BY-4.0",
        [f"{_OPENSLR}/17/musan.tar.gz"],
        intended_subtypes=["AC", "CA", "CB"],
    ),
    "urbansound8k": NoiseSource(
        "urbansound8k",
        "CC-BY-NC-3.0",
        ["https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"],
        intended_subtypes=["BA", "BB", "BC"],
    ),
}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        print(f"  cached: {dest.name}")
        return dest
    print(f"  downloading: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)
    if sha256 is not None and _sha256(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {url}")
    tmp.replace(dest)
    return dest


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"unknown archive type: {archive.name}")
    return dest


def _read_line_index(root: Path) -> dict[str, str]:
    """OpenSLR crowdsourced series: line_index.tsv maps <file_id> -> transcript."""
    mapping: dict[str, str] = {}
    for tsv in root.rglob("line_index.tsv"):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) >= 2:
                mapping[parts[0]] = parts[-1]
    return mapping


def _read_librispeech_trans(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for trans in root.rglob("*.trans.txt"):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, _, text = line.partition(" ")
            if uid and text:
                mapping[uid] = text.strip()
    return mapping


# --------------------------------------------------------------------------- #
# Speech ingestion
# --------------------------------------------------------------------------- #
def ingest_speech(src: SpeechSource, settings: Settings, limit_hours: float | None) -> Path:
    data_root = Path(settings.paths.data)
    staging = data_root / "_staging" / src.name
    out_dir = data_root / "clean" / src.lang
    audio_dir = out_dir / src.name
    audio_dir.mkdir(parents=True, exist_ok=True)

    for url in src.urls:
        archive = download_file(url, staging / Path(url).name)
        extract_archive(archive, staging / "extracted")

    ext_root = staging / "extracted"
    is_libri = src.name == "librispeech"
    transcripts = _read_librispeech_trans(ext_root) if is_libri else _read_line_index(ext_root)
    audio_glob = "*.flac" if is_libri else "*.wav"

    rows: list[dict] = []
    total_s = 0.0
    seq = 0
    limit_s = None if limit_hours is None else limit_hours * 3600
    for audio in sorted(ext_root.rglob(audio_glob)):
        uid = audio.stem
        text = transcripts.get(uid)
        if not text:
            continue
        rel = f"{src.name}/{uid}.wav"
        dur = to_wav_16k_mono(audio, out_dir / rel)
        rows.append(
            {
                "utterance_id": f"cl-{src.lang}-{seq:05d}",
                "path": rel,
                "lang": src.lang,
                "text": text,
                "duration_s": round(dur, 3),
                "source": src.name,
                "accent": src.accent,
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
        seq += 1
        total_s += dur
        if limit_s is not None and total_s >= limit_s:
            break

    _write_manifest(out_dir, rows, src, settings)
    print(f"  {src.name}: {len(rows)} utts, {total_s / 3600:.2f} h -> {out_dir}")
    return out_dir


def _write_manifest(out_dir: Path, rows: list[dict], src: SpeechSource, settings: Settings) -> None:
    if not rows:
        raise RuntimeError(f"{src.name}: no utterances ingested (check archive layout)")
    df = pd.DataFrame(rows)
    # merge with any manifest already there (multiple sources share data/clean/<lang>)
    manifest = out_dir / "manifest.parquet"
    if manifest.exists():
        prev = pd.read_parquet(manifest)
        prev = prev[prev["source"] != src.name]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(manifest, index=False)
    info = {
        "dataset_id": f"clean-{src.lang}-v1",
        "created_at": _now(),
        "generator": "scripts.download_datasets",
        "generator_version": GENERATOR_VERSION,
        "config_hash": "",
        "seed": settings.seed,
        "row_count": int(len(df)),
        "langs": [src.lang],
    }
    (out_dir / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Noise staging
# --------------------------------------------------------------------------- #
def stage_noise(src: NoiseSource, settings: Settings) -> Path:
    staging = Path(settings.paths.data) / "_staging" / "noise_sources" / src.name
    staging.mkdir(parents=True, exist_ok=True)
    for url in src.urls:
        archive = download_file(url, staging / Path(url).name)
        extract_archive(archive, staging / "extracted")
    (staging / "LICENSE.txt").write_text(
        f"source: {src.name}\nlicense: {src.license}\n"
        f"intended_subtypes: {', '.join(src.intended_subtypes)}\n"
        f"staged_at: {_now()}\n",
        encoding="utf-8",
    )
    print(f"  {src.name}: staged -> {staging} (license {src.license})")
    return staging


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS dataset bootstrap")
    choices = ["all", *SPEECH_SOURCES, *NOISE_SOURCES]
    parser.add_argument("source", choices=choices)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--limit-hours", type=float, default=None, help="cap ingested hours per speech source"
    )
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)

    speech = list(SPEECH_SOURCES) if args.source == "all" else [args.source]
    noise = list(NOISE_SOURCES) if args.source == "all" else [args.source]

    for name in speech:
        if name in SPEECH_SOURCES:
            print(f"[speech] {name}")
            ingest_speech(SPEECH_SOURCES[name], settings, args.limit_hours)
    for name in noise:
        if name in NOISE_SOURCES:
            print(f"[noise] {name}")
            stage_noise(NOISE_SOURCES[name], settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
