"""Restaurant-order TTS domain corpus (plan/phases/phase-0 §0.6).

Two stages, deliberately split so the grammar/coverage is unit-testable without TTS:
  1. `generate_utterances(...)` — pure, seeded: builds order sentences from the demo
     menu x templates, and GUARANTEES every confusion `target` (configs/confusion_seed.yaml)
     appears in at least one utterance's `keywords` (KER targets).
  2. `synthesize(...)` — renders each sentence with Piper voices (>=3 per lang, prioritizing
     es_MX / en_US) into 16 kHz mono WAV. Requires a Piper binary; this is the local heavy path.

Output: data/datasets/tts-<lang>-v1/{manifest.parquet, dataset.json, audio/}.

    python -m scripts.tts_corpus                 # both languages, full synth
    python -m scripts.tts_corpus --dry-run       # write sentences.jsonl only, no audio
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from ars.config import DEFAULT_CONFIG, Settings
from scripts.audio_io import to_wav_16k_mono

GENERATOR_VERSION = "0.1.0"
MIN_PER_LANG = 500
SEQ_OFFSET = 50000  # keep TTS ids clear of downloaded clean-speech ids

# Piper voices per language, most-preferred first (locale -> accent tag).
VOICES: dict[str, list[tuple[str, str]]] = {
    # LatAm-first for Spanish (plan/phases/phase-0 §0.6); en_US for English.
    # Names verified against the rhasspy/piper-voices catalog.
    "es": [
        ("es_MX-claude-high", "es-mx"),
        ("es_MX-ald-medium", "es-mx"),
        ("es_AR-daniela-high", "es-ar"),
    ],
    "en": [
        ("en_US-amy-medium", "en-us"),
        ("en_US-ryan-high", "en-us"),
        ("en_US-lessac-medium", "en-us"),
    ],
}

TEMPLATES: dict[str, list[str]] = {
    "es": [
        "hola, me da {a} por favor",
        "quería {a}",
        "para mí {a} y {b}",
        "¿me traés {a}?",
        "una {a} para llevar",
        "me das {a} y también {b}",
    ],
    "en": [
        "hi, can I get {a} please",
        "I'd like {a}",
        "let me get {a} and {b}",
        "could I have {a}",
        "{a} to go please",
        "can I get {a} and {b} as well",
    ],
}

# Templates that embed a single confusion target verbatim (coverage guarantee).
TARGET_TEMPLATES: dict[str, list[str]] = {
    "es": ["me trae {w} por favor", "¿tiene {w}?", "quería el {w}", "¿dónde está la {w}?"],
    "en": ["can I get the {w}", "do you have a {w}", "I'd like the {w}", "where is the {w}"],
}


@dataclass
class Utterance:
    utterance_id: str
    lang: str
    text: str
    keywords: list[str] = field(default_factory=list)
    accent: str | None = None
    path: str | None = None
    duration_s: float | None = None


def load_menu(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_confusion(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def generate_utterances(
    lang: str, menu: dict, confusion: dict, n: int = MIN_PER_LANG, seed: int = 1337
) -> list[Utterance]:
    """Pure, deterministic. Every confusion target for `lang` is covered."""
    rng = random.Random(seed + (0 if lang == "es" else 1))
    item_names = [it["name"][lang] for it in menu["items"]]
    service_names = [s[lang] for s in menu.get("service_terms", [])]
    targets = [c["target"] for c in confusion.get(lang, [])]

    utts: list[Utterance] = []
    seq = SEQ_OFFSET

    def _add(text: str, keywords: list[str]) -> None:
        nonlocal seq
        utts.append(
            Utterance(
                utterance_id=f"cl-{lang}-{seq:05d}",
                lang=lang,
                text=text,
                keywords=sorted(set(keywords)),
                accent=VOICES[lang][0][1],
            )
        )
        seq += 1

    # 1) coverage pass — one utterance per confusion target
    for t in targets:
        tmpl = TARGET_TEMPLATES[lang][rng.randrange(len(TARGET_TEMPLATES[lang]))]
        _add(tmpl.format(w=t), [t])

    # 2) ordinary order sentences until we reach n
    pool = item_names + service_names
    while len(utts) < n:
        tmpl = TEMPLATES[lang][rng.randrange(len(TEMPLATES[lang]))]
        a = rng.choice(item_names)
        b = rng.choice(pool)
        text = tmpl.format(a=a, b=b)
        keywords = [a] + ([b] if "{b}" in tmpl else [])
        _add(text, keywords)

    return utts


def _piper_bin() -> str:
    return os.environ.get("ARS_PIPER_BIN", "piper")


def _voice_model(voice: str) -> Path:
    voices_dir = Path(os.environ.get("ARS_PIPER_VOICES", "models/piper"))
    return voices_dir / f"{voice}.onnx"


def synthesize(utts: list[Utterance], lang: str, out_dir: Path) -> list[Utterance]:
    """Render each utterance with a rotating set of Piper voices -> 16 kHz mono WAV."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    voices = VOICES[lang]
    for i, u in enumerate(utts):
        voice, accent = voices[i % len(voices)]
        model = _voice_model(voice)
        if not model.exists():
            raise FileNotFoundError(
                f"Piper voice model missing: {model}. Download voices to "
                f"$ARS_PIPER_VOICES (see plan/phases/phase-0 §0.6 pitfalls)."
            )
        raw = audio_dir / f"{u.utterance_id}.raw.wav"
        subprocess.run(
            [_piper_bin(), "--model", str(model), "--output_file", str(raw)],
            input=u.text.encode("utf-8"),
            check=True,
        )
        rel = f"audio/{u.utterance_id}.wav"
        u.duration_s = round(to_wav_16k_mono(raw, out_dir / rel), 3)
        raw.unlink(missing_ok=True)
        u.path = rel
        u.accent = accent
    return utts


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest(out_dir: Path, utts: list[Utterance], lang: str, seed: int) -> None:
    rows = [
        {
            "utterance_id": u.utterance_id,
            "path": u.path,
            "lang": lang,
            "text": u.text,
            "duration_s": u.duration_s,
            "source": "tts-piper",
            "accent": u.accent,
            "clean_id": None,
            "noise_subtype": None,
            "noise_level": None,
            "noise_clip_id": None,
            "snr_db_target": None,
            "snr_db_achieved": None,
            "mix_seed": None,
            "keywords": u.keywords,
        }
        for u in utts
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_dir / "manifest.parquet", index=False)
    info = {
        "dataset_id": f"tts-{lang}-v1",
        "created_at": _now(),
        "generator": "scripts.tts_corpus",
        "generator_version": GENERATOR_VERSION,
        "config_hash": "",
        "seed": seed,
        "row_count": len(rows),
        "langs": [lang],
    }
    (out_dir / "dataset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS TTS domain corpus")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--n", type=int, default=MIN_PER_LANG)
    parser.add_argument("--dry-run", action="store_true", help="no audio; write sentences.jsonl")
    parser.add_argument("--langs", nargs="+", default=["es", "en"], choices=["es", "en"])
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    menu = load_menu(Path(settings.keydetector.menu_dir) / "demo.yaml")
    confusion = load_confusion(Path(settings.paths.configs) / "confusion_seed.yaml")

    for lang in args.langs:
        utts = generate_utterances(lang, menu, confusion, args.n, settings.seed)
        out_dir = Path(settings.paths.data) / "datasets" / f"tts-{lang}-v1"
        if args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "sentences.jsonl").open("w", encoding="utf-8") as fh:
                for u in utts:
                    fh.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
            print(f"[{lang}] dry-run: {len(utts)} sentences -> {out_dir / 'sentences.jsonl'}")
            continue
        synthesize(utts, lang, out_dir)
        write_manifest(out_dir, utts, lang, settings.seed)
        print(f"[{lang}] {len(utts)} utterances synthesized -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
