"""Generate tiny WER-bearing TTS fixtures into data/fixtures/ (plan/testing §2).

A handful of real speech clips per language for tests that actually measure WER
(later phases). Generated once via Piper; gitignored. Tests skip cleanly when absent.
Seeded audio fixtures (tones/noise/silence) are generated at test time by
tests/conftest.py, not here.

    python -m scripts.make_fixtures --per-lang 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ars.config import DEFAULT_CONFIG, Settings
from scripts.tts_corpus import generate_utterances, load_confusion, load_menu, synthesize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS WER fixture generator")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--per-lang", type=int, default=20)
    parser.add_argument("--langs", nargs="+", default=["es", "en"], choices=["es", "en"])
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    menu = load_menu(Path(settings.keydetector.menu_dir) / "demo.yaml")
    confusion = load_confusion(Path(settings.paths.configs) / "confusion_seed.yaml")
    fixtures_root = Path(settings.paths.data) / "fixtures"

    for lang in args.langs:
        utts = generate_utterances(lang, menu, confusion, args.per_lang, settings.seed)
        out_dir = fixtures_root / lang
        synthesize(utts, lang, out_dir)
        rows = [
            {
                "utterance_id": u.utterance_id,
                "path": u.path,
                "lang": lang,
                "text": u.text,
                "duration_s": u.duration_s,
                "keywords": u.keywords,
            }
            for u in utts
        ]
        (out_dir / "fixtures.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{lang}] {len(utts)} fixture clips -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
