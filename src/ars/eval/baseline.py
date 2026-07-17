"""Baseline metrics run (plan/phases/phase-1 §1.5).

Initializes the `0.1.0` production registry entry, then evaluates the candidate
(whisper-small int8) on the held-out clean eval sets (WER/CER) and the TTS domain
corpus (WER + KER over menu keywords). Writes the frozen `reports/baseline/
baseline-<lang>.json` files that all later phases compare against by run_id.

    python -m ars.eval.baseline            # both languages, real model (local heavy path)
    python -m ars.eval.baseline --limit 40 # smoke
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from ars.config import DEFAULT_CONFIG, Settings
from ars.eval.run import run_eval
from ars.registry import init_baseline


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS baseline run")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--base-model", default="whisper-small")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--langs", nargs="+", default=["es", "en"], choices=["es", "en"])
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    entry = init_baseline(Path(settings.paths.models) / "registry.json", base_model=args.base_model)
    size = (
        args.base_model[len("whisper-") :]
        if args.base_model.startswith("whisper-")
        else args.base_model
    )

    from ars.asr.engine import WhisperEngine  # noqa: PLC0415 (heavy, only when running for real)

    engine = WhisperEngine(settings.asr, model=size)
    data = Path(settings.paths.data)
    baseline_dir = Path(settings.paths.reports) / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    for lang in args.langs:
        clean_manifest = data / "datasets" / f"eval-clean-{lang}-v1" / "manifest.parquet"
        tts_manifest = data / "datasets" / f"tts-{lang}-v1" / "manifest.parquet"

        clean_report = run_eval(
            engine,
            clean_manifest,
            settings,
            model_version=entry.version,
            dataset_id=f"eval-clean-{lang}-v1",
            limit=args.limit,
        )
        tts_report = run_eval(
            engine,
            tts_manifest,
            settings,
            model_version=entry.version,
            dataset_id=f"tts-{lang}-v1",
            limit=args.limit,
        )

        baseline = {
            "lang": lang,
            "model_version": entry.version,
            "base_model": args.base_model,
            "compute_type": settings.asr.compute_type,
            "created_at": _now(),
            "eval_clean": {
                "run_id": clean_report["run_id"],
                "dataset_id": f"eval-clean-{lang}-v1",
                **clean_report["metrics"].get(lang, {}),
            },
            "tts": {
                "run_id": tts_report["run_id"],
                "dataset_id": f"tts-{lang}-v1",
                **tts_report["metrics"].get(lang, {}),
            },
        }
        out = baseline_dir / f"baseline-{lang}.json"
        out.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        clean = baseline["eval_clean"]
        print(
            f"[{lang}] clean WER={clean.get('wer')} CER={clean.get('cer')} "
            f"n={clean.get('n_utts')} | tts WER={baseline['tts'].get('wer')} "
            f"KER={baseline['tts'].get('ker')} -> {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
