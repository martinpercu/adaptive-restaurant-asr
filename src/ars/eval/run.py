"""Evaluation runner (plan/phases/phase-1 §1.4).

Loads a dataset manifest, transcribes each utterance with a given engine, computes
per-language metrics, and writes `reports/eval/<run_id>.json` + a `metric_runs` row.
The engine is injected so tests can drive it with a fake.

    python -m ars.eval.run --manifest data/datasets/eval-clean-es-v1/manifest.parquet \
        --model-version 0.1.0
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import soundfile as sf

from ars.asr.engine import AsrEngine
from ars.asr.guard import apply_guard
from ars.config import Settings
from ars.db import connect, insert_metric_run
from ars.eval.metrics import UttRecord, compute_metrics

SR = 16000
_GUARD_FLAGS = {"repetition_truncated", "low_speech_dropped", "vad_dropped"}


def _run_id(model_version: str | None) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{ts}-{model_version or 'unknown'}"


def transcribe_dataset(
    engine: AsrEngine,
    manifest_path: str | Path,
    audio_root: str | Path | None = None,
    guard_cfg=None,
    limit: int | None = None,
    seed: int = 1337,
    use_language_hint: bool = True,
) -> dict[str, list[UttRecord]]:
    manifest_path = Path(manifest_path)
    audio_root = Path(audio_root) if audio_root else manifest_path.parent
    df = pd.read_parquet(manifest_path)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    by_lang: dict[str, list[UttRecord]] = {}
    for row in df.to_dict(orient="records"):
        lang = row["lang"]
        audio, sr = sf.read(str(audio_root / row["path"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError(f"{row['path']}: expected {SR} Hz, got {sr}")
        raw = engine.transcribe(audio, SR, language=lang if use_language_hint else None)
        guarded = apply_guard(raw, None, guard_cfg)
        kws = row.get("keywords")
        by_lang.setdefault(lang, []).append(
            UttRecord(
                ref=row["text"],
                hyp=guarded.text,
                lang=lang,
                keywords=list(kws) if kws is not None else [],
                guard_fired=any(f in _GUARD_FLAGS for f in guarded.guard_flags),
                avg_logprob=raw.avg_logprob,
            )
        )
    return by_lang


def run_eval(
    engine: AsrEngine,
    manifest_path: str | Path,
    settings: Settings,
    model_version: str | None,
    dataset_id: str | None = None,
    audio_root: str | Path | None = None,
    limit: int | None = None,
    seed: int = 1337,
    report_dir: str | Path = "reports/eval",
) -> dict:
    by_lang = transcribe_dataset(engine, manifest_path, audio_root, settings.asr.guard, limit, seed)
    metrics = {lang: compute_metrics(recs, lang) for lang, recs in by_lang.items()}
    run_id = _run_id(model_version)
    dataset_id = dataset_id or Path(manifest_path).parent.name
    report = {
        "run_id": run_id,
        "model_version": model_version,
        "dataset_id": dataset_id,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "metrics": metrics,
    }
    out = Path(report_dir) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    conn = connect(settings.paths.db)
    insert_metric_run(
        conn, run_id=run_id, model_version=model_version, dataset_id=dataset_id, metrics=metrics
    )
    conn.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS eval runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    settings = Settings.load(args.config)
    from ars.asr.engine import (
        WhisperEngine,  # noqa: PLC0415 (heavy import only when running for real)
    )
    from ars.registry import ModelRegistry  # noqa: PLC0415

    reg = ModelRegistry.load(Path(settings.paths.models) / "registry.json")
    prod = reg.get(args.model_version) if args.model_version else reg.production()
    size = settings.asr.model_size
    if prod and prod.base_model.startswith("whisper-"):
        size = prod.base_model[len("whisper-") :]
    engine = WhisperEngine(settings.asr, model=size)

    report = run_eval(
        engine,
        args.manifest,
        settings,
        model_version=args.model_version or (prod.version if prod else None),
        dataset_id=args.dataset_id,
        audio_root=args.audio_root,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
