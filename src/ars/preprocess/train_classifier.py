"""Train the noise subtype classifier (plan/phases/phase-3 §3.1).

Training data is free: raw noise-bank windows (label = subtype) + speech×noise mixtures
built with the phase-2 mixer (label = subtype) + CLEAN windows from clean speech. Only
**train-split** noise clips train; **eval-split** clips (held-out source recordings)
evaluate — never held-out windows of a training recording.

    python -m ars.preprocess.train_classifier --config configs/default.yaml --epochs 8
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from ars.config import DEFAULT_CONFIG, Settings
from ars.noise_lab.mixer import mix
from ars.noise_lab.taxonomy import load_taxonomy
from ars.preprocess.classifier import (
    CLEAN,
    Classifier,
    NoisePrediction,
    build_model,
    classes_from_subtypes,
    log_mel_windows,
)
from ars.vad import SileroVad

SR = 16000
LEVEL_SNR = {"05": 10.0, "10": 0.0, "15": -5.0}


@dataclass
class Example:
    audio: np.ndarray
    label: str
    group: str  # source recording id (or "clean") — split unit


def _load(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return a.mean(axis=1).astype(np.float32)


def _build_examples(
    settings: Settings, split: str, clean_ids: list[str], seed: int
) -> list[Example]:
    data = Path(settings.paths.data)
    bank = pd.read_parquet(data / "noise_bank" / "manifest.parquet")
    bank = bank[bank["split"] == split]
    rng = random.Random(seed)
    vad = SileroVad(settings.vad)
    examples: list[Example] = []

    # clean speech -> CLEAN + a pool for mixtures
    clean_pool: list[tuple[str, np.ndarray, float]] = []
    tts = pd.read_parquet(data / "datasets" / "tts-es-v1" / "manifest.parquet")
    tts2 = pd.read_parquet(data / "datasets" / "tts-en-v1" / "manifest.parquet")
    for lang, df in (("es", tts), ("en", tts2)):
        rows = df[df["utterance_id"].isin(clean_ids)].to_dict("records")
        for r in rows:
            audio = _load(data / "datasets" / f"tts-{lang}-v1" / r["path"])
            vr = vad.detect(audio, SR)
            s_rms = vr.speech_rms if vr.speech_rms > 0 else float(np.sqrt(np.mean(audio**2)))
            clean_pool.append((r["utterance_id"], audio, s_rms))
            examples.append(Example(audio, CLEAN, "clean"))

    # raw noise windows + mixtures per subtype
    for st, g in bank.groupby("subtype"):
        clips = g.to_dict("records")
        for clip in clips:
            noise = _load(data / "noise_bank" / clip["path"])
            examples.append(Example(noise, st, clip["recording_id"]))  # raw noise
            # a few mixtures with random clean utts / levels
            for _ in range(2):
                cid, caudio, s_rms = rng.choice(clean_pool)
                level = rng.choice(list(LEVEL_SNR))
                res = mix(caudio, noise, LEVEL_SNR[level], s_rms, seed=rng.randrange(1 << 30))
                examples.append(Example(res.mixed, st, clip["recording_id"]))
    rng.shuffle(examples)
    return examples


def _windows(examples: list[Example], classes: list[str]):
    idx = {c: i for i, c in enumerate(classes)}
    xs, ys = [], []
    for ex in examples:
        feats = log_mel_windows(ex.audio, SR)
        xs.append(feats)
        ys.extend([idx[ex.label]] * len(feats))
    return np.concatenate(xs), np.array(ys, dtype=np.int64)


def _family(subtype: str) -> str:
    return subtype[0] if subtype != CLEAN else CLEAN


def _macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(model, classes: list[str], examples: list[Example], min_conf: float) -> dict:
    clf = Classifier(model, classes, min_conf)
    y_true, y_pred = [], []
    clean_tp = clean_fp = 0
    for ex in examples:
        pred: NoisePrediction = clf.predict(ex.audio, SR)
        pred_label = pred.subtype or CLEAN
        y_true.append(ex.label)
        y_pred.append(pred_label)
        if pred_label == CLEAN:
            if ex.label == CLEAN:
                clean_tp += 1
            else:
                clean_fp += 1
    clean_prec = clean_tp / (clean_tp + clean_fp) if (clean_tp + clean_fp) else 1.0
    return {
        "subtype_macro_f1": round(_macro_f1(y_true, y_pred, classes), 4),
        "family_macro_f1": round(
            _macro_f1(
                [_family(y) for y in y_true],
                [_family(p) for p in y_pred],
                sorted({_family(c) for c in classes}),
            ),
            4,
        ),
        "clean_precision": round(clean_prec, 4),
        "n_eval": len(examples),
    }


def main(argv: list[str] | None = None) -> int:
    import torch  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Train ARS noise classifier")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--clean-per-lang", type=int, default=40)
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)
    torch.manual_seed(settings.seed)

    tax = load_taxonomy(settings.paths.noise_taxonomy)
    bank = pd.read_parquet(Path(settings.paths.data) / "noise_bank" / "manifest.parquet")
    classes = classes_from_subtypes(sorted(bank["subtype"].unique()))

    # clean utterance ids: disjoint train/eval samples per lang
    rng = random.Random(settings.seed)
    clean_ids_train, clean_ids_eval = [], []
    for lang in ("es", "en"):
        df = pd.read_parquet(
            Path(settings.paths.data) / "datasets" / f"tts-{lang}-v1" / "manifest.parquet"
        )
        ids = df["utterance_id"].tolist()
        rng.shuffle(ids)
        clean_ids_train += ids[: args.clean_per_lang]
        clean_ids_eval += ids[args.clean_per_lang : args.clean_per_lang + args.clean_per_lang // 2]

    train_ex = _build_examples(settings, "train", clean_ids_train, settings.seed)
    eval_ex = _build_examples(settings, "eval", clean_ids_eval, settings.seed + 1)

    X, y = _windows(train_ex, classes)
    model = build_model(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    model.train()
    n = len(Xt)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        print(f"  epoch {epoch + 1}/{args.epochs} loss={total / n:.4f}")

    report = evaluate(model, classes, eval_ex, settings.preprocess.min_confidence)
    report.update(
        {
            "version": args.version,
            "classes": classes,
            "families": sorted({_family(s) for s in classes if s != CLEAN}),
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_train_windows": int(n),
            "targets": {"subtype_macro_f1": 0.80, "family_macro_f1": 0.90, "clean_precision": 0.95},
        }
    )

    out = Path(settings.paths.models) / "noise_classifier" / args.version
    out.mkdir(parents=True, exist_ok=True)
    Classifier(model, classes, settings.preprocess.min_confidence).save(str(out / "model.pt"))
    (out / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = Path(settings.paths.models) / "noise_classifier" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    Classifier(model, classes, settings.preprocess.min_confidence).save(str(latest / "model.pt"))
    (latest / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _ = tax
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
