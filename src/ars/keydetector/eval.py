"""KER-with/without-keydetector eval (plan/phases/phase-5 §5.6).

Builds eval-confusion-<lang> (TTS order phrases embedding confusion restaurant words,
mixed at levels 10/15) and measures KER with the keydetector OFF vs ON (replace) on the
same ASR hypotheses. Reports relative KER improvement per language.

    python -m ars.keydetector.eval --build
    python -m ars.keydetector.eval --run --model-version 0.1.0
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

from ars.asr.guard import apply_guard
from ars.config import DEFAULT_CONFIG, Settings
from ars.eval.metrics import UttRecord
from ars.keydetector.pipeline import Keydetector
from ars.noise_lab.mixer import mix
from ars.training.regression import keyword_recall
from ars.vad import SileroVad

SR = 16000
LEVEL_SNR = {"10": 0.0, "15": -5.0}


def build(settings: Settings, per_lang: int = 60, seed: int = 1337) -> list[Path]:
    data = Path(settings.paths.data)
    confusion = yaml.safe_load(
        (Path(settings.paths.configs) / "confusion_seed.yaml").read_text(encoding="utf-8")
    )
    bank = pd.read_parquet(data / "noise_bank" / "manifest.parquet")
    bank_eval = bank[bank["split"] == "eval"]
    vad = SileroVad(settings.vad)
    out_dirs = []
    for lang in ("es", "en"):
        targets = {c["target"] for c in confusion[lang]}
        tts = pd.read_parquet(data / "datasets" / f"tts-{lang}-v1" / "manifest.parquet")
        # utterances whose keywords include a confusion target
        mask = tts["keywords"].map(
            lambda ks, t=targets: bool(set(ks) & t) if ks is not None else False
        )
        pool = tts[mask].head(per_lang).to_dict("records")
        out = data / "datasets" / f"eval-confusion-{lang}-v1"
        (out / "audio").mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed + (0 if lang == "es" else 1))
        clips = bank_eval.reset_index(drop=True)
        rows = []
        for i, r in enumerate(pool):
            audio, _ = sf.read(
                str(data / "datasets" / f"tts-{lang}-v1" / r["path"]), dtype="float32"
            )
            vr = vad.detect(audio, SR)
            s_rms = vr.speech_rms if vr.speech_rms > 0 else float((audio**2).mean() ** 0.5)
            level = rng.choice(list(LEVEL_SNR))
            clip = clips.iloc[rng.randrange(len(clips))]
            noise, _ = sf.read(str(data / "noise_bank" / clip["path"]), dtype="float32")
            res = mix(audio, noise, LEVEL_SNR[level], s_rms, rng.randrange(1 << 30))
            uid = f"ec-{lang}-{i:03d}"
            sf.write(str(out / "audio" / f"{uid}.wav"), res.mixed.clip(-1, 1), SR, subtype="PCM_16")
            rows.append(
                {
                    "utterance_id": uid,
                    "path": f"audio/{uid}.wav",
                    "lang": lang,
                    "text": r["text"],
                    "duration_s": round(len(res.mixed) / SR, 3),
                    "source": "eval-confusion",
                    "noise_subtype": clip["subtype"],
                    "noise_level": level,
                    "keywords": list(r["keywords"]),
                }
            )
        pd.DataFrame(rows).to_parquet(out / "manifest.parquet", index=False)
        (out / "dataset.json").write_text(
            json.dumps({"dataset_id": out.name, "row_count": len(rows)})
        )
        print(f"{out.name}: {len(rows)} utts")
        out_dirs.append(out)
    return out_dirs


def run(settings: Settings, model_version: str | None) -> dict:
    from ars.asr.engine import WhisperEngine  # noqa: PLC0415
    from ars.registry import ModelRegistry  # noqa: PLC0415

    reg = ModelRegistry.load(Path(settings.paths.models) / "registry.json")
    prod = reg.get(model_version) if model_version else reg.production()
    size = settings.asr.model_size
    if prod and prod.base_model.startswith("whisper-"):
        size = prod.base_model[len("whisper-") :]
    engine = WhisperEngine(settings.asr, model=size)
    kd = Keydetector.from_settings(settings)
    kd.mode = "replace"

    data = Path(settings.paths.data)
    report: dict = {}
    for lang in ("es", "en"):
        d = data / "datasets" / f"eval-confusion-{lang}-v1"
        if not (d / "manifest.parquet").exists():
            continue
        off, on = [], []
        for r in pd.read_parquet(d / "manifest.parquet").to_dict("records"):
            audio = sf.read(str(d / r["path"]), dtype="float32")[0]
            raw = engine.transcribe(audio, SR, language=lang)
            hyp = apply_guard(raw, None, settings.asr.guard).text
            kws = list(r["keywords"])
            off.append(UttRecord(ref=r["text"], hyp=hyp, lang=lang, keywords=kws))
            on.append(
                UttRecord(ref=r["text"], hyp=kd.correct(hyp, lang)[0], lang=lang, keywords=kws)
            )
        recall_off, _ = keyword_recall(off, lang)
        recall_on, _ = keyword_recall(on, lang)
        ker_off, ker_on = 1 - recall_off, 1 - recall_on
        impr = (ker_off - ker_on) / max(ker_off, 1e-6)
        report[lang] = {
            "ker_off": round(ker_off, 4),
            "ker_on": round(ker_on, 4),
            "ker_rel_improvement": round(impr, 4),
            "n": len(off),
        }
    out = Path(settings.paths.reports) / "keydetector"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ker.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS keydetector KER eval")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--build", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--per-lang", type=int, default=60)
    p.add_argument("--model-version", default=None)
    args = p.parse_args(argv)
    settings = Settings.load(args.config)
    if args.build:
        build(settings, args.per_lang)
    if args.run:
        run(settings, args.model_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
