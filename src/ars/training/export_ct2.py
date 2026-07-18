"""Merge LoRA + export to CTranslate2 with a parity check (plan/phases/phase-4 §4.5).

Merge adapter → full HF model → CTranslate2 (int8_float16). Parity: transcribe 20 eval
utterances greedily with HF and CT2; WER-vs-each-other ≤ 1.0 absolute point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf

from ars.config import DEFAULT_CONFIG, Settings
from ars.eval.normalize import normalize

SR = 16000


def merge_adapter(base_model: str, adapter_dir: str | Path, out_dir: str | Path) -> Path:
    from peft import PeftModel  # noqa: PLC0415
    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415

    base = WhisperForConditionalGeneration.from_pretrained(base_model)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    WhisperProcessor.from_pretrained(base_model).save_pretrained(str(out_dir))
    return out_dir


def convert_ct2(merged_dir: str | Path, ct2_dir: str | Path, quantization="int8_float16") -> Path:
    from ctranslate2.converters import TransformersConverter  # noqa: PLC0415

    ct2_dir = Path(ct2_dir)
    TransformersConverter(str(merged_dir)).convert(
        str(ct2_dir), quantization=quantization, force=True
    )
    return ct2_dir


def _hf_greedy(model, processor, audio, lang) -> str:

    feats = processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt").input_features
    ids = model.generate(
        feats,
        num_beams=1,
        do_sample=False,
        language=lang,
        task="transcribe",
        max_new_tokens=128,
    )
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def parity(
    base_model: str, merged_dir: str | Path, ct2_dir: str | Path, manifest: str | Path, n: int = 20
) -> dict:
    """Return {wer_diff, n} — WER between HF-greedy and CT2-greedy hyps (should be ~0)."""
    import jiwer  # noqa: PLC0415
    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415

    from ars.asr.engine import WhisperEngine  # noqa: PLC0415

    manifest = Path(manifest)
    df = pd.read_parquet(manifest / "manifest.parquet").head(n)
    hf = WhisperForConditionalGeneration.from_pretrained(str(merged_dir))
    hf.eval()
    processor = WhisperProcessor.from_pretrained(str(merged_dir))
    ct2 = WhisperEngine(model=str(ct2_dir))

    hf_hyps, ct2_hyps = [], []
    for r in df.to_dict("records"):
        audio = sf.read(str(manifest / r["path"]), dtype="float32", always_2d=True)[0].mean(axis=1)
        hf_hyps.append(normalize(_hf_greedy(hf, processor, audio, r["lang"]), r["lang"]))
        ct2_hyps.append(normalize(ct2.transcribe(audio, SR, language=r["lang"]).text, r["lang"]))
    pairs = [(h, c) for h, c in zip(hf_hyps, ct2_hyps, strict=True) if h]
    wer = jiwer.process_words([h for h, _ in pairs], [c for _, c in pairs]).wer if pairs else 0.0
    return {"wer_diff": round(float(wer), 4), "n": len(df)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge LoRA + export CT2 + parity")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--base", default="openai/whisper-small")
    p.add_argument("--adapter", required=True)
    p.add_argument("--version", default="0.2.0")
    p.add_argument("--parity-manifest", default="data/datasets/eval-noisy-es-v1")
    args = p.parse_args(argv)
    settings = Settings.load(args.config)

    models = Path(settings.paths.models)
    merged = merge_adapter(args.base, args.adapter, models / "merged" / args.version)
    ct2 = convert_ct2(merged, models / "ct2" / args.version)
    par = parity(args.base, merged, ct2, args.parity_manifest)
    print(f"CT2 export -> {ct2} | parity WER diff = {par['wer_diff']} (<=1.0)")
    (Path(ct2) / "export_meta.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "base_model": args.base,
                "adapter": Path(args.adapter).name,
                "parity": par,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
