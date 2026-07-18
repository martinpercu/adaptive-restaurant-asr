"""LoRA fine-tuning (plan/phases/phase-4 §4.3).

PEFT/LoRA over a multilingual Whisper base; only adapter weights train. Each sample uses
its own language token (a fixed token on the wrong language silently trains garbage —
phase-4 pitfall). Device-agnostic; the CPU path is a smoke-scale run, the real recipe
runs on GPU (see docker/Dockerfile.gpu). Manual loop for control at smoke scale.

    python -m ars.training.train_lora --base openai/whisper-small \
        --dataset data/datasets/train-noisy-xx-v1 --epochs 3
    python -m ars.training.train_lora --base openai/whisper-tiny --steps 20 --smoke   # CPU
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
LORA_TARGETS = ["q_proj", "v_proj"]


def _iso_week() -> str:
    now = datetime.now(UTC)
    return f"lora-{now.isocalendar().year}w{now.isocalendar().week:02d}"


def _load_audio(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz")
    return a.mean(axis=1).astype(np.float32)


def build_peft_model(base_model: str, r=32, alpha=64, dropout=0.05):
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import WhisperForConditionalGeneration  # noqa: PLC0415

    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias="none"
    )
    return get_peft_model(model, cfg)


def _encode_labels(processor, text: str, lang: str):
    processor.tokenizer.set_prefix_tokens(language=lang, task="transcribe")
    return processor.tokenizer(text).input_ids


def train_lora(
    base_model: str,
    dataset_dir: str | Path,
    out_root: str | Path = "models/adapters",
    epochs: int = 3,
    steps: int | None = None,
    batch_size: int = 8,
    lr: float = 5e-4,
    seed: int = 1337,
    r: int = 32,
    alpha: int = 64,
    dataset_id: str | None = None,
    ndi_run: str | None = None,
) -> dict:
    import torch  # noqa: PLC0415
    from transformers import WhisperProcessor  # noqa: PLC0415

    torch.manual_seed(seed)
    dataset_dir = Path(dataset_dir)
    df = pd.read_parquet(dataset_dir / "manifest.parquet")
    processor = WhisperProcessor.from_pretrained(base_model)
    model = build_peft_model(base_model, r=r, alpha=alpha)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = np.random.default_rng(seed)
    rows = df.to_dict("records")
    total_steps = steps if steps is not None else max(1, (len(rows) * epochs) // batch_size)
    warmup = max(1, int(0.1 * total_steps))

    losses = []
    for step in range(total_steps):
        idx = rng.integers(0, len(rows), size=batch_size)
        feats, labels = [], []
        for i in idx:
            r_ = rows[int(i)]
            audio = _load_audio(dataset_dir / r_["path"])
            f = processor.feature_extractor(audio, sampling_rate=SR, return_tensors="pt")
            feats.append(f.input_features[0])
            labels.append(_encode_labels(processor, r_["text"], r_["lang"]))
        input_features = torch.stack(feats)
        maxlen = max(len(x) for x in labels)
        pad = processor.tokenizer.pad_token_id or -100
        label_ids = torch.full((len(labels), maxlen), -100, dtype=torch.long)
        for j, lab in enumerate(labels):
            label_ids[j, : len(lab)] = torch.tensor(lab)
        # cosine lr with warmup
        frac = step / max(total_steps, 1)
        scale = (step + 1) / warmup if step < warmup else 0.5 * (1 + np.cos(np.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr * scale
        opt.zero_grad()
        out = model(input_features=input_features, labels=label_ids)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
        if step % max(1, total_steps // 5) == 0:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  step {step + 1}/{total_steps} loss={losses[-1]:.4f} lr={cur_lr:.2e}")
        _ = pad

    adapter_name = _iso_week()
    out_dir = Path(out_root) / adapter_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    meta = {
        "adapter": adapter_name,
        "base_model": base_model,
        "dataset_id": dataset_id or dataset_dir.name,
        "ndi_run": ndi_run,
        "seed": seed,
        "lora": {"r": r, "alpha": alpha, "targets": LORA_TARGETS},
        "steps": total_steps,
        "batch_size": batch_size,
        "lr": lr,
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "loss_curve": [round(x, 4) for x in losses],
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"adapter {adapter_name}: loss {meta['loss_first']} -> {meta['loss_last']} -> {out_dir}")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS LoRA training")
    p.add_argument("--base", default="openai/whisper-small")
    p.add_argument("--dataset", default="data/datasets/train-noisy-xx-v1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run (few steps)")
    args = p.parse_args(argv)
    steps = args.steps if args.steps is not None else (20 if args.smoke else None)
    bs = 2 if args.smoke else args.batch_size
    train_lora(
        args.base,
        args.dataset,
        epochs=args.epochs,
        steps=steps,
        batch_size=bs,
        lr=args.lr,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
