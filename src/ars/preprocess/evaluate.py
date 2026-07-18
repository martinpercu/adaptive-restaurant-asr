"""Chain effectiveness evaluation (plan/phases/phase-3 §3.3, §3.5).

For each matrix cell (subtype × level × lang) and each candidate chain, run the phase-1
harness with that chain forced, producing an effectiveness table the policy generator
consumes. Denoisers can HURT ASR — this measures ΔWER rather than assuming. Clean rows
are evaluated too (harm guard, §3.5). Staged per (lang, chain) so DeepFilterNet's cost
fits background limits.

    python -m ars.preprocess.evaluate --stage eval --lang es --chain deepfilternet \
        --subtypes BB CA BC --per-cell 20 --matrix-out frag.parquet
    python -m ars.preprocess.evaluate --stage combine --fragments f1.parquet f2.parquet ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

from ars.asr.guard import apply_guard
from ars.config import Settings
from ars.eval.metrics import UttRecord, compute_metrics
from ars.preprocess.denoisers import get_denoiser

SR = 16000
_GUARD_FLAGS = {"repetition_truncated", "low_speech_dropped", "vad_dropped"}
CLEAN_KEY = "__clean__"


def evaluate_chain(
    engine,
    settings: Settings,
    lang: str,
    chain: str,
    subtypes: list[str] | None,
    levels: list[str],
    per_cell: int,
    seed: int,
):
    """Return effectiveness rows for one (lang, chain): per (subtype, level) + clean."""
    d = Path(settings.paths.data) / "datasets" / f"eval-matrix-{lang}-v1"
    df = pd.read_parquet(d / "manifest.parquet")
    denoiser = get_denoiser(chain)
    guard = settings.asr.guard

    groups: dict[tuple, list[UttRecord]] = {}
    lat_ms: dict[tuple, list[float]] = {}

    def _add(key, row, audio):
        t0 = time.perf_counter()
        proc = denoiser.process(audio, SR)
        lat = (time.perf_counter() - t0) * 1000.0
        raw = engine.transcribe(proc, SR, language=lang)
        guarded = apply_guard(raw, None, guard)
        kws = row.get("keywords")
        groups.setdefault(key, []).append(
            UttRecord(
                ref=row["text"],
                hyp=guarded.text,
                lang=lang,
                keywords=list(kws) if kws is not None else [],
                guard_fired=any(f in _GUARD_FLAGS for f in guarded.guard_flags),
                avg_logprob=raw.avg_logprob,
            )
        )
        lat_ms.setdefault(key, []).append(lat)

    # clean rows (harm guard)
    clean = df[df["noise_subtype"].isna()].sample(
        n=min(per_cell, (df["noise_subtype"].isna()).sum()), random_state=seed
    )
    for r in clean.to_dict("records"):
        _add((CLEAN_KEY, None), r, sf.read(str(d / r["path"]), dtype="float32")[0])

    mixed = df[df["noise_subtype"].notna()]
    if subtypes:
        mixed = mixed[mixed["noise_subtype"].isin(subtypes)]
    for (st, lv), cell in mixed.groupby(["noise_subtype", "noise_level"]):
        if lv not in levels:
            continue
        sample = cell.sample(n=min(per_cell, len(cell)), random_state=seed)
        for r in sample.to_dict("records"):
            _add((st, lv), r, sf.read(str(d / r["path"]), dtype="float32")[0])

    rows = []
    for key, recs in groups.items():
        st, lv = key
        m = compute_metrics(recs, lang)
        rows.append(
            {
                "lang": lang,
                "subtype": st,
                "level": lv,
                "chain": chain,
                "wer": m["wer"],
                "ker": m["ker"],
                "n": m["n_utts"],
                "latency_ms": round(sum(lat_ms[key]) / len(lat_ms[key]), 1),
            }
        )
    return pd.DataFrame(rows)


def combine(fragments: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(f) for f in fragments], ignore_index=True)
    # d_wer_vs_none per (lang, subtype, level)
    none = df[df["chain"] == "none"].set_index(["lang", "subtype", "level"])["wer"]
    df["d_wer_vs_none"] = df.apply(
        lambda r: r["wer"] - none.get((r["lang"], r["subtype"], r["level"]), float("nan")), axis=1
    )
    return df


def _engine(settings, model_version):
    from ars.asr.engine import WhisperEngine  # noqa: PLC0415
    from ars.registry import ModelRegistry  # noqa: PLC0415

    reg = ModelRegistry.load(Path(settings.paths.models) / "registry.json")
    prod = reg.get(model_version) if model_version else reg.production()
    size = settings.asr.model_size
    if prod and prod.base_model.startswith("whisper-"):
        size = prod.base_model[len("whisper-") :]
    return WhisperEngine(settings.asr, model=size)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS chain effectiveness eval")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stage", choices=["eval", "combine"], default="eval")
    p.add_argument("--lang", default="es")
    p.add_argument("--chain", default="none")
    p.add_argument("--subtypes", nargs="+", default=None)
    p.add_argument("--levels", nargs="+", default=["05", "10", "15"])
    p.add_argument("--per-cell", type=int, default=20)
    p.add_argument("--model-version", default=None)
    p.add_argument("--matrix-out", default=None)
    p.add_argument("--fragments", nargs="+", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    settings = Settings.load(args.config)

    if args.stage == "combine":
        df = combine(args.fragments)
        out = args.out or "reports/mitigation/effectiveness.parquet"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"effectiveness -> {out} ({len(df)} rows)")
        return 0

    engine = _engine(settings, args.model_version)
    frag = evaluate_chain(
        engine,
        settings,
        args.lang,
        args.chain,
        args.subtypes,
        args.levels,
        args.per_cell,
        settings.seed,
    )
    frag.to_parquet(args.matrix_out, index=False)
    print(f"[{args.lang}/{args.chain}] -> {args.matrix_out} ({len(frag)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
