"""Sensitivity runner + NDI report (plan/phases/phase-2 §2.4, plan/03-data-spec.md §3).

Evaluates the eval-matrix corpus cell by cell with the phase-1 harness (raw model,
preprocessing disabled), aggregates per (subtype, level), computes the Noise Damage
Index, and emits matrix.parquet + ndi.json + heatmap-<lang>.png + ANALYSIS.md.

    python -m ars.noise_lab.sensitivity --model-version 0.1.0        # both langs
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
from ars.eval.metrics import UttRecord, compute_metrics
from ars.noise_lab.ndi import compute_ndi, top_subtypes

SR = 16000
_GUARD_FLAGS = {"repetition_truncated", "low_speech_dropped", "vad_dropped"}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_id(model_version: str) -> str:
    return f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{model_version}"


def evaluate_matrix(engine: AsrEngine, manifest_path: Path, guard_cfg, limit=None, seed=1337):
    """Transcribe every matrix row; return records keyed by (lang, subtype, level)."""
    root = manifest_path.parent
    df = pd.read_parquet(manifest_path)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)
    groups: dict[tuple, list[UttRecord]] = {}
    for r in df.to_dict(orient="records"):
        audio, sr = sf.read(str(root / r["path"]), dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{r['path']}: expected {SR} Hz")
        raw = engine.transcribe(audio.mean(axis=1), SR, language=r["lang"])
        guarded = apply_guard(raw, None, guard_cfg)
        kws = r.get("keywords")
        key = (r["lang"], r["noise_subtype"], r["noise_level"])
        groups.setdefault(key, []).append(
            UttRecord(
                ref=r["text"],
                hyp=guarded.text,
                lang=r["lang"],
                keywords=list(kws) if kws is not None else [],
                guard_fired=any(f in _GUARD_FLAGS for f in guarded.guard_flags),
                avg_logprob=raw.avg_logprob,
            )
        )
    return groups


def build_matrix(groups: dict[tuple, list[UttRecord]], model_version: str) -> pd.DataFrame:
    rows = []
    for (lang, subtype, level), recs in groups.items():
        m = compute_metrics(recs, lang)
        rows.append(
            {
                "model_version": model_version,
                "lang": lang,
                "noise_subtype": subtype,
                "noise_level": level,
                **m,
            }
        )
    return pd.DataFrame(rows)


def _heatmap(matrix: pd.DataFrame, lang: str, levels: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = matrix[(matrix["lang"] == lang) & (matrix["noise_subtype"].notna())]
    subtypes = sorted(sub["noise_subtype"].unique())
    grid = [[_cell_wer(sub, st, lv) for lv in levels] for st in subtypes]
    fig, ax = plt.subplots(figsize=(1.4 * len(levels) + 1, 0.5 * len(subtypes) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(subtypes)), subtypes)
    ax.set_xlabel("level"), ax.set_ylabel("subtype")
    ax.set_title(f"WER by noise cell — {lang}")
    for i, _st in enumerate(subtypes):
        for j, _ in enumerate(levels):
            v = grid[i][j]
            ax.text(
                j,
                i,
                "-" if v is None else f"{v:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="WER")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _cell_wer(sub: pd.DataFrame, subtype: str, level: str):
    row = sub[(sub["noise_subtype"] == subtype) & (sub["noise_level"] == level)]
    if row.empty or pd.isna(row["wer"].iloc[0]):
        return None
    return float(row["wer"].iloc[0])


def _monotonicity_warnings(matrix: pd.DataFrame, levels: list[str]) -> list[str]:
    warns = []
    for lang in sorted(matrix["lang"].unique()):
        sub = matrix[(matrix["lang"] == lang) & (matrix["noise_subtype"].notna())]
        for st in sorted(sub["noise_subtype"].unique()):
            wers = [_cell_wer(sub, st, lv) for lv in levels]
            if all(w is not None for w in wers) and not (wers[0] <= wers[1] <= wers[2]):
                warns.append(f"{lang}/{st}: WER not monotonic across levels {wers}")
    return warns


def _analysis_md(ndi: dict, matrix: pd.DataFrame, warns: list[str], run_id: str) -> str:
    lines = [f"# Sensitivity analysis — {run_id}", ""]
    for lang in sorted(matrix["lang"].unique()):
        top = top_subtypes(ndi, lang, 3)
        lines.append(f"## {lang}")
        lines.append(f"- Baseline clean WER: {ndi['baseline'].get(lang, {}).get('wer')}")
        lines.append(f"- **Top-3 damaging subtypes (NDI):** {', '.join(top)}")
        lines.append("")
    lines += [
        "## Notes (fill in)",
        "- Does babble (AC/CA) outrank stationary (AB) as theory predicts?",
        "- Ceiling effects at level 15?",
        "- **Drive-thru prior:** family B + BC (car-cabin) are expected to dominate in "
        "production even if dining-room subtypes score high on public-proxy audio.",
        "- Phase-3 mitigation targets = the top-3 above; phase-4 sampling weights ∝ NDI.",
        "",
        "## Monotonicity warnings",
    ]
    lines += [f"- {w}" for w in warns] or ["- none"]
    return "\n".join(lines) + "\n"


def _weights(settings: Settings) -> dict:
    w = settings.eval.ndi_weights.model_dump()
    return {"d_wer": w["d_wer"], "d_ker": w["d_ker"], "hallucination": w["hallucination"]}


def transcribe_lang(
    engine: AsrEngine, settings: Settings, lang: str, model_version: str, limit=None
) -> pd.DataFrame:
    """Heavy step for one language: transcribe its eval-matrix -> per-cell matrix rows."""
    manifest = (
        Path(settings.paths.data) / "datasets" / f"eval-matrix-{lang}-v1" / "manifest.parquet"
    )
    groups = evaluate_matrix(engine, manifest, settings.asr.guard, limit)
    return build_matrix(groups, model_version)


def write_report(
    matrix: pd.DataFrame, settings: Settings, model_version: str, levels=("05", "10", "15")
) -> Path:
    """Light step: NDI + heatmaps + ANALYSIS from an already-computed matrix."""
    ndi_body = compute_ndi(matrix, _weights(settings), tuple(levels))
    run_id = _run_id(model_version)
    out_dir = Path(settings.paths.reports) / "sensitivity" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_dir / "matrix.parquet", index=False)

    ndi_json = {"run_id": run_id, "model_version": model_version, **ndi_body}
    (out_dir / "ndi.json").write_text(json.dumps(ndi_json, indent=2, ensure_ascii=False), "utf-8")

    warns = _monotonicity_warnings(matrix, list(levels))
    for lang in sorted(matrix["lang"].unique()):
        _heatmap(matrix, lang, list(levels), out_dir / f"heatmap-{lang}.png")
    (out_dir / "ANALYSIS.md").write_text(_analysis_md(ndi_json, matrix, warns, run_id), "utf-8")

    print(f"\nsensitivity report -> {out_dir}")
    for lang in sorted(matrix["lang"].unique()):
        print(f"  [{lang}] top-3 damaging subtypes: {top_subtypes(ndi_json, lang, 3)}")
    for w in warns:
        print(f"  WARNING (monotonicity): {w}")
    return out_dir


def run_sensitivity(
    engine: AsrEngine,
    settings: Settings,
    model_version: str,
    langs=("es", "en"),
    levels=("05", "10", "15"),
    limit=None,
) -> Path:
    """All-in-one (small runs/tests). For large corpora use the staged CLI."""
    frames = []
    for lang in langs:
        manifest = (
            Path(settings.paths.data) / "datasets" / f"eval-matrix-{lang}-v1" / "manifest.parquet"
        )
        if not manifest.exists():
            print(f"  skip {lang}: {manifest} missing")
            continue
        frames.append(transcribe_lang(engine, settings, lang, model_version, limit))
    return write_report(pd.concat(frames, ignore_index=True), settings, model_version, levels)


def _engine_for(settings: Settings, model_version: str | None):
    from ars.asr.engine import WhisperEngine  # noqa: PLC0415
    from ars.registry import ModelRegistry  # noqa: PLC0415

    reg = ModelRegistry.load(Path(settings.paths.models) / "registry.json")
    prod = reg.get(model_version) if model_version else reg.production()
    size = settings.asr.model_size
    if prod and prod.base_model.startswith("whisper-"):
        size = prod.base_model[len("whisper-") :]
    version = model_version or (prod.version if prod else "unknown")
    return WhisperEngine(settings.asr, model=size), version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARS noise sensitivity + NDI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--langs", nargs="+", default=["es", "en"], choices=["es", "en"])
    parser.add_argument("--limit", type=int, default=None)
    # Staged execution keeps each per-language transcription within time limits:
    parser.add_argument("--stage", choices=["all", "transcribe", "report"], default="all")
    parser.add_argument("--matrix-out", default=None, help="transcribe stage: write matrix here")
    parser.add_argument("--matrices", nargs="+", default=None, help="report stage: matrix inputs")
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)

    if args.stage == "report":
        matrix = pd.concat([pd.read_parquet(m) for m in args.matrices], ignore_index=True)
        mv = matrix["model_version"].iloc[0] if "model_version" in matrix else args.model_version
        write_report(matrix, settings, mv)
        return 0

    engine, version = _engine_for(settings, args.model_version)
    if args.stage == "transcribe":
        matrix = transcribe_lang(engine, settings, args.langs[0], version, args.limit)
        matrix.to_parquet(args.matrix_out, index=False)
        print(f"[{args.langs[0]}] matrix -> {args.matrix_out} ({len(matrix)} cells)")
        return 0

    run_sensitivity(engine, settings, version, langs=tuple(args.langs), limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
