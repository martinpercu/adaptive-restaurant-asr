"""Policy generation (plan/phases/phase-3 §3.3). Reads the effectiveness table, writes
the generated `configs/mitigation_policy.yaml` and `EFFECTIVENESS.md`.

    python -m ars.preprocess.gen_policy --effectiveness reports/mitigation/effectiveness.parquet
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ars.config import DEFAULT_CONFIG, Settings
from ars.preprocess.evaluate import CLEAN_KEY
from ars.preprocess.policy import generate_policy, write_policy_yaml

MIN_REL = 0.03


def _rel(none_wer: float, chain_wer: float) -> float:
    return (none_wer - chain_wer) / max(none_wer, 1e-6)


def effectiveness_md(df: pd.DataFrame, policy: dict[str, str], run_id: str) -> str:
    lines = [f"# Mitigation effectiveness — {run_id}", ""]
    subtypes = sorted(s for s in df["subtype"].unique() if s != CLEAN_KEY)
    helped, residual = [], []
    lines.append("| subtype | chain chosen | mean ΔWER rel (worst lang) | mean latency ms |")
    lines.append("|---------|--------------|----------------------------|-----------------|")
    for st in subtypes:
        chain = policy.get(st, "none")
        sdf = df[df["subtype"] == st]
        if chain == "none":
            residual.append(st)
            rel_txt, lat_txt = "—", "—"
        else:
            rels, lats = [], []
            for lang in sorted(sdf["lang"].unique()):
                nw = sdf[(sdf["lang"] == lang) & (sdf["chain"] == "none")]["wer"].mean()
                cw = sdf[(sdf["lang"] == lang) & (sdf["chain"] == chain)]["wer"].mean()
                rels.append(_rel(nw, cw))
                lats.append(
                    sdf[(sdf["lang"] == lang) & (sdf["chain"] == chain)]["latency_ms"].mean()
                )
            helped.append(st)
            rel_txt, lat_txt = f"{min(rels):.1%}", f"{max(lats):.0f}"
        lines.append(f"| {st} | {chain} | {rel_txt} | {lat_txt} |")

    lines += [
        "",
        "## Residual damage (no chain helped → phase-4 priority)",
        ("- " + ", ".join(residual)) if residual else "- none",
    ]

    # clean-harm summary (§3.5)
    clean = df[df["subtype"] == CLEAN_KEY]
    if not clean.empty:
        lines += ["", "## Clean-audio harm guard"]
        for lang in sorted(clean["lang"].unique()):
            nw = clean[(clean["lang"] == lang) & (clean["chain"] == "none")]["wer"].mean()
            for chain in sorted(c for c in clean["chain"].unique() if c != "none"):
                cw = clean[(clean["lang"] == lang) & (clean["chain"] == chain)]["wer"].mean()
                harm = _rel(nw, cw)  # negative = worse on clean
                flag = "  ⚠ harms clean" if harm < -0.01 else ""
                lines.append(f"- {lang}/{chain}: clean WER {nw:.3f} → {cw:.3f} ({harm:+.1%}){flag}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate mitigation policy")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--effectiveness", default="reports/mitigation/effectiveness.parquet")
    p.add_argument("--run-id", default=None)
    args = p.parse_args(argv)
    settings = Settings.load(args.config)

    df = pd.read_parquet(args.effectiveness)
    eval_df = df[df["subtype"] != CLEAN_KEY]  # policy over real subtypes only
    policy = generate_policy(eval_df, latency_budget_ms=400.0, min_rel_improvement=MIN_REL)
    # every bank subtype must be mapped (default none for subtypes not measured)
    bank = pd.read_parquet(Path(settings.paths.data) / "noise_bank" / "manifest.parquet")
    for st in bank["subtype"].unique():
        policy.setdefault(st, "none")

    run_id = args.run_id or f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    write_policy_yaml(
        policy,
        settings.preprocess.policy_path,
        meta={"run_id": run_id, "effectiveness": args.effectiveness},
    )

    report_dir = Path(settings.paths.reports) / "mitigation" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(report_dir / "effectiveness.parquet", index=False)
    (report_dir / "EFFECTIVENESS.md").write_text(effectiveness_md(df, policy, run_id), "utf-8")

    print(f"policy -> {settings.preprocess.policy_path}")
    for st, chain in sorted(policy.items()):
        print(f"  {st}: {chain}")
    print(f"report -> {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
