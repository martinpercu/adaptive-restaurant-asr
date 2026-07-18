"""Static dashboard generator (plan/phases/phase-7 §7.1).

Self-contained HTML (no server, no external assets) summarizing: model/gate history, WER
proxy trend, per-store confidence, NDI evolution, keydetector rule activity, latency
percentiles. Every section handles the empty state (fresh install) without crashing.

    python -m ars.eval.dashboard --out reports/dashboard
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ars.config import DEFAULT_CONFIG, Settings
from ars.registry import ModelRegistry


def _table(headers: list[str], rows: list[list], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{empty}</p>'
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _section(title: str, html: str) -> str:
    return f"<section><h2>{title}</h2>{html}</section>"


def _models(reg: ModelRegistry) -> str:
    rows = [
        [e.version, e.stage, e.base_model, e.adapter or "—", e.promoted_at or "—"]
        for e in reg.entries
    ]
    return _section(
        "Model registry",
        _table(["version", "stage", "base", "adapter", "promoted"], rows, "no models registered"),
    )


def _metric_runs(conn: sqlite3.Connection) -> str:
    rows = []
    try:
        for run_id, mv, ds, mj, ca in conn.execute(
            "SELECT run_id, model_version, dataset_id, metrics_json, created_at "
            "FROM metric_runs ORDER BY created_at DESC LIMIT 20"
        ):
            m = json.loads(mj)
            wer = next((v.get("wer") for v in m.values() if isinstance(v, dict)), "—")
            rows.append([run_id, mv, ds, wer, ca])
    except sqlite3.OperationalError:
        pass
    return _section(
        "Eval runs (WER trend)",
        _table(["run", "model", "dataset", "wer", "when"], rows, "no eval runs yet"),
    )


def _ndi(reports: Path) -> str:
    runs = (
        sorted((reports / "sensitivity").glob("run-*"))
        if (reports / "sensitivity").exists()
        else []
    )
    rows = []
    for run in runs[-5:]:
        try:
            ndi = json.loads((run / "ndi.json").read_text())
            top = ", ".join(e["subtype"] for e in ndi["ranking"][:3])
            rows.append([ndi["run_id"], top])
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue
    return _section(
        "NDI evolution (top-3 damaging)",
        _table(["sensitivity run", "top-3 subtypes"], rows, "no sensitivity runs yet"),
    )


def _latency(conn: sqlite3.Connection, telemetry_dir: Path) -> str:
    lats = []
    for f in sorted(telemetry_dir.glob("*.jsonl")) if telemetry_dir.exists() else []:
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                lats.append(json.loads(line)["latency_ms"]["total"])
            except (json.JSONDecodeError, KeyError):
                continue
    if not lats:
        return _section("Latency", '<p class="empty">no telemetry yet</p>')
    lats.sort()
    n = len(lats)
    p = lambda q: lats[min(n - 1, int(q * n))]  # noqa: E731
    rows = [["p50", f"{p(0.5):.0f} ms"], ["p95", f"{p(0.95):.0f} ms"], ["p99", f"{p(0.99):.0f} ms"]]
    return _section("Latency (end-to-end)", _table(["pct", "ms"], rows, ""))


def generate(settings: Settings, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = ModelRegistry.load(Path(settings.paths.models) / "registry.json")
    reports = Path(settings.paths.reports)
    telemetry = Path(settings.ingest.telemetry_dir)
    conn = None
    if Path(settings.paths.db).exists():
        conn = sqlite3.connect(settings.paths.db)

    sections = [
        _models(reg),
        _metric_runs(conn)
        if conn
        else _section("Eval runs (WER trend)", '<p class="empty">no db</p>'),
        _ndi(reports),
        _latency(conn, telemetry) if conn else _section("Latency", '<p class="empty">no db</p>'),
    ]
    if conn:
        conn.close()
    html = _PAGE.format(body="\n".join(sections))
    index = out_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    print(f"dashboard -> {index}")
    return index


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ARS Dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}}
h1{{border-bottom:2px solid #333}} section{{margin:1.5rem 0}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:.4rem;text-align:left}}
th{{background:#f0f0f0}} .empty{{color:#999;font-style:italic}}
</style></head><body><h1>ARS — Adaptive Restaurant Speech</h1>{body}</body></html>"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ARS dashboard")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out", default="reports/dashboard")
    args = p.parse_args(argv)
    generate(Settings.load(args.config), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
