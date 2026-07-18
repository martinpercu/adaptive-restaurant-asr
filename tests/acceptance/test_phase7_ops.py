"""Phase 7 exit gate (plan/phases/phase-7-hardening-ops.md). `make gate PHASE=7`."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from ars.config import Settings
from ars.eval.dashboard import generate
from ars.ops.drift import check_drift, psi
from ars.ops.retention import purge_old_audio

pytestmark = pytest.mark.acceptance
ROOT = Path(__file__).resolve().parent.parent.parent


# 1 -------------------------------------------------------------------------- #
def test_dashboard_generates(tmp_path):
    s = Settings.load(ROOT / "configs" / "default.yaml")
    s.paths.models = str(tmp_path / "models")  # empty -> exercises empty-state
    s.paths.reports = str(tmp_path / "reports")
    s.paths.db = str(tmp_path / "nope.db")
    s.ingest.telemetry_dir = str(tmp_path / "tel")
    index = generate(s, tmp_path / "dash")
    html = index.read_text()
    for section in ("Model registry", "Eval runs", "NDI evolution", "Latency"):
        assert section in html
    assert "empty" in html  # empty-state rendered, no crash


# 2 -------------------------------------------------------------------------- #
def test_psi_zero_for_identical():
    x = list(np.random.default_rng(0).normal(size=500))
    assert psi(x, x) < 0.01


def test_drift_monitors():
    s = Settings.load(ROOT / "configs" / "default.yaml").ops
    ref = list(np.random.default_rng(0).normal(-0.3, 0.1, 500))
    shifted = list(np.random.default_rng(1).normal(-0.9, 0.3, 500))  # clear input drift
    alerts = check_drift(
        {
            "logprob_reference": ref,
            "logprob_current": shifted,
            "noise_shares": {"store-A": {"CB": (0.2, 0.6)}},  # 3x -> noise drift
            "rule_fire_rate": {"es-0001": (30, 5)},  # 6x -> rule storm
            "judge_shares": {"wrong": 15, "hallucination": 5, "total": 100},  # 20% -> quality
        },
        s,
    )
    types = {a["type"] for a in alerts}
    assert types == {"input_drift", "noise_drift", "rule_storm", "quality_alarm"}
    # healthy signals -> silent
    assert (
        check_drift(
            {
                "logprob_reference": ref,
                "logprob_current": ref,
                "noise_shares": {"s": {"CB": (0.3, 0.3)}},
                "rule_fire_rate": {"r": (5, 5)},
                "judge_shares": {"wrong": 2, "hallucination": 1, "total": 100},
            },
            s,
        )
        == []
    )


# 3 -------------------------------------------------------------------------- #
def _wav(dur=0.5):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(dur * 16000), np.float32), 16000, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def test_api_auth_and_limits():
    from tests.fakes import FakeEngine, FakeVad

    from ars.api.app import _RATE, app, get_pipeline, get_settings
    from ars.pipeline import Pipeline

    _RATE.clear()
    s = Settings.load(ROOT / "configs" / "default.yaml")
    s.security.auth_enabled = True
    s.security.tokens = {"demo": "secret"}
    s.security.rate_limit_per_min = 3
    s.security.max_upload_s = 1.0
    pipe = Pipeline(s, FakeVad(0.9), FakeEngine(text="ok"), telemetry_sink=lambda _: None)
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[get_settings] = lambda: s
    client = TestClient(app)
    try:
        files = {"file": ("a.wav", _wav())}
        assert client.post("/v1/transcribe", files={"file": ("a.wav", _wav())}).status_code == 401
        bad = {"authorization": "Bearer nope"}
        assert client.post("/v1/transcribe", files=files, headers=bad).status_code == 403
        ok = {"authorization": "Bearer secret"}
        # oversize (2 s > max 1 s)
        big = {"file": ("b.wav", _wav(2.0))}
        assert client.post("/v1/transcribe", files=big, headers=ok).status_code == 413
        # rate limit: 3 allowed, 4th -> 429
        _RATE.clear()
        codes = [
            client.post("/v1/transcribe", files={"file": ("a.wav", _wav())}, headers=ok).status_code
            for _ in range(4)
        ]
        assert codes[:3] == [200, 200, 200] and codes[3] == 429
    finally:
        app.dependency_overrides.clear()


# 4 -------------------------------------------------------------------------- #
def test_retention_job(tmp_path):
    raw = tmp_path / "raw"
    (raw / "d").mkdir(parents=True)
    old = raw / "d" / "old.wav"
    new = raw / "d" / "new.wav"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    import os

    old_time = time.time() - 200 * 86400
    os.utime(old, (old_time, old_time))
    # dry-run default: nothing deleted
    rep = purge_old_audio(raw, retention_days=90)
    assert rep["dry_run"] and old.exists() and "d/old.wav" in rep["purged"]
    # real purge: old gone, new kept
    rep = purge_old_audio(raw, retention_days=90, dry_run=False)
    assert not old.exists() and new.exists() and rep["kept"] == 1


# 5 -------------------------------------------------------------------------- #
def test_log_scrubbing():
    from ars.telemetry import telemetry_line

    line = telemetry_line(
        trace_id="t",
        store_id="s",
        duration_s=4.0,
        speech_ratio=0.8,
        noise_pred=None,
        noise_confidence=0.0,
        chain_applied=[],
        language="es",
        avg_logprob=-0.3,
        guard_flags=[],
        rules_fired=[],
        latency_ms={"total": 100},
        model_version="0.1.0",
    )
    # no transcript / text field ever in telemetry (privacy — §7.3)
    assert "transcript" not in line and "text" not in line and "raw_text" not in line


# 7 -------------------------------------------------------------------------- #
def test_runbooks_exist():
    md = (ROOT / "docs" / "RUNBOOKS.md").read_text().lower()
    for header in (
        "model rollback",
        "rule retirement",
        "flywheel pause",
        "judge outage",
        "disk-full",
        "restore-from-backup",
    ):
        assert header in md, f"RUNBOOKS.md missing '{header}'"
    assert (ROOT / "docs" / "FIELD-RECORDING-PROTOCOL.md").exists()


# 6 (slow) ------------------------------------------------------------------- #
@pytest.mark.slow
def test_load_smoke():
    from concurrent.futures import ThreadPoolExecutor

    from tests.fakes import FakeEngine, FakeVad

    from ars.api.app import app, get_pipeline, get_settings
    from ars.pipeline import Pipeline

    s = Settings.load(ROOT / "configs" / "default.yaml")
    s.security.auth_enabled = False
    s.security.rate_limit_per_min = 100000
    pipe = Pipeline(s, FakeVad(0.9), FakeEngine(), telemetry_sink=lambda _: None)
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[get_settings] = lambda: s
    client = TestClient(app)
    try:

        def one():
            t0 = time.perf_counter()
            client.post("/v1/transcribe", files={"file": ("a.wav", _wav(5.0))})
            return time.perf_counter() - t0

        single = one()
        with ThreadPoolExecutor(max_workers=10) as ex:
            times = list(ex.map(lambda _: one(), range(10)))
        assert max(times) <= 3 * single + 0.5
    finally:
        app.dependency_overrides.clear()
