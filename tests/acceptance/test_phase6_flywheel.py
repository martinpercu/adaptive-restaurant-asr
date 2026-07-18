"""Phase 6 exit gate (plan/phases/phase-6-flywheel.md). `make gate PHASE=6`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ars.config import Settings
from ars.db import connect, insert_transcription, insert_utterance
from ars.flywheel.harvester import HarvestConfig, harvest
from ars.flywheel.lifecycle import advance_rules
from ars.flywheel.pair_miner import mine
from ars.flywheel.promote import atomic_save, promotion_gate
from ars.judge import AnthropicJudge, MockJudge, OpenAIJudge, route_verdict
from ars.registry import ModelRegistry, RegistryEntry

pytestmark = pytest.mark.acceptance
ROOT = Path(__file__).resolve().parent.parent.parent


def _tmp_settings(tmp_path):
    s = Settings.load(ROOT / "configs" / "default.yaml")
    s.paths.db = str(tmp_path / "ars.db")
    s.paths.models = str(tmp_path / "models")
    s.paths.reports = str(tmp_path / "reports")
    (tmp_path / "models").mkdir()
    return s


# 1 -------------------------------------------------------------------------- #
def test_harvester_selection(tmp_path):
    conn = connect(tmp_path / "ars.db")
    # 4 flagged utterances + 96 clean -> control 2% = ~2
    for i in range(96):
        insert_utterance(
            conn,
            utterance_id=f"c{i}",
            path="",
            lang="es",
            store_id="s",
            captured_at=None,
            duration_s=1.0,
            speech_ratio=0.9,
        )
        insert_transcription(
            conn,
            utterance_id=f"c{i}",
            model_version="0.1.0",
            raw_text="ok",
            final_text="ok",
            avg_logprob=-0.2,
            guard_flags=[],
        )
    insert_utterance(
        conn,
        utterance_id="low",
        path="",
        lang="es",
        store_id="s",
        captured_at=None,
        duration_s=1.0,
        speech_ratio=0.9,
    )
    insert_transcription(
        conn,
        utterance_id="low",
        model_version="0.1.0",
        raw_text="x",
        final_text="x",
        avg_logprob=-0.9,
        guard_flags=[],
    )
    insert_utterance(
        conn,
        utterance_id="guard",
        path="",
        lang="es",
        store_id="s",
        captured_at=None,
        duration_s=1.0,
        speech_ratio=0.9,
    )
    insert_transcription(
        conn,
        utterance_id="guard",
        model_version="0.1.0",
        raw_text="x",
        final_text="x",
        avg_logprob=-0.2,
        guard_flags=["repetition_truncated"],
    )

    sel = harvest(conn, HarvestConfig(control_frac=0.02), seed=1337)
    by_uid = {s["utterance_id"]: s["reasons"] for s in sel}
    assert "low_logprob" in by_uid["low"]
    assert "guard_flags" in by_uid["guard"]
    controls = [s for s in sel if s["reasons"] == ["control"]]
    assert len(controls) == 2  # 2% of 96
    assert len(harvest(conn, HarvestConfig(cap=1))) == 1  # cap honored


# 2 -------------------------------------------------------------------------- #
def test_judge_routing_table():
    from ars.contracts import JudgeVerdict

    correct = JudgeVerdict(verdict="correct", order_core_match=True, confidence=0.9)
    assert route_verdict(correct) == "training"
    minor = JudgeVerdict(
        verdict="minor_errors", corrected_reference="x", order_core_match=True, confidence=0.85
    )
    assert route_verdict(minor) == "training+miner"
    minor_low = JudgeVerdict(
        verdict="minor_errors", corrected_reference="x", order_core_match=True, confidence=0.5
    )
    assert route_verdict(minor_low) == "review"
    assert (
        route_verdict(JudgeVerdict(verdict="wrong", order_core_match=False, confidence=0.9))
        == "review"
    )


def test_judge_client_contract():
    valid = json.dumps({"verdict": "correct", "order_core_match": True, "confidence": 0.9})
    # MockJudge scripted
    mj = MockJudge(default=json.loads(valid))
    assert mj.judge_one({"transcript": "hi"}).verdict == "correct"

    # provider request shapes
    a = AnthropicJudge("claude-sonnet-5").build_request({"transcript": "x", "lang": "es"})
    assert a["output_config"]["format"]["type"] == "json_schema" and "temperature" not in a
    o = OpenAIJudge("gpt-4o-mini").build_request({"transcript": "x", "lang": "es"})
    assert o["response_format"]["json_schema"]["strict"] and o["temperature"] == 0

    # invalid-then-valid retry via injected transport
    seq = iter(["not json", valid])
    j = OpenAIJudge("gpt-4o-mini", transport=lambda _p: next(seq))
    assert j.judge_one({"transcript": "x"}).verdict == "correct"
    # hard fail: invalid twice -> raises
    from pydantic import ValidationError

    j2 = OpenAIJudge("gpt-4o-mini", transport=lambda _p: "still not json")
    with pytest.raises(ValidationError):
        j2.judge_one({"transcript": "x"})


# 3 -------------------------------------------------------------------------- #
def test_pair_miner_finds_injected_pair():
    pairs = [("me da una silueta", "me da una servilleta", "es")] * 7
    mined = mine(pairs)
    assert any(m.wrong == "silueta" and m.right == "servilleta" for m in mined)
    m = next(m for m in mined if m.wrong == "silueta")
    rule = m.to_rule("mined-es-000")
    assert rule["status"] == "candidate" and rule["provenance"]["source"] == "mined"
    assert len(m.golden_skeleton()) == 2  # positive + negative template

    # 4 observations -> below evidence bar -> no rule
    assert not mine([("me da una silueta", "me da una servilleta", "es")] * 4)
    # wrong is a menu term -> suppressed
    assert not mine([("quiero un cafe", "quiero un te", "es")] * 7, menu_terms={"es": {"cafe"}})


# 4 -------------------------------------------------------------------------- #
def test_rule_lifecycle_automation():
    rules = [{"id": "es-9001", "status": "approved"}]
    # enough evidence -> flip
    flips = advance_rules(
        rules, {"es-9001": {"weeks_approved": 3, "correct_fires": 8, "conflicts": 0}}
    )
    assert flips == [("es-9001", "approved", "active")] and rules[0]["status"] == "active"
    # without evidence -> no flip
    r2 = [{"id": "es-9002", "status": "approved"}]
    assert not advance_rules(
        r2, {"es-9002": {"weeks_approved": 1, "correct_fires": 8, "conflicts": 0}}
    )
    assert not advance_rules(
        r2, {"es-9002": {"weeks_approved": 3, "correct_fires": 8, "conflicts": 2}}
    )


# 5 -------------------------------------------------------------------------- #
def test_shadow_and_promotion_logic():
    good = {
        "prod_wer": 0.30,
        "cand_wer": 0.25,
        "prod_logprob": -0.3,
        "cand_logprob": -0.32,
        "cand_latency_ms": 2000,
    }
    assert promotion_gate(good).promote
    assert not promotion_gate({**good, "cand_wer": 0.31}).promote  # WER worse
    assert not promotion_gate({**good, "cand_logprob": -0.45}).promote  # logprob worse >0.1
    assert not promotion_gate({**good, "cand_latency_ms": 5000}).promote  # latency
    assert not promotion_gate(good, regression_ok=False).promote  # regression


def test_registry_flip_atomic_and_rollback(tmp_path):
    reg = ModelRegistry()
    reg.add(RegistryEntry(version="0.1.0", stage="production", base_model="whisper-small"))
    reg.add(RegistryEntry(version="0.2.0", stage="candidate", base_model="whisper-small"))
    path = tmp_path / "registry.json"
    reg.promote("0.2.0", promoted_by="flywheel")
    atomic_save(reg, path)
    loaded = ModelRegistry.load(path)  # file is valid after the flip
    assert loaded.production().version == "0.2.0" and loaded.get("0.1.0").stage == "previous"
    loaded.rollback()
    assert loaded.production().version == "0.1.0"


# 6 -------------------------------------------------------------------------- #
def test_cycle_idempotent_resume(tmp_path):
    from ars.flywheel.simulate import simulate

    s = _tmp_settings(tmp_path)
    first = simulate(s, cycle_id="cyc-1")
    second = simulate(s, cycle_id="cyc-1")  # rerun same cycle_id
    assert first["mined_candidates"] == second["mined_candidates"]  # no duplication
    assert first["promotion"] == second["promotion"]


# 7 (the phase gate) --------------------------------------------------------- #
def test_full_simulated_cycle(tmp_path):
    from ars.flywheel.simulate import simulate

    s = _tmp_settings(tmp_path)
    cycle = simulate(s, days=7, seed=1337, cycle_id="cyc-full")
    # (a) injected confusion pair mined as candidate
    assert cycle["mined_candidates"], "no candidate rule mined"
    # (b) noise-shift flagged
    assert any(f["subtype"] == "CB" for f in cycle["noise_flags"])
    # (c) retrain triggered (pool over threshold)
    assert cycle["retrain"]["triggered"]
    # (d)+(e) shadow report -> promotion decision matches passing gates
    assert cycle["promotion"]["decision"] == "promoted"
    # (f) CYCLE.md complete
    md = (Path(s.paths.reports) / "flywheel" / "cyc-full" / "CYCLE.md").read_text()
    assert "Judge cost estimate" in md and "Runbook" in md
    # registry actually flipped to the candidate
    assert (
        ModelRegistry.load(Path(s.paths.models) / "registry.json").production().version == "0.2.0"
    )
