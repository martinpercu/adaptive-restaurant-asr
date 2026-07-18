"""Phase 5 exit gate (plan/phases/phase-5-axis3-keydetector.md). `make gate PHASE=5`."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ars.keydetector.lexicon import LexiconIndex, load_menu
from ars.keydetector.phonetics import en_key, es_key
from ars.keydetector.pipeline import Keydetector
from ars.keydetector.rules import load_rules

pytestmark = pytest.mark.acceptance
ROOT = Path(__file__).resolve().parent.parent.parent


def _menu():
    return load_menu(ROOT / "configs" / "menu", "demo")


def _rules(lang):
    return load_rules(ROOT / "configs" / "rules" / f"rules-{lang}.yaml")


def _kd(mode="replace"):
    return Keydetector(_menu(), {lang: _rules(lang) for lang in ("es", "en")}, mode=mode)


# 1 -------------------------------------------------------------------------- #
def test_phonetic_keys_table():
    for w, k in {
        "bocina": "bosina",
        "cocina": "kosina",
        "vaso": "baso",
        "bazo": "baso",
        "hielo": "ielo",
        "cielo": "sielo",
    }.items():
        assert es_key(w) == k
    assert es_key("vaso") == es_key("bazo")  # seseo collision (lexicon-recoverable)
    assert es_key("hielo") != es_key("cielo")  # NOT a collision -> needs a rule
    assert en_key("soup") == en_key("soap")  # collides -> lexicon-recoverable
    assert en_key("flies") != en_key("fries")  # NOT -> needs a rule


# 2 -------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "corrupt,expected",
    [("papas fritaz", "papas fritas"), ("ensalda", "ensalada"), ("hamburgesa", "hamburguesa")],
)
def test_lexicon_recovers_menu_terms(corrupt, expected):
    idx = LexiconIndex(_menu()).build("es")
    hits = idx.match(corrupt.split(), "es")
    recovered = {h[2] for h in hits}
    assert any(expected in r for r in recovered), f"{corrupt!r} -> {recovered}"


def test_lexicon_length_and_stopword_guards():
    idx = LexiconIndex(_menu()).build("es")
    # a far-off word must not fuzzy-match a menu term (length guard)
    assert not idx.match(["universidad"], "es")
    # stopwords never corrected
    assert not idx.match(["de"], "es")


# 3 -------------------------------------------------------------------------- #
def test_rule_engine_semantics():
    kd = _kd("replace")
    # casing preservation
    text, _ = kd.correct("Me da un Tejedor", "es")
    assert "Tenedor" in text
    # active fires in replace; context gate blocks the legit sense
    assert kd.correct("me da un tejedor", "es")[0] == "me da un tenedor"
    assert kd.correct("el tejedor de la fabrica", "es")[0] == "el tejedor de la fabrica"
    # approved does NOT replace in replace mode
    assert kd.correct("quiero un plazo de comida", "es")[0] == "quiero un plazo de comida"
    # no re-trigger: a fired output is not re-corrected
    out, corr = kd.correct("me da un tejedor", "es")
    assert kd.correct(out, "es")[0] == out


def test_lexicon_precedence_over_rules():
    # a correct menu term is locked by the lexicon and never touched by a rule
    kd = _kd("replace")
    assert kd.correct("me da una servilleta", "es")[0] == "me da una servilleta"


# 4 -------------------------------------------------------------------------- #
def test_rules_files_valid():
    for lang in ("es", "en"):
        rules = _rules(lang)
        ids = [r.id for r in rules]
        assert len(ids) == len(set(ids))
        golden = yaml.safe_load((ROOT / "tests" / "golden" / f"cases-{lang}.yaml").read_text())
        fires = {}
        for c in golden:
            fires.setdefault(c["rule_id"], set()).add(c["fires"])
        for r in rules:
            if r.status != "retired":
                assert fires.get(r.id) == {True, False}, f"{r.id} missing golden pair"


def test_rules_reject_bad_schema(tmp_path):
    bad = tmp_path / "rules-es.yaml"
    bad.write_text("- {id: x, lang: es, wrong: a, right: b, bogus: 1}\n")
    with pytest.raises(ValueError):
        load_rules(bad)


# 5 -------------------------------------------------------------------------- #
def test_golden_all():
    kds = {m: _kd(m) for m in ("replace", "log_only")}
    for lang in ("es", "en"):
        for c in yaml.safe_load((ROOT / "tests" / "golden" / f"cases-{lang}.yaml").read_text()):
            text, corr = kds[c["mode"]].correct(c["input"], lang)
            assert text == c["expected"]
            assert (c["rule_id"] in {x.rule_id for x in corr}) == c["fires"]


# 6 -------------------------------------------------------------------------- #
def test_modes():
    text_in = "me da un tejedor"
    log = _kd("log_only").correct(text_in, "es")
    assert log[0] == text_in and any(c.rule_id == "es-0001" for c in log[1])  # logged, unchanged
    rep = _kd("replace").correct(text_in, "es")
    assert rep[0] == "me da un tenedor"  # applied


# 8 -------------------------------------------------------------------------- #
def test_false_correction_rate():
    kd = _kd("replace")
    fired = total = 0
    for lang in ("es", "en"):
        m = ROOT / "data" / "datasets" / f"eval-clean-{lang}-v1" / "manifest.parquet"
        if not m.exists():
            continue
        for text in pd.read_parquet(m)["text"].head(200):
            total += 1
            _, corr = kd.correct(text, lang)
            if corr:
                fired += 1
    if total == 0:
        pytest.skip("no eval-clean sets")
    rate = fired / total
    assert rate <= 0.005, f"false-correction rate {rate:.3%} > 0.5% ({fired}/{total})"


# 9 -------------------------------------------------------------------------- #
def test_latency():
    kd = _kd("replace")
    kd.correct("warm up the index", "en")  # build lazily
    t0 = time.perf_counter()
    for _ in range(50):
        kd.correct("me da un tejedor y una silueta con cielo", "es")
    per = (time.perf_counter() - t0) / 50 * 1000
    assert per <= 20.0, f"keydetector {per:.1f} ms/utt > 20 ms"


# 7 (slow, needs model) ------------------------------------------------------ #
@pytest.mark.slow
def test_ker_improvement(repo_root):
    report = repo_root / "reports" / "keydetector" / "ker.json"
    if not report.exists():
        pytest.skip("no KER report; run the phase-5 eval")
    import json

    r = json.loads(report.read_text())
    for lang in ("es", "en"):
        assert r[lang]["ker_rel_improvement"] >= 0.10
