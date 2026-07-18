"""Golden suite for axis-3 rules (plan/phases/phase-5 §5.5). Runs in CI.

Each case: {rule_id, input, expected, mode, fires}. A rule without both a positive and a
negative case fails collection (CLAUDE.md rule 5). This IS the axis-3 regression net.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ars.keydetector.lexicon import load_menu
from ars.keydetector.pipeline import Keydetector
from ars.keydetector.rules import load_rules

ROOT = Path(__file__).resolve().parent.parent.parent


def _cases():
    out = []
    for lang in ("es", "en"):
        for c in yaml.safe_load((Path(__file__).parent / f"cases-{lang}.yaml").read_text()):
            out.append((lang, c))
    return out


def _rules(lang):
    return load_rules(ROOT / "configs" / "rules" / f"rules-{lang}.yaml")


@pytest.fixture(scope="module")
def kds():
    menu = load_menu(ROOT / "configs" / "menu", "demo")
    rules = {lang: _rules(lang) for lang in ("es", "en")}
    return {mode: Keydetector(menu, rules, mode=mode) for mode in ("replace", "log_only")}


def test_every_rule_has_positive_and_negative():
    cases = _cases()
    for lang in ("es", "en"):
        by_rule: dict[str, set[bool]] = {}
        for cl, c in cases:
            if cl == lang:
                by_rule.setdefault(c["rule_id"], set()).add(c["fires"])
        for rule in _rules(lang):
            if rule.status == "retired":
                continue
            assert by_rule.get(rule.id) == {True, False}, f"{rule.id}: needs positive AND negative"


@pytest.mark.parametrize("lang,case", _cases())
def test_golden_case(lang, case, kds):
    kd = kds[case["mode"]]
    text, corrections = kd.correct(case["input"], lang)
    fired = case["rule_id"] in {c.rule_id for c in corrections}
    assert text == case["expected"], f"{case['rule_id']}: {text!r} != {case['expected']!r}"
    assert fired == case["fires"], f"{case['rule_id']}: fired={fired} expected={case['fires']}"
