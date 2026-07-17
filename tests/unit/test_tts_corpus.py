"""Grammar/coverage guards for the TTS corpus — pure, no audio (runs in CI)."""

from __future__ import annotations

import pytest
from scripts.tts_corpus import generate_utterances, load_confusion, load_menu


@pytest.fixture
def menu(repo_root):
    return load_menu(repo_root / "configs" / "menu" / "demo.yaml")


@pytest.fixture
def confusion(repo_root):
    return load_confusion(repo_root / "configs" / "confusion_seed.yaml")


@pytest.mark.parametrize("lang", ["es", "en"])
def test_count_meets_minimum(lang, menu, confusion):
    utts = generate_utterances(lang, menu, confusion, n=500)
    assert len(utts) >= 500


@pytest.mark.parametrize("lang", ["es", "en"])
def test_every_confusion_target_covered(lang, menu, confusion):
    utts = generate_utterances(lang, menu, confusion, n=500)
    covered = {kw for u in utts for kw in u.keywords}
    targets = {c["target"] for c in confusion[lang]}
    missing = targets - covered
    assert not missing, f"uncovered confusion targets ({lang}): {sorted(missing)}"


@pytest.mark.parametrize("lang", ["es", "en"])
def test_deterministic(lang, menu, confusion):
    a = generate_utterances(lang, menu, confusion, n=500, seed=1337)
    b = generate_utterances(lang, menu, confusion, n=500, seed=1337)
    assert [u.text for u in a] == [u.text for u in b]


@pytest.mark.parametrize("lang", ["es", "en"])
def test_ids_unique_and_well_formed(lang, menu, confusion):
    utts = generate_utterances(lang, menu, confusion, n=500)
    ids = [u.utterance_id for u in utts]
    assert len(ids) == len(set(ids))
    assert all(uid.startswith(f"cl-{lang}-") for uid in ids)
