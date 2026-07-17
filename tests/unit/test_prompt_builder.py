from __future__ import annotations

from ars.asr.prompt_builder import (
    build_bilingual_prompt,
    build_initial_prompt,
    load_menu,
    menu_terms,
)


def _menu(repo_root):
    return load_menu(repo_root / "configs" / "menu", "demo")


def test_menu_terms_include_names_and_service(repo_root):
    terms = menu_terms(_menu(repo_root), "es")
    assert "café" in terms
    assert "papas fritas" in terms
    assert "cuenta" in terms  # service term
    assert len(terms) == len(set(t.lower() for t in terms))  # de-duped


def test_initial_prompt_token_cap(repo_root):
    prompt = build_initial_prompt(_menu(repo_root), "en", max_tokens=10)
    assert len(prompt.split()) <= 10
    assert "," in prompt


def test_bilingual_prompt_has_both_langs(repo_root):
    prompt = build_bilingual_prompt(_menu(repo_root), max_tokens=200).lower()
    assert "café" in prompt or "coffee" in prompt
    assert "fries" in prompt or "papas" in prompt


def test_empty_menu_prompt():
    assert build_bilingual_prompt({}, 200) == ""
