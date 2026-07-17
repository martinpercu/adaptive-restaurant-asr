from __future__ import annotations

from ars.eval import fold_diacritics, normalize, tokens


def test_lowercase_and_punctuation():
    assert normalize("Hola, ¿me da café?", "es") == "hola me da café"
    assert normalize("Fries, please!", "en") == "fries please"


def test_en_keeps_intraword_apostrophe():
    assert normalize("I don't want water", "en") == "i don't want water"


def test_apostrophe_at_edge_is_dropped():
    assert normalize("'quote' end", "en") == "quote end"


def test_diacritics_preserved_in_surface_form():
    assert "café" in normalize("un Café", "es")


def test_fold_diacritics_for_keys_only():
    assert fold_diacritics("café") == "cafe"
    assert fold_diacritics("piñón") == "pinon"


def test_tokens():
    assert tokens("me da papas fritas", "es") == ["me", "da", "papas", "fritas"]
    assert tokens("", "es") == []


def test_number_words_left_as_is():
    assert normalize("dos hamburguesas", "es") == "dos hamburguesas"
    assert "2" not in normalize("dos hamburguesas", "es")
