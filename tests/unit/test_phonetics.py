from __future__ import annotations

import pytest

from ars.keydetector.phonetics import en_key, es_key, phonetic_key


@pytest.mark.parametrize(
    "word,expected",
    [
        ("bocina", "bosina"),
        ("cocina", "kosina"),
        ("vaso", "baso"),
        ("bazo", "baso"),  # seseo homophone -> same key (intended collision)
        ("hielo", "ielo"),
        ("cielo", "sielo"),
    ],
)
def test_es_key_spec_examples(word, expected):
    assert es_key(word) == expected


def test_es_seseo_collisions():
    assert es_key("plato") != es_key("plazo")  # differ (t vs s)
    assert es_key("caza") == es_key("casa")  # z->s, seseo


def test_en_key_confusion_pairs_collide():
    assert en_key("soup") == en_key("soap")
    assert en_key("glass") == en_key("class")


def test_phonetic_key_dispatch():
    assert phonetic_key("cocina", "es") == es_key("cocina")
    assert phonetic_key("soup", "en") == en_key("soup")
