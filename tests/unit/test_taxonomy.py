from __future__ import annotations

import pytest
import yaml

from ars.noise_lab.taxonomy import Taxonomy, load_taxonomy


def test_seed_taxonomy_loads(repo_root):
    tax = load_taxonomy(repo_root / "configs" / "noise_taxonomy.yaml")
    assert set(tax.families) == {"A", "B", "C", "D"}
    assert len(tax.subtypes) == 8
    # canonical grid = 8 subtypes x 3 canonical levels
    assert len(tax.canonical_cells()) == 24
    assert tax.canonical_levels == ["05", "10", "15"]


def test_levels_map_to_documented_snr(repo_root):
    tax = load_taxonomy(repo_root / "configs" / "noise_taxonomy.yaml")
    assert tax.levels["05"].snr_db == 10.0
    assert tax.levels["10"].snr_db == 0.0
    assert tax.levels["15"].snr_db == -5.0


def _base() -> dict:
    return {
        "families": {"A": {"name": "kitchen"}},
        "subtypes": {"AA": {"family": "A", "name": "x"}},
        "levels": {"10": {"snr_db": 0.0, "name": "equal"}},
        "canonical_levels": ["10"],
    }


def test_subtype_unknown_family_rejected():
    d = _base()
    d["subtypes"]["AB"] = {"family": "Z", "name": "y"}
    with pytest.raises(ValueError):
        Taxonomy.model_validate(d)


def test_subtype_family_letter_mismatch_rejected():
    d = _base()
    d["subtypes"]["BA"] = {"family": "A", "name": "y"}  # first letter B != family A
    with pytest.raises(ValueError):
        Taxonomy.model_validate(d)


def test_wrong_snr_rejected():
    d = _base()
    d["levels"]["10"]["snr_db"] = 3.0
    with pytest.raises(ValueError):
        Taxonomy.model_validate(d)


def test_canonical_level_must_exist():
    d = _base()
    d["canonical_levels"] = ["99"]
    with pytest.raises(ValueError):
        Taxonomy.model_validate(d)


def test_extra_key_rejected(repo_root):
    raw = yaml.safe_load((repo_root / "configs" / "noise_taxonomy.yaml").read_text())
    raw["surprise"] = 1
    with pytest.raises(ValueError):
        Taxonomy.model_validate(raw)
