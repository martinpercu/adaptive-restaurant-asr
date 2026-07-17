from __future__ import annotations

import pytest

from ars.registry import ModelRegistry, RegistryEntry, init_baseline


def test_add_and_query():
    reg = ModelRegistry()
    reg.add(RegistryEntry(version="0.1.0", stage="production", base_model="whisper-small"))
    assert reg.production().version == "0.1.0"
    assert reg.get("0.1.0").base_model == "whisper-small"
    assert reg.shadow() is None


def test_single_production_invariant():
    reg = ModelRegistry()
    reg.add(RegistryEntry(version="0.1.0", stage="production", base_model="whisper-small"))
    with pytest.raises(ValueError):
        reg.add(RegistryEntry(version="0.2.0", stage="production", base_model="whisper-small"))


def test_duplicate_version_rejected():
    reg = ModelRegistry()
    reg.add(RegistryEntry(version="0.1.0", stage="production", base_model="whisper-small"))
    with pytest.raises(ValueError):
        reg.add(RegistryEntry(version="0.1.0", stage="candidate", base_model="whisper-small"))


def test_roundtrip(tmp_path):
    reg = ModelRegistry()
    reg.add(RegistryEntry(version="0.1.0", stage="production", base_model="whisper-small"))
    p = tmp_path / "registry.json"
    reg.save(p)
    loaded = ModelRegistry.load(p)
    assert loaded.production().version == "0.1.0"


def test_init_baseline_idempotent(tmp_path):
    p = tmp_path / "registry.json"
    e1 = init_baseline(p, base_model="whisper-small")
    e2 = init_baseline(p, base_model="whisper-small")
    assert e1.version == e2.version == "0.1.0"
    assert ModelRegistry.load(p).production().promoted_by == "phase1-baseline"
    assert len(ModelRegistry.load(p).entries) == 1
