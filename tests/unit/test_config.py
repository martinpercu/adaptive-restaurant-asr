from __future__ import annotations

import pytest
from pydantic import ValidationError

from ars.config import Settings


def test_defaults_load(repo_root):
    s = Settings.load(repo_root / "configs" / "default.yaml")
    assert s.seed == 1337
    assert s.vad.min_speech_ratio == 0.2
    assert s.asr.condition_on_previous_text is False
    assert s.asr.temperature == [0.0, 0.2, 0.4]
    assert s.judge.provider == "anthropic"


def test_unknown_top_level_key_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("seed: 1\nbogus_key: 3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load(cfg)


def test_unknown_nested_key_rejected(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("vad:\n  not_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load(cfg)


def test_env_override(repo_root, monkeypatch):
    monkeypatch.setenv("ARS_VAD__MIN_SPEECH_RATIO", "0.35")
    monkeypatch.setenv("ARS_SEED", "42")
    s = Settings.load(repo_root / "configs" / "default.yaml")
    assert s.vad.min_speech_ratio == 0.35
    assert s.seed == 42


def test_extra_env_var_does_not_break_load(repo_root, monkeypatch):
    # ARS_S3_* are read by S3Storage, not Settings; they must not trip extra="forbid".
    monkeypatch.setenv("ARS_S3_ENDPOINT", "http://localhost:9000")
    s = Settings.load(repo_root / "configs" / "default.yaml")
    assert s.storage.backend == "local"
