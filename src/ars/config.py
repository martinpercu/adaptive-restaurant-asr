"""Root configuration (plan/01-conventions.md §6, plan/phases/phase-0 §0.2).

Single source of truth: `configs/default.yaml`, validated by `Settings`.
- `extra="forbid"` everywhere -> unknown keys fail fast.
- Env overrides use the `ARS_` prefix with `__` nesting (e.g. `ARS_VAD__MIN_SPEECH_RATIO=0.25`).
- S3 credentials (`ARS_S3_*`) are read directly by `ars.storage.S3Storage`, not here.
No other module reads YAML directly.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG = "configs/default.yaml"
_yaml_path: ContextVar[Path] = ContextVar("_yaml_path", default=Path(DEFAULT_CONFIG))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Section):
    data: str = "data"
    models: str = "models"
    reports: str = "reports"
    configs: str = "configs"
    db: str = "data/db/ars.db"
    noise_taxonomy: str = "configs/noise_taxonomy.yaml"


class VadCfg(_Section):
    backend: str = "silero"
    min_speech_ratio: float = 0.2
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100


class AsrGuardCfg(_Section):
    no_speech_prob_max: float = 0.85
    avg_logprob_min: float = -1.0
    repetition_ngram: int = 3
    repetition_max_repeats: int = 3


class AsrCfg(_Section):
    backend: str = "faster-whisper"
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    temperature: list[float] = [0.0, 0.2, 0.4]
    initial_prompt_max_tokens: int = 200
    guard: AsrGuardCfg = AsrGuardCfg()


class PreprocessCfg(_Section):
    enabled: bool = True
    mode: str = "log_only"  # off | log_only | active (production-safe default: log_only)
    classifier: str = "spectral"
    classifier_path: str = "models/noise_classifier/latest/model.pt"
    policy_path: str = "configs/mitigation_policy.yaml"
    min_confidence: float = 0.6


class KeydetectorCfg(_Section):
    enabled: bool = True
    mode: str = "log_only"  # replace | log_only
    rules_dir: str = "configs/rules"
    menu_dir: str = "configs/menu"
    fuzzy_threshold: float = 0.85
    max_false_correction_rate: float = 0.005


class TrainingCfg(_Section):
    base_model: str = "whisper-medium"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    learning_rate: float = 1.0e-4
    batch_size: int = 8
    min_noisy_wer_rel_improvement: float = 0.15
    max_clean_wer_rel_regression: float = 0.02
    max_keyword_recall_drop: float = 0.01


class JudgeCfg(_Section):
    provider: str = "anthropic"  # anthropic | openai
    model: str = "claude-sonnet-5"
    batch: bool = True
    calibration_min_agreement: float = 0.90
    calibration_min_items: int = 100


class FlywheelCfg(_Section):
    cadence: str = "weekly"
    low_confidence_logprob: float = -0.8
    review_queue_only_doubtful: bool = True


class NdiWeights(_Section):
    d_wer: float = 0.5
    d_ker: float = 0.4
    hallucination: float = 0.1


class EvalCfg(_Section):
    ndi_weights: NdiWeights = NdiWeights()
    wer_decimals: int = 3
    snr_tolerance_db: float = 0.5


class IngestCfg(_Section):
    store_audio: bool = True
    telemetry_dir: str = "data/telemetry"


class StorageCfg(_Section):
    backend: str = "local"  # local | s3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARS_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    seed: int = 1337
    paths: PathsCfg = PathsCfg()
    vad: VadCfg = VadCfg()
    asr: AsrCfg = AsrCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    keydetector: KeydetectorCfg = KeydetectorCfg()
    training: TrainingCfg = TrainingCfg()
    judge: JudgeCfg = JudgeCfg()
    flywheel: FlywheelCfg = FlywheelCfg()
    eval: EvalCfg = EvalCfg()
    ingest: IngestCfg = IngestCfg()
    storage: StorageCfg = StorageCfg()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (highest first): explicit init kwargs > env > .env > YAML file.
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=_yaml_path.get())
        return (init_settings, env_settings, dotenv_settings, yaml_source)

    @classmethod
    def load(cls, config_path: str | Path | None = None, **overrides: object) -> Settings:
        """Load settings from a YAML file, with env and explicit-kwarg overrides."""
        token = _yaml_path.set(Path(config_path) if config_path else Path(DEFAULT_CONFIG))
        try:
            return cls(**overrides)
        finally:
            _yaml_path.reset(token)
