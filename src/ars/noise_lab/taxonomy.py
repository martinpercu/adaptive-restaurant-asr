"""Noise taxonomy loader & validator (plan/01-conventions.md §3).

`configs/noise_taxonomy.yaml` is the single source of truth. Everything downstream
discovers subtypes from here. Loading validates structural invariants and fails loud.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

# level -> documented SNR in dB (plan/01 §3.1). Used to check the registry.
DOCUMENTED_SNR: dict[str, float] = {"05": 10.0, "10": 0.0, "15": -5.0, "20": -10.0}


class Family(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class Subtype(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: str
    name: str
    desc: str = ""


class Level(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snr_db: float
    name: str


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    families: dict[str, Family]
    subtypes: dict[str, Subtype]
    levels: dict[str, Level]
    canonical_levels: list[str]

    @model_validator(mode="after")
    def _check(self) -> Taxonomy:
        for code, sub in self.subtypes.items():
            if len(code) != 2 or not code.isupper() or not code.isalpha():
                raise ValueError(f"subtype code must be two uppercase letters: {code!r}")
            if sub.family not in self.families:
                raise ValueError(f"subtype {code} references unknown family {sub.family!r}")
            if code[0] != sub.family:
                raise ValueError(
                    f"subtype {code} first letter must equal its family {sub.family!r}"
                )
        for code, lvl in self.levels.items():
            if code in DOCUMENTED_SNR and lvl.snr_db != DOCUMENTED_SNR[code]:
                raise ValueError(
                    f"level {code}: snr_db {lvl.snr_db} != documented {DOCUMENTED_SNR[code]}"
                )
        for code in self.canonical_levels:
            if code not in self.levels:
                raise ValueError(f"canonical level {code!r} is not defined in levels")
        return self

    def subtype_codes(self) -> list[str]:
        return sorted(self.subtypes)

    def canonical_cells(self) -> list[tuple[str, str]]:
        """All (subtype, level) pairs on the canonical grid."""
        return [(s, lvl) for s in self.subtype_codes() for lvl in self.canonical_levels]


def load_taxonomy(path: str | Path = "configs/noise_taxonomy.yaml") -> Taxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Taxonomy.model_validate(data)
