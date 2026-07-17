"""Model registry (plan/02-architecture.md §6, plan/03-data-spec.md §5).

`models/registry.json` holds all model entries. Exactly one entry may be in
`production` and at most one in `shadow`. Promotion/rollback is an atomic rewrite
of this file (phase 6). Phase 1 initializes the `0.1.0` production baseline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Stage = Literal["production", "shadow", "candidate", "retired"]


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    stage: Stage
    base_model: str
    adapter: str | None = None
    ct2_path: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["es", "en"])
    gates: dict = Field(default_factory=dict)
    sensitivity_run: str | None = None
    promoted_at: str | None = None
    promoted_by: str | None = None


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[RegistryEntry] = Field(default_factory=list)

    # --- loading / saving -------------------------------------------------- #
    @classmethod
    def load(cls, path: str | Path = "models/registry.json") -> ModelRegistry:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.model_validate_json(p.read_text(encoding="utf-8"))

    def save(self, path: str | Path = "models/registry.json") -> None:
        self._check_invariants()
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )

    # --- queries ----------------------------------------------------------- #
    def get(self, version: str) -> RegistryEntry | None:
        return next((e for e in self.entries if e.version == version), None)

    def production(self) -> RegistryEntry | None:
        return next((e for e in self.entries if e.stage == "production"), None)

    def shadow(self) -> RegistryEntry | None:
        return next((e for e in self.entries if e.stage == "shadow"), None)

    # --- mutation ---------------------------------------------------------- #
    def add(self, entry: RegistryEntry) -> None:
        if self.get(entry.version) is not None:
            raise ValueError(f"version already registered: {entry.version}")
        self.entries.append(entry)
        self._check_invariants()

    def _check_invariants(self) -> None:
        if sum(e.stage == "production" for e in self.entries) > 1:
            raise ValueError("more than one production entry")
        if sum(e.stage == "shadow" for e in self.entries) > 1:
            raise ValueError("more than one shadow entry")


def init_baseline(
    path: str | Path = "models/registry.json",
    version: str = "0.1.0",
    base_model: str = "whisper-small",
) -> RegistryEntry:
    """Idempotently ensure a production baseline entry exists."""
    reg = ModelRegistry.load(path)
    existing = reg.get(version)
    if existing is not None:
        return existing
    entry = RegistryEntry(
        version=version,
        stage="production",
        base_model=base_model,
        adapter=None,
        ct2_path=None,
        languages=["es", "en"],
        promoted_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        promoted_by="phase1-baseline",
    )
    reg.add(entry)
    reg.save(path)
    return entry
