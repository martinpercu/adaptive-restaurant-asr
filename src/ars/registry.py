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

Stage = Literal["production", "shadow", "candidate", "previous", "retired"]


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

    def promote(self, version: str, promoted_by: str = "manual") -> None:
        """Atomic blue-green: current production -> `previous`, `version` -> production.
        Any older `previous` is retired so exactly one rollback target remains."""
        target = self.get(version)
        if target is None:
            raise ValueError(f"unknown version: {version}")
        for e in self.entries:
            if e.stage == "previous":
                e.stage = "retired"
        for e in self.entries:
            if e.stage == "production":
                e.stage = "previous"
        target.stage = "production"
        target.promoted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        target.promoted_by = promoted_by
        self._check_invariants()

    def rollback(self) -> RegistryEntry:
        """One command: restore the `previous` entry to production (current -> previous)."""
        prev = next((e for e in self.entries if e.stage == "previous"), None)
        if prev is None:
            raise ValueError("no `previous` entry to roll back to")
        for e in self.entries:
            if e.stage == "production":
                e.stage = "previous"
        prev.stage = "production"
        prev.promoted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        prev.promoted_by = "rollback"
        self._check_invariants()
        return prev


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


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="ars.registry", description="model registry")
    p.add_argument("command", choices=["init", "list", "promote", "rollback"])
    p.add_argument("version", nargs="?", default=None)
    p.add_argument("--path", default="models/registry.json")
    args = p.parse_args(argv)

    if args.command == "init":
        e = init_baseline(args.path)
        print(f"baseline {e.version} ({e.stage})")
        return 0
    reg = ModelRegistry.load(args.path)
    if args.command == "list":
        for e in reg.entries:
            print(f"  {e.version:8s} {e.stage:11s} base={e.base_model} adapter={e.adapter}")
    elif args.command == "promote":
        reg.promote(args.version)
        reg.save(args.path)
        print(f"promoted {args.version} -> production")
    elif args.command == "rollback":
        e = reg.rollback()
        reg.save(args.path)
        print(f"rolled back to {e.version} -> production")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
