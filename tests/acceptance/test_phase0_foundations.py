"""Phase 0 exit gate (plan/phases/phase-0-foundations.md).

Run via `make gate PHASE=0`. Dataset/corpus-dependent tests skip with a clear
message when the local heavy path (`make download-data` / `make tts-corpus`) has
not been run.
"""

from __future__ import annotations

import math
import subprocess

import pandas as pd
import pytest
import soundfile as sf
import yaml
from pydantic import ValidationError

from ars.config import Settings
from ars.contracts import DatasetManifestRow
from ars.noise_lab.taxonomy import load_taxonomy
from ars.storage import LocalStorage

pytestmark = pytest.mark.acceptance


# 1 -------------------------------------------------------------------------- #
def test_config_loads_and_rejects_unknown_keys(repo_root, tmp_path):
    s = Settings.load(repo_root / "configs" / "default.yaml")
    assert s.seed == 1337
    bad = tmp_path / "bad.yaml"
    bad.write_text("definitely_not_a_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load(bad)


# 2 -------------------------------------------------------------------------- #
def test_storage_roundtrip(tmp_path):
    st = LocalStorage(tmp_path / "root")
    st.put("clean/es/a.wav", b"data")
    assert st.exists("clean/es/a.wav")
    assert st.get("clean/es/a.wav") == b"data"
    assert st.list("clean/") == ["clean/es/a.wav"]


# 3 -------------------------------------------------------------------------- #
def test_taxonomy_registry_valid(repo_root):
    tax = load_taxonomy(repo_root / "configs" / "noise_taxonomy.yaml")
    for code, sub in tax.subtypes.items():
        assert sub.family in tax.families
        assert code[0] == sub.family
    assert tax.levels["05"].snr_db == 10.0
    assert tax.levels["10"].snr_db == 0.0
    assert tax.levels["15"].snr_db == -5.0


# 4 -------------------------------------------------------------------------- #
def test_dataset_manifests_valid(repo_root):
    manifests = sorted((repo_root / "data").rglob("manifest.parquet"))
    if not manifests:
        pytest.skip("no dataset manifests present; run `make download-data`")
    for manifest in manifests:
        df = pd.read_parquet(manifest)
        assert len(df) > 0, f"empty manifest: {manifest}"
        if "utterance_id" not in df.columns:
            continue  # noise-bank manifest is NoiseBankRow — validated by the phase-2 gate
        # utterance_ids must be unique within a manifest (per-source seq collisions
        # scramble downstream text/audio pairing — guards the fix in DECISIONS).
        ids = df["utterance_id"].tolist()
        assert len(ids) == len(set(ids)), f"duplicate utterance_ids in {manifest}"
        for rec in df.to_dict(orient="records"):
            # parquet stores SQL-null as NaN in numeric columns; normalize to None so the
            # optional int/float contract fields validate (NaN is the logical null here).
            rec = {
                k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()
            }
            kw = rec.get("keywords")
            rec["keywords"] = list(kw) if kw is not None else []
            row = DatasetManifestRow.model_validate(rec)
            assert row.lang in {"es", "en"}
            assert (manifest.parent / row.path).exists(), f"missing audio: {row.path}"


# 5 -------------------------------------------------------------------------- #
def test_tts_corpus_manifest(repo_root):
    confusion = yaml.safe_load(
        (repo_root / "configs" / "confusion_seed.yaml").read_text(encoding="utf-8")
    )
    seen_any = False
    for lang in ("es", "en"):
        manifest = repo_root / "data" / "datasets" / f"tts-{lang}-v1" / "manifest.parquet"
        if not manifest.exists():
            continue
        seen_any = True
        df = pd.read_parquet(manifest)
        assert len(df) >= 500, f"{lang}: expected >=500 utts, got {len(df)}"

        covered = {kw for kws in df["keywords"] if kws is not None for kw in kws}
        targets = {c["target"] for c in confusion[lang]}
        assert not (targets - covered), f"{lang}: uncovered targets {sorted(targets - covered)}"

        for rel in df["path"]:
            info = sf.info(str(manifest.parent / rel))
            assert info.samplerate == 16000 and info.channels == 1
    if not seen_any:
        pytest.skip("no TTS corpus present; run `make tts-corpus`")


# 6 -------------------------------------------------------------------------- #
def test_make_targets(repo_root):
    if not (repo_root / "Makefile").exists():
        pytest.skip("no Makefile")
    for target in ("lint", "test"):
        proc = subprocess.run(["make", target], cwd=repo_root, capture_output=True, text=True)
        assert proc.returncode == 0, f"`make {target}` failed:\n{proc.stdout}\n{proc.stderr}"
