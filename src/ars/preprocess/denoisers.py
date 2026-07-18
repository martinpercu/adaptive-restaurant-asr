"""Mitigation chains (plan/phases/phase-3 §3.2).

One interface, several implementations, chosen per subtype by the generated policy.
Denoisers can *hurt* ASR even when audio sounds cleaner — the policy trusts measured
ΔWER only, never assumptions. Heavy backends (DeepFilterNet, Demucs) are lazy-imported
so the base install and CI stay light; `available_chains()` reports which are usable.

Contract: `process(audio, sr) -> np.ndarray` returns the same length/dtype/sr (16 kHz
mono float32 at the interface; a chain that works at another rate resamples internally).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

SR = 16000


@runtime_checkable
class Denoiser(Protocol):
    chain_id: str

    def process(self, audio: np.ndarray, sr: int = SR) -> np.ndarray: ...


def _fit_length(out: np.ndarray, n: int) -> np.ndarray:
    """Trim/pad a chain's output back to the input length (resample round-trips drift)."""
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if len(out) > n:
        return out[:n]
    if len(out) < n:
        return np.pad(out, (0, n - len(out)))
    return out


class NoneDenoiser:
    """Identity control chain."""

    chain_id = "none"

    def process(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        return np.ascontiguousarray(audio, dtype=np.float32)


class SpectralGate:
    """Stationary spectral gating via noisereduce — for stationary hum/sizzle (AB)."""

    chain_id = "spectral_gate"

    def process(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        import noisereduce as nr  # noqa: PLC0415 (optional dep, lazy)

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not np.any(audio):
            return audio  # silence in -> silence out
        out = nr.reduce_noise(y=audio, sr=sr, stationary=True)
        return _fit_length(out, len(audio))


class DeepFilterNet:
    """DeepFilterNet3 general non-stationary denoiser. Operates at 48 kHz internally."""

    chain_id = "deepfilternet"

    def __init__(self) -> None:
        self._model = None
        self._state = None

    def _ensure(self):
        if self._model is None:
            from df.enhance import init_df  # noqa: PLC0415

            self._model, self._state, _ = init_df()
        return self._model, self._state

    def process(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        import torch  # noqa: PLC0415
        import torchaudio.functional as AF  # noqa: PLC0415
        from df.enhance import enhance  # noqa: PLC0415

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not np.any(audio):
            return audio
        model, state = self._ensure()
        df_sr = state.sr()
        t = torch.from_numpy(audio).unsqueeze(0)
        t48 = AF.resample(t, sr, df_sr)
        enhanced = enhance(model, state, t48)
        out = AF.resample(enhanced, df_sr, sr).squeeze(0).numpy()
        return _fit_length(out, len(audio))


class DemucsVocals:
    """Demucs vocal-stem separation — for babble (AC/CA). Heavy; usually blows CPU budget."""

    chain_id = "demucs_vocals"

    def __init__(self) -> None:
        self._model = None

    def _ensure(self):
        if self._model is None:
            from demucs.pretrained import get_model  # noqa: PLC0415

            self._model = get_model("htdemucs")
            self._model.eval()
        return self._model

    def process(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        import torch  # noqa: PLC0415
        import torchaudio.functional as AF  # noqa: PLC0415
        from demucs.apply import apply_model  # noqa: PLC0415

        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not np.any(audio):
            return audio
        model = self._ensure()
        model_sr = model.samplerate
        t = torch.from_numpy(audio).unsqueeze(0)
        stereo = AF.resample(t, sr, model_sr).repeat(2, 1).unsqueeze(0)  # (1, 2, T)
        with torch.no_grad():
            stems = apply_model(model, stereo, split=True, overlap=0.1)[0]
        vocals = stems[model.sources.index("vocals")].mean(0, keepdim=True)
        out = AF.resample(vocals, model_sr, sr).squeeze(0).numpy()
        return _fit_length(out, len(audio))


# Registry: chain_id -> zero-arg factory (heavy backends construct lazily).
_FACTORIES = {
    "none": NoneDenoiser,
    "spectral_gate": SpectralGate,
    "deepfilternet": DeepFilterNet,
    "demucs_vocals": DemucsVocals,
}

_DEP = {
    "none": [],
    "spectral_gate": ["noisereduce"],
    "deepfilternet": ["df"],
    "demucs_vocals": ["demucs"],
}


def registered_chains() -> list[str]:
    return list(_FACTORIES)


def _importable(mod: str) -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec(mod) is not None


def available_chains() -> list[str]:
    """Chains whose optional dependencies are importable in this environment."""
    return [c for c, deps in _DEP.items() if all(_importable(m) for m in deps)]


def get_denoiser(chain_id: str) -> Denoiser:
    if chain_id not in _FACTORIES:
        raise ValueError(f"unknown chain: {chain_id!r} (known: {registered_chains()})")
    return _FACTORIES[chain_id]()
