"""Noise subtype classifier (plan/phases/phase-3 §3.1).

Small log-mel CNN. Predicts one of the taxonomy subtypes or CLEAN from speech+noise
audio (it never sees noise alone in production). Utterance prediction = mean of window
logits. A confidence gate maps low-confidence predictions to `None` (treated as clean —
no mitigation), because misclassifying clean audio triggers pointless denoising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
CLEAN = "CLEAN"
N_MELS = 64
WIN_S = 1.0
HOP_S = 0.5
_N_FFT = 400
_HOP = 160


@dataclass
class NoisePrediction:
    subtype: str | None  # taxonomy subtype code, or None (clean / low-confidence)
    confidence: float


def classes_from_subtypes(subtypes: list[str]) -> list[str]:
    """Deterministic class order: sorted subtypes then CLEAN (last index)."""
    return sorted(subtypes) + [CLEAN]


def decide(mean_logits: np.ndarray, classes: list[str], min_confidence: float) -> NoisePrediction:
    """Pure decision from mean window logits -> subtype or None (03 §3.1 confidence gate)."""
    z = mean_logits - np.max(mean_logits)
    probs = np.exp(z) / np.sum(np.exp(z))
    i = int(np.argmax(probs))
    conf = float(probs[i])
    label = classes[i]
    if label == CLEAN or conf < min_confidence:
        return NoisePrediction(None, conf)
    return NoisePrediction(label, conf)


def log_mel_windows(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """(n_windows, 1, N_MELS, frames) log-mel features over 1 s windows, 0.5 s hop."""
    import torch  # noqa: PLC0415
    import torchaudio.transforms as T  # noqa: PLC0415

    audio = np.ascontiguousarray(audio, dtype=np.float32)
    win = int(WIN_S * sr)
    hop = int(HOP_S * sr)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    starts = list(range(0, len(audio) - win + 1, hop)) or [0]
    mel = T.MelSpectrogram(sample_rate=sr, n_fft=_N_FFT, hop_length=_HOP, n_mels=N_MELS)
    feats = []
    for s in starts:
        w = torch.from_numpy(audio[s : s + win])
        m = mel(w)
        feats.append(torch.log(m + 1e-6).unsqueeze(0))
    return torch.stack(feats).numpy()


def build_model(n_classes: int):
    """~4-conv-block CNN, < 5 MB. Returns an nn.Module."""
    import torch.nn as nn  # noqa: PLC0415

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    class NoiseCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, n_classes)
            )

        def forward(self, x):
            return self.head(self.features(x))

    return NoiseCNN()


class Classifier:
    """Inference wrapper: window features -> mean logits -> confidence-gated prediction."""

    def __init__(self, model, classes: list[str], min_confidence: float = 0.6) -> None:
        self.model = model
        self.classes = classes
        self.min_confidence = min_confidence

    def mean_logits(self, audio: np.ndarray, sr: int = SR) -> np.ndarray:
        import torch  # noqa: PLC0415

        feats = log_mel_windows(audio, sr)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(feats)).numpy()
        return logits.mean(axis=0)

    def predict(self, audio: np.ndarray, sr: int = SR) -> NoisePrediction:
        return decide(self.mean_logits(audio, sr), self.classes, self.min_confidence)

    # --- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        import torch  # noqa: PLC0415

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "classes": self.classes,
                "min_confidence": self.min_confidence,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, min_confidence: float | None = None) -> Classifier:
        import torch  # noqa: PLC0415

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(len(ckpt["classes"]))
        model.load_state_dict(ckpt["state_dict"])
        return cls(
            model,
            ckpt["classes"],
            min_confidence if min_confidence is not None else ckpt["min_confidence"],
        )
