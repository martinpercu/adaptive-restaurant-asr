from __future__ import annotations

from tests.conftest import speechlike

from ars.preprocess.classifier import (
    CLEAN,
    Classifier,
    build_model,
    classes_from_subtypes,
    log_mel_windows,
)

SR = 16000


def test_classes_order():
    cs = classes_from_subtypes(["BB", "AA", "AC"])
    assert cs == ["AA", "AC", "BB", CLEAN]  # sorted subtypes, CLEAN last


def test_log_mel_window_shape():
    feats = log_mel_windows(speechlike(dur=2.0, amp=0.2), SR)
    # 2 s with 1 s windows / 0.5 s hop -> 3 windows; (n, 1, 64, frames)
    assert feats.ndim == 4 and feats.shape[1] == 1 and feats.shape[2] == 64
    assert feats.shape[0] == 3


def test_model_forward_and_predict():
    classes = classes_from_subtypes(["AA", "BB"])
    model = build_model(len(classes))
    clf = Classifier(model, classes, min_confidence=0.6)
    pred = clf.predict(speechlike(dur=1.5, amp=0.2), SR)
    assert pred.subtype in {*classes[:-1], None}  # a subtype or None (never CLEAN string)
    assert 0.0 <= pred.confidence <= 1.0


def test_model_under_5mb():
    import torch  # noqa: PLC0415

    model = build_model(8)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params * 4 < 5 * 1024 * 1024  # float32 weights < 5 MB
    _ = torch
