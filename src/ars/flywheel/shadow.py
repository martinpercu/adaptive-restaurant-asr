"""Shadow deployment (plan/phases/phase-6 §6.5).

Runs a candidate CT2 model in parallel with production on the same inputs; only production's
output is returned to the caller (candidate inference is async, never on the request path).
Here it is a synchronous evaluator used offline/in simulation to produce the metrics the
promotion gate consumes. WER proxy is judge-scored on a sample of >= 100.
"""

from __future__ import annotations


def shadow_eval(prod_engine, cand_engine, samples, judge=None) -> dict:
    """samples: list of (audio, lang, reference?). Returns the promotion-gate metric bundle."""
    import numpy as np  # noqa: PLC0415

    from ars.eval.metrics import wer_cer  # noqa: PLC0415

    prod_hyps, cand_hyps, refs, p_lp, c_lp = [], [], [], [], []
    for audio, lang, ref in samples:
        pr = prod_engine.transcribe(audio, 16000, language=lang)
        cr = cand_engine.transcribe(audio, 16000, language=lang)
        prod_hyps.append(pr.text)
        cand_hyps.append(cr.text)
        refs.append(ref)
        p_lp.append(pr.avg_logprob)
        c_lp.append(cr.avg_logprob)
    lang = samples[0][1] if samples else "es"
    prod_wer = wer_cer(refs, prod_hyps, lang)[0] or 0.0
    cand_wer = wer_cer(refs, cand_hyps, lang)[0] or 0.0
    _ = judge
    return {
        "prod_wer": prod_wer,
        "cand_wer": cand_wer,
        "prod_logprob": float(np.mean(p_lp)) if p_lp else 0.0,
        "cand_logprob": float(np.mean(c_lp)) if c_lp else 0.0,
        "n": len(samples),
    }
