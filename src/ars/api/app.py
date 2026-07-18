"""FastAPI transcription service (plan/02-architecture.md §3).

Endpoints:
  POST /v1/transcribe  — multipart WAV (`file`) or JSON {audio_b64, store_id, meta}
  GET  /healthz        — liveness
  GET  /v1/model       — current production registry entry
  GET  /metrics        — rolling latency/confidence stats

The heavy `Pipeline` (Whisper model) is built lazily on first use; tests override
`get_pipeline` with a fake so CI never loads a real model.
"""

from __future__ import annotations

import base64
import io
from collections import deque
from functools import lru_cache
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, Request

from ars.asr.engine import WhisperEngine
from ars.asr.prompt_builder import load_menu
from ars.config import Settings
from ars.contracts import FinalTranscript
from ars.pipeline import Pipeline
from ars.preprocess import build_preprocessor
from ars.registry import ModelRegistry
from ars.vad import SileroVad

SR = 16000
_RECENT: deque[dict] = deque(maxlen=500)


def _base_model_to_size(base_model: str, fallback: str) -> str:
    return base_model[len("whisper-") :] if base_model.startswith("whisper-") else fallback


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    settings = get_settings()
    reg = ModelRegistry.load(settings.paths.models + "/registry.json")
    prod = reg.production()
    size = (
        _base_model_to_size(prod.base_model, settings.asr.model_size)
        if prod
        else settings.asr.model_size
    )
    engine = WhisperEngine(settings.asr, model=size)
    menu = load_menu(settings.keydetector.menu_dir, "demo")
    return Pipeline(
        settings=settings,
        vad=SileroVad(settings.vad),
        engine=engine,
        preprocess=build_preprocessor(settings),
        menu=menu,
        model_version=prod.version if prod else None,
    )


app = FastAPI(title="ARS", version="0.1.0")


def _read_wav(data: bytes) -> np.ndarray:
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"cannot decode audio: {exc}") from exc
    if sr != SR:
        raise HTTPException(status_code=400, detail=f"audio must be {SR} Hz, got {sr}")
    return audio.mean(axis=1).astype(np.float32)


async def _extract(request: Request) -> tuple[bytes, str | None, dict[str, Any]]:
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="missing 'file'")
        data = await upload.read()
        store_id = form.get("store_id")
        return data, store_id, {}
    body = await request.json()
    if "audio_b64" not in body:
        raise HTTPException(status_code=400, detail="missing 'audio_b64'")
    return base64.b64decode(body["audio_b64"]), body.get("store_id"), body.get("meta") or {}


@app.post("/v1/transcribe", response_model=FinalTranscript)
async def transcribe(
    request: Request, pipeline: Pipeline = Depends(get_pipeline)
) -> FinalTranscript:
    data, store_id, meta = await _extract(request)
    audio = _read_wav(data)
    result = pipeline.transcribe(audio, SR, store_id=store_id, meta=meta)
    _RECENT.append(
        {
            "total_ms": result.trace.latency_ms.total,
            "guarded": bool(result.trace.guard_flags),
            "language": result.language,
        }
    )
    return result


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/model")
async def model(settings: Settings = Depends(get_settings)) -> dict:
    prod = ModelRegistry.load(settings.paths.models + "/registry.json").production()
    if prod is None:
        raise HTTPException(status_code=404, detail="no production model registered")
    return prod.model_dump(mode="json")


@app.get("/metrics")
async def metrics() -> dict:
    if not _RECENT:
        return {"n": 0}
    lat = sorted(r["total_ms"] for r in _RECENT)
    n = len(lat)
    return {
        "n": n,
        "latency_ms_p50": lat[n // 2],
        "latency_ms_p95": lat[min(n - 1, int(0.95 * n))],
        "guarded_rate": round(sum(r["guarded"] for r in _RECENT) / n, 3),
    }
