"""Judge client (plan/03-data-spec.md §8).

Pluggable provider (Anthropic reference `claude-sonnet-5`; OpenAI `gpt-4o-mini` /
`gpt-4.1-nano` alternates). Structured outputs enforced, pydantic-validated, one retry
on invalid. The weekly cycle submits a batch. `MockJudge` serves all tests — no test hits
the network for any provider. The judge is a labeler, not an oracle (see routing rules).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import ValidationError

from ars.contracts import JudgeVerdict

# The exact JSON schema the provider must return (03 §8).
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["correct", "minor_errors", "wrong", "hallucination"]},
        "corrected_reference": {"type": ["string", "null"]},
        "confusion_candidates": {"type": "array"},
        "order_core_match": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "order_core_match", "confidence"],
}

JUDGE_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}


class JudgeClient(ABC):
    @abstractmethod
    def judge_one(self, request: dict) -> JudgeVerdict: ...

    def judge_batch(self, requests: list[dict]) -> list[JudgeVerdict]:
        return [self.judge_one(r) for r in requests]


class MockJudge(JudgeClient):
    """Scripted verdicts. `script` maps a transcript to a verdict dict, or is a callable."""

    def __init__(
        self, script: dict | Callable[[dict], dict] | None = None, default: dict | None = None
    ) -> None:
        self.script = script or {}
        self.default = default or {
            "verdict": "correct",
            "corrected_reference": None,
            "confusion_candidates": [],
            "order_core_match": True,
            "confidence": 0.95,
        }
        self.calls = 0

    def judge_one(self, request: dict) -> JudgeVerdict:
        self.calls += 1
        if callable(self.script):
            data = self.script(request)
        else:
            data = self.script.get(request.get("transcript", ""), self.default)
        return JudgeVerdict.model_validate(data)


class _ProviderJudge(JudgeClient):
    """Shared structured-output + validate + one-retry logic. Transport is injectable."""

    provider = "base"

    def __init__(self, model: str, transport: Callable[[dict], str] | None = None) -> None:
        self.model = model
        self._transport = transport  # (request_payload) -> raw JSON string; None = real API

    def build_request(self, request: dict) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def judge_one(self, request: dict) -> JudgeVerdict:
        payload = self.build_request(request)
        for attempt in range(2):  # validate, one retry
            raw = self._call(payload)
            try:
                return JudgeVerdict.model_validate_json(raw)
            except (ValidationError, ValueError):
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable")

    def _call(self, payload: dict) -> str:
        if self._transport is not None:
            return self._transport(payload)
        return self._real_call(payload)  # pragma: no cover - network

    def _real_call(self, payload: dict) -> str:  # pragma: no cover - network
        raise NotImplementedError("real provider call; inject transport in tests")


class AnthropicJudge(_ProviderJudge):
    provider = "anthropic"

    def build_request(self, request: dict) -> dict:
        # Sonnet 5: output_config.format json_schema; do NOT set temperature/top_p/top_k.
        return {
            "model": self.model,
            "max_tokens": 512,
            "output_config": {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            "messages": [{"role": "user", "content": _prompt(request)}],
        }


class OpenAIJudge(_ProviderJudge):
    provider = "openai"

    def build_request(self, request: dict) -> dict:
        # response_format json_schema strict:true, temperature 0.
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": VERDICT_SCHEMA},
            },
            "messages": [{"role": "user", "content": _prompt(request)}],
        }


def _prompt(request: dict) -> str:
    return (
        "You are a strict ASR transcription judge for a restaurant drive-thru.\n"
        f"lang: {request.get('lang')}\n"
        f"transcript: {request.get('transcript')!r}\n"
        f"menu_items: {json.dumps(request.get('menu_items', [])[:60], ensure_ascii=False)}\n"
        f"pos_ticket: {request.get('pos_ticket')}\n"
        "Return the verdict JSON."
    )


def route_verdict(verdict: JudgeVerdict, min_confidence: float = 0.8) -> str:
    """Routing table (03 §8 / phase-6 §6.2): where a judged item goes."""
    if verdict.verdict == "correct" and verdict.order_core_match:
        return "training"
    if (
        verdict.verdict == "minor_errors"
        and verdict.corrected_reference
        and verdict.confidence >= min_confidence
    ):
        return "training+miner"
    return "review"


def build_judge(settings, transport: Callable[[dict], str] | None = None) -> JudgeClient:
    provider = settings.judge.provider
    model = settings.judge.model or JUDGE_MODELS.get(provider)
    if provider == "anthropic":
        return AnthropicJudge(model, transport)
    if provider == "openai":
        return OpenAIJudge(model, transport)
    raise ValueError(f"unknown judge provider: {provider}")
