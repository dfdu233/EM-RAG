"""Frozen model adapters; no model or gate training."""
from __future__ import annotations
import json
from typing import Protocol


class Model(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str: ...


def parse_json(text: str) -> dict:
    body = text.strip()
    if body.startswith("~~~") or body.startswith(chr(96) * 3):
        body = "\n".join(body.splitlines()[1:])
    decoder = json.JSONDecoder()
    for index, char in enumerate(body):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(body[index:])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("model response did not contain a JSON object")


class OpenAICompatibleModel:
    """Supports an OpenAI-compatible vLLM server or an external API."""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 90.0):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install optional API dependency: pip install -e '.[api]'") from exc
        kwargs: dict = {"timeout": timeout, "max_retries": 2}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls += 1
        result = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return result.choices[0].message.content or ""


class ScriptedModel:
    """Deterministic fixture backend for tests, not a medical model."""

    def __init__(self, replies: list[str]):
        self.replies = iter(replies)
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls += 1
        return next(self.replies)


def request_json(model: Model, system: str, user: str) -> dict:
    return parse_json(model.complete(system, user, temperature=0.0))
