"""Frozen OpenAI-compatible LLMs and optional original MedCPT reranker."""
from __future__ import annotations

import json
import re
from typing import Protocol

from .data import Passage, Question


class ChatModel(Protocol):
    calls: int
    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 seed: int | None = None, max_tokens: int = 512) -> str: ...


class OpenAIChat:
    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 180.0,
                 send_seed: bool = True, send_temperature: bool = True):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("pip install -e '.[api]' for OpenAI-compatible inference") from exc
        if not api_key and base_url and any(
                x in base_url for x in ("localhost", "127.0.0.1")):
            api_key = "EMPTY"
        kwargs: dict = {"timeout": timeout, "max_retries": 2}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
        self.model = model
        self.calls = 0
        self.send_seed = send_seed
        self.send_temperature = send_temperature

    def complete(self, system: str, user: str, *, temperature: float = 0.0,
                 seed: int | None = None, max_tokens: int = 512) -> str:
        options = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.send_temperature:
            options["temperature"] = temperature
        if self.send_seed and seed is not None:
            options["seed"] = seed
        self.calls += 1
        result = self.client.chat.completions.create(**options)
        return result.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Strict about permitted fields, forgiving of control chars in JSON strings.

    This parser NEVER uses gold labels and never defaults a failed parse to A.
    """
    decoder = json.JSONDecoder(strict=False)
    for offset, c in enumerate(text):
        if c != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No parseable JSON object")


def parse_answer(text: str, labels: tuple[str, ...]) -> str | None:
    try:
        value = parse_json(text)
    except ValueError:
        return None
    selected = value.get("answer_choice", value.get("choice", None))
    if not isinstance(selected, str):
        return None
    label = selected.strip().upper()
    return label if label in labels else None


def parse_selection(text: str, candidate_count: int, limit: int = 5
                    ) -> tuple[int, ...]:
    """Parse original KADS Final Selection list or JSON indexes."""
    raw: list | None = None
    try:
        obj = parse_json(text)
        for key in ("selected_indices", "final_list", "selected"):
            if isinstance(obj.get(key), list):
                raw = obj[key]
                break
    except ValueError:
        pass
    if raw is None:
        # Validate the WHOLE KADS selection line, not a valid prefix:
        # otherwise "Final Selection: [0] [INVALID]" silently becomes [0].
        match = re.search(
            r"Final\s+Selection\s*:\s*([^\r\n]+)",
            text, flags=re.IGNORECASE)
        if match:
            line = match.group(1).strip()
            if not re.fullmatch(r"(?:\[\s*\d+\s*\]\s*)+", line):
                raise ValueError("Malformed KADS selection line")
            raw = [int(x) for x in re.findall(r"\d+", line)]
    if (not isinstance(raw, list) or not 1 <= len(raw) <= limit
            or any(type(x) is not int or x < 0 or x >= candidate_count for x in raw)
            or len(set(raw)) != len(raw)):
        raise ValueError("KADS selector returned invalid or out-of-range indices")
    return tuple(raw)


class MedCPTReranker:
    """Reproduce MedRGAG's selected-passage cross-encoder ranking.

    Use ncbi/MedCPT-Cross-Encoder (or a locally mirrored copy) on the
    SAME model weights as the MedRGAG paper baseline.
    """

    def __init__(self, model_name: str = "ncbi/MedCPT-Cross-Encoder",
                 device: str | None = None, max_length: int = 512):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("Install PyTorch and transformers for MedCPT reranking") from exc
        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.model_name = model_name
        self.calls = 0

    def rank(self, q: Question, docs: list[Passage]) -> list[Passage]:
        if not docs:
            return []
        query = q.for_reranker()
        scores: list[float] = []
        for i in range(0, len(docs), 8):
            subset = docs[i:i + 8]
            pairs = [[query, doc.text] for doc in subset]
            encoded = self.tokenizer(
                pairs, return_tensors="pt", truncation=True, padding=True,
                max_length=self.max_length)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded).logits.reshape(-1).detach().cpu().tolist()
            scores.extend(float(value) for value in logits)
            self.calls += 1
        permutation = sorted(range(len(docs)), key=lambda i: -scores[i])
        return [docs[i] for i in permutation]


class IdentityRanker:
    """Explicit ablation only. NOT an exact MedRGAG reproduction."""
    model_name = "none"
    def __init__(self):
        self.calls = 0

    def rank(self, q: Question, docs: list[Passage]) -> list[Passage]:
        self.calls += 1
        return list(docs)
