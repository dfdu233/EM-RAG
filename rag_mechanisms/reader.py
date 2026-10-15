"""Frozen Reader and EXACT upstream MedRGAG prompt for controlled conditions.

Does not import or instantiate MedRGAG, EM-RAG, a trained Gate, or any
RAG-specific policy. The template is extracted by safe Python AST parsing.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import json
from typing import Any

from .data import Passage, Question


def load_medrgag_rag_template(template_py: str | Path) -> str:
    """Safely extract MedRGAG src/template.py::rag_template (not exec/import)."""
    path = Path(template_py)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "rag_template" for t in node.targets
        ):
            template = ast.literal_eval(node.value)
            if not isinstance(template, str) or "{documents}" not in template or "{question}" not in template:
                raise ValueError("Invalid upstream rag_template contents")
            return template
    raise ValueError("MedRGAG src/template.py does not define rag_template")


@dataclass(frozen=True)
class Prompt:
    text: str
    document_ids: tuple[str, ...]
    context_chars: int
    prompt_chars: int
    prompt_tokens: int | None
    sha256: str


class PromptBuilder:
    def __init__(self, template: str, tokenizer_path: str | None = None,
                 max_input_tokens: int = 24576, max_prompt_chars: int = 100000):
        self.template = template
        self.max_input_tokens = max_input_tokens
        self.max_prompt_chars = max_prompt_chars
        if max_input_tokens < 1 or max_prompt_chars < 1:
            raise ValueError("Positive prompt budgets required")
        self.tokenizer = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("Install transformers to count exact model tokens") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=False)

    def build(self, question: Question, docs: tuple[Passage, ...]) -> Prompt:
        contexts = [
            f"Document [{idx}]: {' '.join(doc.text.split())}"
            for idx, doc in enumerate(docs)
        ]
        joined = "\n".join(contexts)
        # Keep exact upstream MedRGAG question+options formatting.
        options = "".join(
            f"{k}: {v}\n" for k, v in question.options.items())
        full = self.template.format(
            documents=joined, question=question.text + "\n" + options)
        if len(full) > self.max_prompt_chars:
            raise ValueError(f"prompt_exceeds_char_budget:{len(full)}")
        tokens = None
        if self.tokenizer is not None:
            tokens = len(self.tokenizer.encode(full, add_special_tokens=False))
            if tokens > self.max_input_tokens:
                raise ValueError(f"prompt_exceeds_token_budget:{tokens}")
        return Prompt(
            text=full, document_ids=tuple(doc.id for doc in docs),
            context_chars=len(joined), prompt_chars=len(full),
            prompt_tokens=tokens, sha256=sha256(full.encode("utf-8")).hexdigest())


def parse_answer(text: str, choices: tuple[str, ...]) -> str | None:
    """Strict candidate extraction: do NOT default malformed output to A.

    Upstream MedRGAG's evaluate_utils.locate_answer defaults to A; that is
    unsuitable for a neutral mechanism study due to label-class asymmetry.
    """
    if not text or not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    data: dict[str, Any] | None = None
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            maybe, _ = decoder.raw_decode(text[i:])
            if isinstance(maybe, dict):
                data = maybe
                break
        except json.JSONDecodeError:
            continue
    if data is not None:
        candidate = str(data.get("answer_choice", data.get("choice", ""))).strip().upper()
        return candidate if candidate in choices else None
    # A few Reader implementations output a single letter instead of JSON.
    stripped = text.strip().upper()
    return stripped if stripped in choices else None


class FrozenReader:
    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.1,
                 max_output_tokens: int = 512, seed: int | None = 42):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("pip install openai or pip install -e '.[api]'") from exc
        kwargs: dict[str, Any] = {"max_retries": 2, "timeout": 180.0}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        elif base_url and any(x in base_url for x in ("localhost", "127.0.0.1")):
            kwargs["api_key"] = "EMPTY"
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.calls = 0

    def answer(self, prompt: Prompt, choices: tuple[str, ...]) -> dict:
        self.calls += 1
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.text},
            ],
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        result = self.client.chat.completions.create(**kwargs)
        output = result.choices[0].message.content or ""
        usage = getattr(result, "usage", None)
        return {
            "prediction": parse_answer(output, choices),
            "raw_output": output,
            "prompt_tokens_billed": (getattr(usage, "prompt_tokens", None)
                                     if usage is not None else None),
            "completion_tokens_billed": (getattr(usage, "completion_tokens", None)
                                         if usage is not None else None),
        }


def model_parameters(reader: FrozenReader) -> dict:
    return {"model": reader.model, "temperature": reader.temperature,
            "max_output_tokens": reader.max_output_tokens, "seed": reader.seed}
