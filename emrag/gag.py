"""Read MedRGAG's PRECOMPUTED generative knowledge passages.

These are parametric model outputs, not externally verified facts. They may
help the Reader generate a correct candidate, but are NEVER placed in the
EvidenceLedger or shown to the evidence-only Judge as support.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def load_generated_snippets(path: str | Path) -> dict[str, list[str]]:
    """Expected MedRGAG outputs/<dataset>/docs/generate/snippets.json format.

    {"question_id": {"snippets": [{"id": "...", "contents": "..."}]}}
    No ground-truth answers, predictions or rationales are loaded.
    """
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Expected MedRGAG generated-snippet JSON object")
    result: dict[str, list[str]] = {}
    for qid, value in payload.items():
        if not isinstance(value, dict) or not isinstance(value.get("snippets"), list):
            continue
        passages: list[str] = []
        seen: set[str] = set()
        for entry in value["snippets"]:
            if not isinstance(entry, dict):
                continue
            body = entry.get("contents")
            if not isinstance(body, str):
                continue
            body = body.strip()
            if not body or len(body) > 20_000:
                continue
            dedup_key = re.sub(r"\s+", " ", body).casefold()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            passages.append(body)
        result[str(qid)] = passages
    return result


def bounded_background(passages: list[str], *, max_docs: int = 3,
                       max_chars: int = 1500) -> list[str]:
    """Bound total background size without changing external evidence context.

    We cap each generated passage as well as total chars to prevent one
    long hallucinated passage from dominating the Reader.
    """
    if max_docs < 1 or max_chars < 1:
        raise ValueError("max_docs and max_chars must be positive")
    output: list[str] = []
    remaining = max_chars
    # Spread a fixed budget over multiple candidate knowledge passages.
    per_doc = max(1, max_chars // max_docs)
    for body in passages:
        if len(output) >= max_docs or remaining <= 0:
            break
        if not isinstance(body, str) or not body.strip():
            continue
        excerpt = body.strip()[:min(per_doc, remaining)]
        if excerpt:
            output.append(excerpt)
            remaining -= len(excerpt)
    return output
