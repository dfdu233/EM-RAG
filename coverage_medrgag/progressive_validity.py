"""PSV v2 input/output reliability primitives.

No gold labels, learned policies, or extra LLM calls. All query changes and
answer recovery methods are recorded to separate protocol repair from science.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .models import parse_answer, parse_json

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*|[\u4e00-\u9fff]+")
_CONDITION = frozenset({
    "after", "adjust", "adjusted", "adjustment", "controlling", "controlled",
    "control", "independent", "conditional", "without", "negative", "not",
    "unless", "versus", "compared", "comparison", "confounding", "confounder",
    "multivariate", "multivariable", "subgroup", "hazard", "risk", "ratio",
    "trial", "randomized", "placebo", "significant", "significance",
})
_NOISE = frozenset({
    "the", "is", "was", "are", "were", "of", "and", "or", "to", "in",
    "for", "that", "with", "a", "an", "as", "at", "on", "from",
    "which", "what", "does", "do", "did", "can", "would", "should",
})
_SPACE = re.compile(r"[ \t\r\n]+")


@dataclass(frozen=True)
class PlannedQuery:
    query: str | None
    status: str
    original_length: int
    final_length: int
    original_query: str | None


def _terms(value: str) -> list[str]:
    return _WORD.findall(value)


def fit_query(query: str, *, relation: str, question: str,
              max_chars: int = 300, strategy: str = "compact"
              ) -> PlannedQuery:
    """Hard interface limit, explicitly trace lossy deterministic repair.

    The BM25 bag-of-words rewrite is NOT guaranteed semantically equivalent.
    Keeping original-relation and condition words early is a safeguard, not
    medical entailment. Reject mode permits exact historical behavior.
    """
    if strategy not in {"reject", "compact"}:
        raise ValueError("Unsupported query-size strategy")
    if not isinstance(query, str):
        return PlannedQuery(None, "non_string_query", 0, 0, None)
    raw = query
    clean = _SPACE.sub(" ", query).strip()
    if len(clean) < 5:
        return PlannedQuery(None, "too_short", len(raw), 0, raw)
    if len(clean) <= max_chars:
        return PlannedQuery(clean, "unchanged", len(raw), len(clean), raw)
    if strategy == "reject":
        return PlannedQuery(None, "over_limit_rejected", len(raw), 0, raw)

    # Do not truncate long medical terms or silently remove conditional
    # qualifiers that occur near the end of verbose generated queries.
    qualifiers = [t for t in _terms(clean) if t.casefold() in _CONDITION]
    ordered = (_terms(relation)
               + [t for t in _terms(question)
                  if t.casefold() not in _NOISE]
               + qualifiers + _terms(clean))
    output: list[str] = []
    seen: set[str] = set()
    size = 0
    for token in ordered:
        key = token.casefold()
        if key in seen or (key in _NOISE and key not in _CONDITION):
            continue
        add = len(token) + (1 if output else 0)
        if size + add > max_chars:
            continue
        output.append(token)
        seen.add(key)
        size += add
    short = " ".join(output)
    if len(short) < 5:
        return PlannedQuery(None, "compact_failed", len(raw), 0, raw)
    return PlannedQuery(short, "compacted_unverified", len(raw),
                        len(short), raw)


@dataclass(frozen=True)
class AnswerParse:
    prediction: str | None
    method: str


def _explicit_choice(value: str, labels: tuple[str, ...]) -> str | None:
    stripped = value.strip().upper()
    if stripped in labels:
        return stripped
    match = re.fullmatch(r"(?i)(?:option|choice|answer)\s+([A-Z])",
                         value.strip())
    if match and match.group(1).upper() in labels:
        return match.group(1).upper()
    return None


def recover_answer(raw: str, labels: tuple[str, ...]) -> AnswerParse:
    """Uniform conservative answer parser for old reference and new views.

    Only explicit answer fields are eligible. Never infer answer from
    rationale, retrieved facts, confidence, citations, or option lists.
    Conflicting explicit choice labels are deliberately unparseable.
    """
    try:
        obj = parse_json(raw)
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        candidates: list[str] = []
        for key in ("answer_choice", "choice", "final_answer", "answer", "option"):
            value = obj.get(key)
            if isinstance(value, str):
                choice = _explicit_choice(value, labels)
                if choice is not None:
                    candidates.append(choice)
        unique = set(candidates)
        if len(unique) > 1:
            return AnswerParse(None, "ambiguous_json_fields")
        if len(unique) == 1:
            label = next(iter(unique))
            old = parse_answer(raw, labels)
            return AnswerParse(label, "original_json" if old == label
                               else "alternate_json_field")

    patterns = [
        r'["\x27]?(?:answer_choice|final_answer)["\x27]?\s*:\s*["\x27]?\s*([A-Za-z])\b',
        r'(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\s+choice)?(?:\*\*)?\s*[:：=-]\s*(?:\*\*)?\s*(?:option\s+)?([A-Za-z])\b',
        r'\\boxed\{\s*([A-Za-z])\s*\}',
        r'<answer>\s*([A-Za-z])\s*</answer>',
    ]
    labels_found = {
        match.group(1).upper()
        for pattern in patterns
        for match in re.finditer(pattern, raw)
        if match.group(1).upper() in labels
    }
    if len(labels_found) == 1:
        return AnswerParse(next(iter(labels_found)), "explicit_field_recovered")
    return AnswerParse(None, "ambiguous_explicit_answers" if len(labels_found) > 1
                       else "unparseable")


def view_provenance(anchors: Sequence, new_docs: Sequence,
                    reference_sources: Sequence) -> dict:
    """Untrusted passage provenance; no attempt to infer factual truth."""
    return {
        "anchor_count": len(anchors),
        "anchor_generated_count": sum(
            p.origin == "parametric_generated" for p in anchors),
        "fresh_count": len(new_docs),
        "fresh_generated_count": sum(
            p.origin == "parametric_generated" for p in new_docs),
        "anchor_ids": [p.id for p in anchors],
        "fresh_ids": [p.id for p in new_docs],
        "reference_generated_count": sum(
            p.origin == "parametric_generated" for p in reference_sources),
    }
