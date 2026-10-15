"""Pre-specified interventions for general RAG scaling, not EM-RAG ablations.

Primary factors:
- closed-book vs incremental retrieval depth on ORIGINAL ranked MedRGAG corpus;
- external retrieval vs parametric generated knowledge (where cache exists);
- out-of-query same-length evidence controls for saturation/distraction.

No test gold answers or model predictions are used to construct contexts.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from statistics import mean

from .data import Passage, Question


@dataclass(frozen=True)
class Condition:
    name: str
    passages: tuple[Passage, ...]
    family: str
    expected_k: int
    source_type: str
    note: str = ""


def _noise_tail(question_id: str, target: list[Passage],
                donor_cache: dict[str, list[Passage]],
                seed: int) -> list[Passage] | None:
    """Cross-question text-length negative control, selected without gold.

    A donor may coincidentally contain relevant facts: this is a deliberately
    IMPERFECT stress control, not a semantically certified irrelevant document.
    """
    result = []
    used: set[tuple[str, str]] = set()
    for position, goal in enumerate(target):
        options = []
        for donor_qid, documents in donor_cache.items():
            if donor_qid == question_id:
                continue
            for doc in documents:
                if len(doc.text) < len(goal.text):
                    continue
                if doc.text.strip() == goal.text.strip():
                    continue
                if (donor_qid, doc.id) not in used:
                    options.append((donor_qid, doc))
        if not options:
            return None
        options.sort(key=lambda item: sha256(
            f"{seed}|{question_id}|{position}|{item[0]}|{item[1].id}".encode()
        ).hexdigest())
        qid, doc = options[0]
        used.add((qid, doc.id))
        result.append(Passage(
            id=f"NOISE_FROM_OTHER_Q_{qid}_{doc.id}",
            text=doc.text[:len(goal.text)],
            title=doc.title, source="OTHER_QUESTION_CACHE_CONTROL",
        ))
    return result


def build_conditions(qid: str, ranked: list[Passage],
                     generated: list[Passage] | None,
                     ranked_all: dict[str, list[Passage]],
                     *, depths: tuple[int, ...] = (0, 1, 3, 5, 10, 20, 32),
                     include_generation: bool = False,
                     include_negative_control: bool = False,
                     seed: int = 2026) -> list[Condition]:
    if not depths or 0 not in depths or any(k < 0 for k in depths):
        raise ValueError("Depths must include 0 and cannot be negative")
    outputs = []
    for k in sorted(set(depths)):
        if k <= len(ranked):
            outputs.append(Condition(
                name=f"retrieved_k{k}", family="depth",
                passages=tuple(ranked[:k]), expected_k=k,
                source_type="external_retrieved"))
    if include_generation:
        gen = generated or []
        if len(gen) >= 5:
            outputs.append(Condition(
                "generated_k5", tuple(gen[:5]), "generation", 5,
                "parametric_generated"))
            if len(ranked) >= 3:
                outputs.append(Condition(
                    "retrieved3_generated2", tuple(ranked[:3] + gen[:2]),
                    "generation", 5, "mixed_external_parametric",
                    "NOT identical to the full MedRGAG KADS selection policy"))
    if include_negative_control and len(ranked) >= 8:
        # Top 3 facts are pinned; test whether low-ranked *own* evidence helps
        # more than a same-length unrelated-sample replacement (5 docs total).
        base = ranked[:3]
        tail = ranked[3:5]
        donor = _noise_tail(qid, tail, ranked_all, seed)
        outputs.append(Condition(
            "prefix3_own_tail2", tuple(base + tail), "stress", 5,
            "external_retrieved"))
        if donor is not None:
            outputs.append(Condition(
                "prefix3_other_question_tail2", tuple(base + donor),
                "stress", 5, "negative_control",
                "Donor text matched in characters, NOT certified semantically irrelevant"))
        own_longer = ranked[3:8]
        donor_longer = _noise_tail(qid, own_longer, ranked_all, seed)
        outputs.append(Condition(
            "prefix3_own_tail5", tuple(base + own_longer), "stress", 8,
            "external_retrieved"))
        if donor_longer is not None:
            outputs.append(Condition(
                "prefix3_other_question_tail5", tuple(base + donor_longer),
                "stress", 8, "negative_control"))
    return outputs


def census(questions: dict[str, Question],
           retrieval: dict[str, list[Passage]],
           ids: list[str], depths: tuple[int, ...]) -> dict:
    """Source coverage and duplication stats. NOT evidence sufficiency labels."""
    rows = []
    for qid in ids:
        docs = retrieval.get(qid, [])
        for k in depths:
            if k > len(docs):
                continue
            prefix = docs[:k]
            rows.append({
                "id": qid, "depth": k,
                "documents": len(prefix),
                "retrieved_characters": sum(len(d.text) for d in prefix),
                "unique_sources": len({d.source for d in prefix}),
                "duplicate_fraction": (
                    1 - len({" ".join(d.text.split()).casefold() for d in prefix)
                    / len(prefix)) if prefix else 0.0,
            })
    summaries = {}
    for k in sorted(set(depths)):
        matches = [r for r in rows if r["depth"] == k]
        summaries[str(k)] = {
            "available_questions": len(matches),
            "mean_retrieved_characters": mean(r["retrieved_characters"] for r in matches)
            if matches else None,
            "mean_unique_sources": mean(r["unique_sources"] for r in matches)
            if matches else None,
            "mean_duplicate_fraction": mean(r["duplicate_fraction"] for r in matches)
            if matches else None,
        }
    return {
        "n_manifest": len(ids),
        "n_with_any_cache": sum(qid in retrieval and bool(retrieval[qid])
                                for qid in ids),
        "depth_census": summaries,
        "warning": "Statistics measure document availability, source coverage and "
                   "text size, not factual sufficiency or reader usefulness.",
    }
