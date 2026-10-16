"""Training-free, question-conditioned evidence utility with a fixed objective.

U_q(C) = mean over question facets f of max_{d in C} sim(f, d).

The facets depend ONLY on the question and answer options. The score for a
given (facet, document) pair is frozen; it is NEVER re-normalized using the
growing retrieval pool. Therefore the SAME context has the SAME utility at
every evidence budget. The objective is normalized, monotone and submodular
for nonnegative facet-document scores.

IMPORTANT: This is evidence coverage, not a calibrated P(answer correct).
Training-free does not imply factual entailment. Accurate medical conditional
support is NOT guaranteed by lexical overlap or a relevance cross-encoder.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Protocol, Sequence

from .data import Passage, Question


# Fixed generic splits only: no biomedical or dataset-specific patterns.
_SPLITS = re.compile(
    r"[,;:]|\b(?:while|whereas|after|before|if|when|without|"
    r"compared\s+(?:to|with)|controlling\s+for|adjusting\s+for)\b",
    flags=re.IGNORECASE,
)
_TOKENS = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_STOP = frozenset({
    "a", "an", "the", "of", "to", "for", "by", "in", "on", "at", "from",
    "is", "are", "was", "were", "be", "been", "can", "could", "would",
    "should", "do", "does", "did", "have", "has", "had", "this", "that",
    "these", "those", "it", "its", "which", "what", "who", "whom",
    "and", "or", "with", "as", "than", "not", "any", "there", "into",
    "may", "might", "between", "among", "using", "used", "use", "most",
})


def tokens(text: str) -> set[str]:
    """Set-valued lexical evidence features (NOT an entailment detector)."""
    return {term for term in _TOKENS.findall(text.lower())
            if term not in _STOP and len(term) > 1 or
            (len(term) == 1 and '\u4e00' <= term <= '\u9fff')}


def fixed_facets(q: Question, *, max_facets: int = 8) -> tuple[str, ...]:
    """Deterministic generic question facets, fixed BEFORE any retrieval.

    Includes entire question, substantive clauses, and option-anchored
    questions when answer options contain substantive concepts. Generic
    yes/no/maybe options are NOT treated as distinct medical facts.
    """
    if max_facets < 1:
        raise ValueError("max_facets must be positive")
    result = [q.text.strip()]
    seen = {q.text.strip().casefold()}
    for fragment in _SPLITS.split(q.text):
        piece = fragment.strip(" ?.\t\r\n")
        if len(tokens(piece)) < 3 or piece.casefold() in seen:
            continue
        result.append(piece)
        seen.add(piece.casefold())
        if len(result) >= max_facets:
            return tuple(result)
    generic = {"yes", "no", "maybe", "true", "false", "not sure",
               "none of the above", "all of the above"}
    for label, answer in q.options.items():
        if answer.strip().casefold() in generic or len(tokens(answer)) < 2:
            continue
        facet = q.text.strip() + " " + answer.strip()
        if facet.casefold() not in seen:
            result.append(facet)
            seen.add(facet.casefold())
        if len(result) >= max_facets:
            break
    return tuple(result)


class FacetScorer(Protocol):
    facets: tuple[str, ...]
    calls: int
    def score(self, documents: Sequence[Passage]) -> dict[str, tuple[float, ...]]: ...


class LexicalScorer:
    """No weights, GPU, fitting, online normalization or auxiliary LLM."""

    name = "lexical"

    def __init__(self, question: Question, *, max_facets: int = 8):
        self.facets = fixed_facets(question, max_facets=max_facets)
        self._features = [tokens(t) for t in self.facets]
        self.calls = 0

    def score(self, documents: Sequence[Passage]) -> dict[str, tuple[float, ...]]:
        values = {}
        for doc in documents:
            features = tokens(doc.text)
            values[doc.id] = tuple(
                len(query & features) / len(query) if query else 0.0
                for query in self._features)
        self.calls += len(documents)
        return values


class MedCPTScorer:
    """Use the ALREADY loaded frozen MedCPT Cross-Encoder for facet matching.

    Logit sigmoid is a bounded uncalibrated similarity surrogate, not an
    answer probability. Each (facet,document) is scored once and cached.
    """

    name = "medcpt"

    def __init__(self, question: Question, ranker, *,
                 max_facets: int = 8, batch_size: int = 8):
        if not all(hasattr(ranker, x) for x in
                   ("torch", "model", "tokenizer", "device", "max_length")):
            raise ValueError("MedCPT facet utility requires MedCPTReranker")
        self.facets = fixed_facets(question, max_facets=max_facets)
        self.ranker = ranker
        self.batch_size = batch_size
        self.calls = 0

    def score(self, documents: Sequence[Passage]) -> dict[str, tuple[float, ...]]:
        docs = list(documents)
        if not docs:
            return {}
        scores: dict[str, list[float]] = {d.id: [] for d in docs}
        for facet in self.facets:
            for start in range(0, len(docs), self.batch_size):
                batch = docs[start:start + self.batch_size]
                pairs = [[facet, d.text] for d in batch]
                encoded = self.ranker.tokenizer(
                    pairs, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.ranker.max_length)
                encoded = {k: v.to(self.ranker.device) for k, v in encoded.items()}
                with self.ranker.torch.no_grad():
                    logits = self.ranker.model(**encoded).logits.reshape(-1)
                    raw = logits.detach().cpu().tolist()
                if len(raw) != len(batch):
                    raise ValueError("Unexpected MedCPT cross-encoder score shape")
                for doc, logit in zip(batch, raw):
                    z = min(30.0, max(-30.0, float(logit)))
                    scores[doc.id].append(1 / (1 + math.exp(-z)))
                self.calls += 1
        return {k: tuple(v) for k, v in scores.items()}


def utility(selection: Sequence[Passage],
            matrix: dict[str, tuple[float, ...]]) -> float:
    """Coverage over FIXED question facets, invariant to future retrieval."""
    if not matrix:
        return 0.0
    n_facets = len(next(iter(matrix.values())))
    if n_facets == 0:
        return 0.0
    if not selection:
        return 0.0
    if any(d.id not in matrix or len(matrix[d.id]) != n_facets for d in selection):
        raise ValueError("Missing/mismatched fixed facet scores")
    value = 0.0
    for index in range(n_facets):
        value += max(matrix[doc.id][index] for doc in selection)
    return value / n_facets


def greedy_coverage(available: Sequence[Passage],
                    matrix: dict[str, tuple[float, ...]],
                    *, capacity: int = 5) -> list[Passage]:
    """Greedy monotone submodular maximization with deterministic tie-breaks.

    A separate incumbent must be retained because greedy solutions across
    larger candidate pools are not necessarily score-monotone.
    """
    if capacity < 1:
        raise ValueError("capacity must be positive")
    pool = list(available)
    if len({d.id for d in pool}) != len(pool):
        raise ValueError("Document IDs must be unique")
    selected: list[Passage] = []
    chosen = set()
    for _ in range(min(capacity, len(pool))):
        best = None
        best_value = -1.0
        for doc in pool:
            if doc.id in chosen:
                continue
            value = utility(selected + [doc], matrix)
            # Exact tie: preserve original retrieval/generated order.
            if value > best_value + 1e-12:
                best, best_value = doc, value
        if best is None:
            break
        selected.append(best)
        chosen.add(best.id)
    return selected
