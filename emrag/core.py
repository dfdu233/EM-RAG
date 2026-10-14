"""Typed protocol and an append-only source-quotation ledger.

Exact quotation grounding does NOT establish that the source is scientifically
correct or that the abstracted claim is entailed by it.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import re
from typing import Any, Mapping


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    options: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question is empty")
        if not self.options or len(self.options) > 12:
            raise ValueError("1-12 explicit answer choices required")
        if any(not k or not str(v).strip() for k, v in self.options.items()):
            raise ValueError("empty answer choice")
        if len(set(k.upper() for k in self.options)) != len(self.options):
            raise ValueError("choice labels differ only in case")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(k.upper() for k in self.options)

    def formatted(self) -> str:
        return self.question + "\n" + "\n".join(
            f"{k.upper()}. {v}" for k, v in self.options.items()
        )


@dataclass(frozen=True)
class Document:
    id: str
    contents: str
    title: str = ""
    source: str = "local"
    score: float = 0.0
    kind: str = "retrieved"
    query: str = ""
    round: int = 0


@dataclass(frozen=True)
class Evidence:
    id: str
    doc_id: str
    statement: str
    quote: str
    source: str
    title: str
    round: int
    provenance: str = "source_quote_not_fact_checked"


@dataclass(frozen=True)
class ScoreCard:
    choice: str
    support_ids: tuple[str, ...] = ()
    refute_ids: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()
    sufficient: bool = False
    explanation: str = ""

    @property
    def score(self) -> int:
        return (2 * len(set(self.support_ids))
                - 4 * len(set(self.refute_ids))
                - len(set(self.missing_facts)))

    def eligible(self, min_support: int = 1) -> bool:
        return (self.sufficient and len(set(self.support_ids)) >= min_support
                and not self.refute_ids and not self.missing_facts)


@dataclass
class RoundTrace:
    round: int
    queries: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    accepted_evidence_ids: list[str] = field(default_factory=list)
    proposed_choice: str | None = None
    selected_choice: str | None = None
    status: str = "unverified"
    action: str = "initial"
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_facts: list[str] = field(default_factory=list)
    disputes: list[str] = field(default_factory=list)
    active_evidence_ids: list[str] = field(default_factory=list)
    model_calls: int = 0


@dataclass
class RunResult:
    id: str
    question: str
    options: dict[str, str]
    baseline: str | None
    selected: str | None
    status: str
    traces: list[RoundTrace]
    evidence: list[Evidence]
    model_calls: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceLedger:
    """Stores only external passages and exact excerpts, never generated KGCC text."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.evidence: dict[str, Evidence] = {}
        self._by_text: set[str] = set()

    def add_documents(self, docs: list[Document], round_id: int) -> list[Document]:
        fresh: list[Document] = []
        for doc in docs:
            if doc.kind != "retrieved":
                continue
            contents = doc.contents.strip()
            if not contents:
                continue
            text_key = digest(norm(contents))
            if text_key in self._by_text:
                continue
            stable_id = "d-" + digest(doc.source + "\0" + doc.id + "\0" + contents)
            saved = Document(id=stable_id, contents=contents, title=doc.title,
                             source=doc.source, score=doc.score, kind="retrieved",
                             query=doc.query, round=round_id)
            self._by_text.add(text_key)
            self.documents[stable_id] = saved
            fresh.append(saved)
        return fresh

    def add_claims(self, claims: list[dict[str, Any]]) -> list[Evidence]:
        fresh: list[Evidence] = []
        for item in claims:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id", ""))
            doc = self.documents.get(doc_id)
            if doc is None or doc.kind != "retrieved":
                continue
            quote = str(item.get("quote", "")).strip()
            statement = str(item.get("claim", "")).strip()
            if (not quote or len(quote) > 750 or len(statement) > 500
                    or len(statement) < 3 or quote not in doc.contents):
                continue
            evidence_id = "e-" + digest(doc_id + "\0" + quote + "\0" + norm(statement))
            if evidence_id in self.evidence:
                continue
            record = Evidence(id=evidence_id, doc_id=doc_id, statement=statement,
                              quote=quote, source=doc.source, title=doc.title,
                              round=doc.round)
            self.evidence[evidence_id] = record
            fresh.append(record)
        return fresh

    def select(self, max_items: int, max_chars: int,
               protected: set[str] | None = None) -> list[Evidence]:
        if max_items < 1 or max_chars < 1:
            return []
        protected = protected or set()
        items = list(self.evidence.values())
        items.sort(key=lambda e: (e.id not in protected, -e.round, e.id))
        pinned = [e for e in items if e.id in protected]
        remaining_items = [e for e in items if e.id not in protected]
        # Give each passage one chance before adding more quotations from it.
        # Pinned support stays ahead of redundant newly retrieved snippets.
        seen_docs = {e.doc_id for e in pinned}
        diverse: list[Evidence] = []
        overflow: list[Evidence] = []
        for e in remaining_items:
            if e.doc_id in seen_docs:
                overflow.append(e)
            else:
                seen_docs.add(e.doc_id)
                diverse.append(e)
        items = pinned + diverse + overflow
        result: list[Evidence] = []
        remaining = max_chars
        for item in items:
            cost = len(format_evidence(item))
            if cost <= remaining:
                result.append(item)
                remaining -= cost
            if len(result) >= max_items:
                break
        return result


def format_evidence(item: Evidence) -> str:
    return (f"[{item.id}] Source: {item.source} / {item.title} "
            f"(document {item.doc_id})\n"
            f"Extracted claim: {item.statement}\n"
            f"Verbatim source quote: {item.quote}\n")


def sanitize_card(choice: str, payload: Any,
                  allowed_ids: set[str]) -> ScoreCard:
    """Discard invented citation identifiers and overlapping support/refute."""
    d = payload if isinstance(payload, dict) else {}
    def ids(key: str) -> tuple[str, ...]:
        v = d.get(key, [])
        if not isinstance(v, list):
            return ()
        return tuple(dict.fromkeys(x for x in v if isinstance(x, str)
                                   and x in allowed_ids))
    refs = ids("refute_ids")
    support = tuple(x for x in ids("support_ids") if x not in refs)
    missing = d.get("missing_facts", [])
    if not isinstance(missing, list):
        missing = []
    missing = tuple(str(x)[:160] for x in missing[:8] if isinstance(x, str) and x.strip())
    return ScoreCard(choice=choice, support_ids=support, refute_ids=refs,
                     missing_facts=missing, sufficient=d.get("sufficient") is True,
                     explanation=str(d.get("explanation", ""))[:400])
