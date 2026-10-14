"""EM-RAG: multi-round retrieval and evidence-conditioned answer selection.

The gate is a transparent proxy policy, not an accuracy guarantee. All
components use frozen inference models. Gold labels never enter the engine.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
from .core import Question, Document, Evidence, EvidenceLedger, RunResult, RoundTrace, ScoreCard, norm, sanitize_card
from .model import Model, request_json
from .prompts import (SYSTEM_PLAN, SYSTEM_EXTRACT, SYSTEM_ANSWER, SYSTEM_JUDGE,
                      plan_prompt, extraction_prompt, answer_prompt, judge_prompt, answer_choice)
from .retrieval import Retriever


@dataclass(frozen=True)
class EMRAGConfig:
    max_rounds: int = 4
    queries_per_round: int = 2
    docs_per_query: int = 5
    max_new_docs: int = 10
    extraction_batch_size: int = 3
    claims_per_doc: int = 3
    max_active_claims: int = 14
    max_context_chars: int = 9000
    min_support: int = 1
    min_score_margin: int = 2
    max_empty_rounds: int = 2
    stop_on_verified: bool = False
    stable_rounds: int = 2
    unverified_policy: str = "baseline"  # baseline or abstain

    def __post_init__(self) -> None:
        for key in ("max_rounds", "queries_per_round", "docs_per_query",
                    "max_new_docs", "extraction_batch_size", "claims_per_doc",
                    "max_active_claims", "max_context_chars", "min_support",
                    "stable_rounds", "max_empty_rounds"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.min_score_margin < 0:
            raise ValueError("min_score_margin must be nonnegative")
        if self.unverified_policy not in {"baseline", "abstain"}:
            raise ValueError("unverified_policy must be baseline or abstain")


class EMRAG:
    """Keeps separate evidence and candidate pools for each question."""

    def __init__(self, model: Model, retriever: Retriever,
                 config: EMRAGConfig | None = None):
        self.model = model
        self.retriever = retriever
        self.config = config or EMRAGConfig()
        self._calls = 0

    def _json(self, system: str, user: str) -> dict:
        self._calls += 1
        return request_json(self.model, system, user)

    def _propose(self, question: Question, evidence: list[Evidence]) -> str | None:
        try:
            payload = self._json(SYSTEM_ANSWER, answer_prompt(question, evidence))
        except ValueError:
            return None
        return answer_choice(payload, question.labels)

    def _plan(self, question: Question, active: list[Evidence],
              history: list[str], proposals: set[str], hints: list[str],
              first_round: bool) -> tuple[list[str], list[str], list[str]]:
        cfg = self.config
        # Anchor the initial search to the unchanged original question.
        original = question.formatted()[:1000]
        slots = cfg.queries_per_round - 1 if first_round else cfg.queries_per_round
        data: dict[str, Any] = {}
        if slots:
            try:
                data = self._json(
                    SYSTEM_PLAN, plan_prompt(question, active, sorted(proposals),
                                            history, hints, slots))
            except ValueError:
                pass
        def string_list(key: str) -> list[str]:
            raw = data.get(key, [])
            if not isinstance(raw, list):
                return []
            return [x.strip()[:160] for x in raw[:10] if isinstance(x, str) and x.strip()]
        missing, disputes = string_list("missing_facts"), string_list("disputes")
        raw_queries = data.get("queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = []
        requested = ([original] if first_round else []) + [
            x.strip()[:240] for x in raw_queries if isinstance(x, str) and x.strip()
        ]
        seen = {norm(x) for x in history}
        queries: list[str] = []
        for query in requested:
            if norm(query) not in seen and query.strip():
                queries.append(query)
                seen.add(norm(query))
            if len(queries) >= cfg.queries_per_round:
                break
        return queries, missing, disputes

    def _search(self, question: Question, queries: list[str]) -> list[Document]:
        docs: list[Document] = []
        for query in queries:
            docs.extend(self.retriever.search(
                query, self.config.docs_per_query, question_id=question.id))
        return docs

    def _extract(self, question: Question, docs: list[Document]) -> list[dict[str, Any]]:
        cfg = self.config
        claims: list[dict[str, Any]] = []
        per_doc: dict[str, int] = {}
        for start in range(0, len(docs), cfg.extraction_batch_size):
            batch = docs[start:start + cfg.extraction_batch_size]
            try:
                payload = self._json(
                    SYSTEM_EXTRACT, extraction_prompt(question, batch, cfg.claims_per_doc))
            except ValueError:
                continue
            items = payload.get("claims", [])
            if not isinstance(items, list):
                continue
            allowed = {d.id for d in batch}
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("doc_id", ""))
                if key not in allowed or per_doc.get(key, 0) >= cfg.claims_per_doc:
                    continue
                claims.append(item)
                per_doc[key] = per_doc.get(key, 0) + 1
        return claims

    def _judge(self, question: Question, active: list[Evidence]) -> dict[str, ScoreCard]:
        allowed_ids = {e.id for e in active}
        try:
            payload = self._json(SYSTEM_JUDGE, judge_prompt(question, active))
        except ValueError:
            payload = {}
        raw = payload.get("cards", {})
        if not isinstance(raw, dict):
            raw = {}
        return {label: sanitize_card(label, raw.get(label), allowed_ids)
                for label in question.labels}

    def _decide(self, current: str | None, status: str,
                cards: dict[str, ScoreCard], proposals: set[str]
                ) -> tuple[str | None, str, str, set[str]]:
        cfg = self.config
        eligible = [card for card in cards.values()
                    if card.choice in proposals and card.eligible(cfg.min_support)]
        eligible.sort(key=lambda card: (-card.score, card.choice))
        best = eligible[0] if eligible else None
        # Compare every option on identical evidence, even unproposed options.
        runner = max((card.score for card in cards.values()
                      if best is None or card.choice != best.choice), default=-1000)
        if best is not None and best.score - runner >= cfg.min_score_margin:
            return (best.choice, "verified",
                    "accepted" if current != best.choice else "reverified",
                    set(best.support_ids))
        old = cards.get(current) if current else None
        if status == "verified" and old is not None and old.refute_ids:
            return None, "abstained", "invalidated_by_contradiction", set()
        if status == "verified" and old is not None and not old.eligible(cfg.min_support):
            return current, "unverified", "support_lost", set()
        return current, status, "preserved", set(old.support_ids) if old else set()

    def run(self, question: Question, *, kgcc_hints: list[str] | None = None) -> RunResult:
        cfg = self.config
        self._calls = 0
        ledger = EvidenceLedger()
        history: list[str] = []
        proposals: set[str] = set()
        traces: list[RoundTrace] = []
        baseline = self._propose(question, [])
        if baseline:
            proposals.add(baseline)
        selected = baseline
        status = "unverified" if baseline else "abstained"
        traces.append(RoundTrace(round=0, proposed_choice=baseline,
                                 selected_choice=selected, status=status,
                                 action="parametric_baseline", model_calls=self._calls))
        protected: set[str] = set()
        empty_rounds = 0
        stable_count = 0

        for round_id in range(1, cfg.max_rounds + 1):
            old_selected = selected
            active = ledger.select(cfg.max_active_claims, cfg.max_context_chars, protected)
            queries, missing, disputes = self._plan(
                question, active, history, proposals, kgcc_hints or [], round_id == 1)
            history.extend(queries)
            fetched = self._search(question, queries) if queries else []
            new_docs = ledger.add_documents(fetched[:cfg.max_new_docs], round_id)
            new_claims = ledger.add_claims(
                self._extract(question, new_docs)) if new_docs else []
            empty_rounds = 0 if new_claims else empty_rounds + 1
            active = ledger.select(cfg.max_active_claims, cfg.max_context_chars, protected)
            proposal = self._propose(question, active) if active else None
            if proposal:
                proposals.add(proposal)
            cards = self._judge(question, active) if active else {}
            action = "no_grounded_evidence"
            if cards:
                selected, status, action, now_protected = self._decide(
                    selected, status, cards, proposals)
                if status == "verified":
                    protected = now_protected
                elif action in {"invalidated_by_contradiction", "support_lost"}:
                    protected = set()
            scores = {choice: {**asdict(card), "score": card.score,
                               "eligible": card.eligible(cfg.min_support)}
                      for choice, card in cards.items()}
            traces.append(RoundTrace(
                round=round_id, queries=queries,
                retrieved_doc_ids=[d.id for d in new_docs],
                accepted_evidence_ids=[e.id for e in new_claims],
                proposed_choice=proposal, selected_choice=selected, status=status,
                action=action, scores=scores, missing_facts=missing,
                disputes=disputes, active_evidence_ids=[e.id for e in active],
                model_calls=self._calls))
            stable_count = stable_count + 1 if selected == old_selected and status == "verified" else 0
            if cfg.stop_on_verified and stable_count >= cfg.stable_rounds:
                break
            if empty_rounds >= cfg.max_empty_rounds:
                break

        if status != "verified" and cfg.unverified_policy == "abstain":
            selected = None
            status = "abstained"
            traces[-1].selected_choice = None
            traces[-1].status = status
            traces[-1].action += "_final_abstain"

        return RunResult(id=question.id, question=question.question,
                         options=dict(question.options), baseline=baseline,
                         selected=selected, status=status, traces=traces,
                         evidence=list(ledger.evidence.values()), model_calls=self._calls)
