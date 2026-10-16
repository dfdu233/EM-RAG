"""Evidence-Confronted MedRGAG (EC-MedRGAG): an EXPLICIT literature fusion.

Sources of ideas (NOT claims of reproducing each complete algorithm):
  - MedRGAG (WWW 2026): cached retrieval + KGCC + original KADS B=0 anchor
  - S2G-RAG (ACL 2026): query a concrete missing answer condition
  - GuarantRAG (ACL 2026): separate model prior from source-grounded answer
  - MoI (Findings NAACL 2025): optional evidence-order perturbation

No weights are trained, no DPO, no token-level joint decoding, no gold at
inference time, no claim that verbatim quotes entail the medical answer.

The B=0 path is the original MedRGAG Reader prompt, unchanged. At B>0
the local Reader sees a SOURCE-ONLY evidence view, with NO shared KGCC.
Only if the evidence view disagrees, quotes actually occur in DISPLAYED
source excerpts, optional reversed order is consistent, AND a final
source-aware comparison also selects the new answer with valid quotes,
is a switch attempted. This is a heuristic, NOT a correctness verifier.

All rounds and changed answers are logged for paired offline evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Protocol

from .data import Passage, Question, reader_prompt
from .models import ChatModel, parse_answer, parse_json
from .progressive import (
    content_fingerprint, normalize_query, source_key, source_records,
)


class Retriever(Protocol):
    calls: int
    def search(self, query: str, *, k: int) -> list[Passage]: ...


@dataclass(frozen=True)
class Config:
    max_rounds: int = 2
    retrieval_k: int = 8
    source_docs: int = 5
    source_context: str = "new_only"   # ablation: "external_anchor"
    planner_mode: str = "kgcc"         # ablation: "question"
    policy: str = "confront"           # ablation: "latest"
    order_check: bool = True          # ablation: False
    max_query_chars: int = 300
    planner_excerpt_chars: int = 500
    source_excerpt_chars: int = 1900
    context_tokens: int = 128000
    max_prompt_chars: int = 150000
    temperature: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.max_rounds <= 8:
            raise ValueError("max_rounds must be 0..8")
        if not 1 <= self.retrieval_k <= 128:
            raise ValueError("retrieval_k must be 1..128")
        if not 1 <= self.source_docs <= 5:
            raise ValueError("Reader source context supports 1..5 documents")
        if self.source_context not in {"new_only", "external_anchor"}:
            raise ValueError("Unknown source-context ablation")
        if self.planner_mode not in {"kgcc", "question"}:
            raise ValueError("Unknown planner ablation")
        if self.policy not in {"confront", "latest"}:
            raise ValueError("Unknown comparison policy")
        if not 40 <= self.max_query_chars <= 300:
            raise ValueError("MedRGAG live retrieval query max is 300")
        if min(self.planner_excerpt_chars, self.source_excerpt_chars,
               self.context_tokens, self.max_prompt_chars) < 1:
            raise ValueError("Text budget must be positive")


def get_choice(raw: str, labels: tuple[str, ...]) -> tuple[str | None, bool]:
    """Parse ONLY explicit answer-choice fields; never infer from reasoning.

    strict=False JSON handles literal newlines in model strings; fallback is
    limited to a malformed but explicitly keyed answer_choice or final label.
    """
    answer = parse_answer(raw, labels)
    if answer is not None:
        return answer, False
    matches = re.findall(
        r"""(?i)["']?answer_choice["']?\s*:\s*["']?\s*([A-Z])(?=["'\s,}\]]|$)""",
        raw,
    )
    finals = re.findall(
        r"(?im)^\s*(?:final\s+answer|answer\s+choice)\s*[:：]\s*\(?([A-Z])\)?\s*[.!]?\s*$",
        raw,
    )
    candidates = [c.upper() for c in matches + finals if c.upper() in labels]
    if len(set(candidates)) == 1:
        return candidates[0], True
    return None, False


def bounded_query(raw: str, max_chars: int = 300) -> tuple[str, bool]:
    """Deterministically enforce the *retriever* interface bound.

    Does not invent missing facts. Records every clip, which may remove a
    crucial qualifier; an abbreviated query is not automatically 'better'.
    """
    if not 40 <= max_chars <= 300:
        raise ValueError("Invalid query bound")
    cleaned = " ".join(raw.replace("\n", " ").split()).strip(" \t,;.")
    if len(cleaned) <= max_chars:
        return cleaned, False
    head = cleaned[:max_chars]
    # Avoid a partial final term; if last separator is very early, the first
    # max_chars are retained instead of silently dropping most of the query.
    last = head.rfind(" ")
    if last > max_chars // 2:
        head = head[:last]
    return head.strip(" \t,;."), True


def quote_is_visible(quote: str, body: str) -> bool:
    normalized = " ".join(quote.split()).casefold().strip(" '\"“”‘’")
    source = " ".join(body.split()).casefold()
    return len(normalized) >= 12 and normalized in source


def verified_quotes(raw: str, visible: dict[str, str]) -> list[dict]:
    """EXACT substring/provenance check, NOT semantic entailment check."""
    try:
        obj = parse_json(raw)
    except ValueError:
        return []
    provided = obj.get("evidence", obj.get("citations", []))
    if not isinstance(provided, list):
        return []
    seen: set[tuple[str, str]] = set()
    result = []
    for item in provided:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id", item.get("doc_id"))
        quote = item.get("quote")
        if (not isinstance(identifier, str)
                or not isinstance(quote, str)
                or identifier not in visible
                or not quote_is_visible(quote, visible[identifier])):
            continue
        key = (identifier, " ".join(quote.casefold().split()))
        if key in seen:
            continue
        seen.add(key)
        result.append({"id": identifier, "quote": quote.strip()})
    return result


GAP_SYSTEM = (
    "You write EXACTLY ONE short BM25 query to search an indexed medical "
    "corpus for facts that may distinguish the QUESTION's alternatives. "
    "Retrieved snippets and KGCC hypotheses may be wrong. "
    "Focus on a yet-UNRESOLVED conditional relationship, intervention, "
    "comparison, negation, timing or confounder. Do not presume the "
    "correct answer or trust KGCC as clinical evidence. "
    "The query must contain at most 20 search words and 180 characters; "
    "do not write any introduction or sentences outside JSON. "
    'Return JSON: {"query":"...","target_relation":"..."}.'
)
SOURCE_SYSTEM = (
    "You are a medical QA Reader inspecting ONLY the supplied EXTERNAL "
    "retrieved text. It may be incomplete, outdated or false. "
    "Pay attention to the exact population, negation, comparison, timing, "
    "adjustment variables and effect direction. Do not infer a conclusion "
    "from merely related facts. Answer using the A/B/C/... OPTION LABEL, "
    "and cite one or more VERBATIM spans from the displayed documents. "
    'Return JSON: {"answer_choice":"A",'
    '"evidence":[{"id":"doc-id","quote":"verbatim source text"}],'
    '"reasoning":"brief"}; if the source cannot establish an answer, '
    'use {"answer_choice":null,"evidence":[],"reasoning":"missing ..."} '
    "rather than inventing a citation."
)
CONFRONT_SYSTEM = (
    "Compare two competing medical answers without assuming either is "
    "correct. The historical MedRGAG answer is a model prediction and "
    "may be wrong. KGCC paragraphs are UNVERIFIED PARAMETRIC suggestions, "
    "never external corroboration. New retrieved sources can ALSO be "
    "wrong or only generally related. Follow exact conditional wording. "
    "If you CHANGE the previous answer, cite exact source excerpts that "
    "directly bear on the question's critical condition. The citation "
    "must be copied VERBATIM with the correct document ID. "
    'Return JSON only: {"answer_choice":"A",'
    '"evidence":[{"id":"doc-id","quote":"verbatim source text"}],'
    '"reasoning":"brief comparison of competing conditions"}.'
)


def source_prompt(q: Question, docs: list[Passage], max_excerpt: int) -> tuple[str, dict]:
    visible: dict[str, str] = {}
    entries = []
    for d in docs:
        if d.origin != "retrieved":
            raise ValueError("Source-only view must never contain generated KGCC")
        if d.id in visible:
            raise ValueError("Duplicate source identifier")
        excerpt = d.text[:max_excerpt].strip()
        visible[d.id] = excerpt
        entries.append(f"[{d.id}] (external corpus ID: {d.original_id})\n{excerpt}")
    prompt = (
        "ORIGINAL QUESTION AND ANSWER OPTIONS:\n" + q.formatted()
        + "\n\nEXTERNAL SOURCE EXCERPTS (not verified truth):\n"
        + "\n\n".join(entries)
        + "\n\nDo NOT quote unseen text or make up missing study details."
    )
    return prompt, visible


class EvidenceConfrontedMedRGAG:
    def __init__(self, *, reader: ChatModel, planner: ChatModel | None,
                 retriever: Retriever | None, ranker, rag_template: str,
                 config: Config | None = None, tokenizer=None):
        self.reader = reader
        self.planner = planner
        self.retriever = retriever
        self.ranker = ranker
        self.rag_template = rag_template
        self.cfg = config or Config()
        self.tokenizer = tokenizer
        if self.cfg.max_rounds and (planner is None or retriever is None):
            raise ValueError("New rounds require a live retriever and planner")

    def _original_medrgag(self, q: Question, docs: list[Passage]) -> dict:
        ordered = self.ranker.rank(q, list(docs))[:5]
        prompt = reader_prompt(
            q, ordered, self.rag_template, tokenizer=self.tokenizer,
            context_tokens=self.cfg.context_tokens,
            max_prompt_chars=self.cfg.max_prompt_chars)
        raw = self.reader.complete(
            "You are a helpful assistant.", prompt,
            temperature=self.cfg.temperature, seed=self.cfg.seed,
            max_tokens=512)
        choice, repaired = get_choice(raw, q.labels)
        return {
            "prediction": choice, "parse_recovered": repaired, "raw": raw,
            "prompt_sha256": sha256(prompt.encode("utf8")).hexdigest(),
            "prompt_chars": len(prompt),
            "ids": [d.id for d in ordered],
            "sources": [d.origin for d in ordered],
        }

    def _plan(self, q: Question, generated: list[Passage],
              previous: list[str], discovered: list[Passage],
              current: str | None, round_no: int) -> dict:
        if self.planner is None:
            raise AssertionError("Missing planner")
        kgcc_text = ""
        if self.cfg.planner_mode == "kgcc":
            kgcc_text = (
                "\nKGCC GENERATED IDEAS: THESE ARE NOT VERIFIED FACTS:\n"
                + "\n".join(g.text[:self.cfg.planner_excerpt_chars]
                            for g in generated[:5]))
        last = "\n".join(
            f"[{d.original_id}] {d.text[:self.cfg.planner_excerpt_chars]}"
            for d in discovered[-3:]) or "None"
        user = (
            "QUESTION:\n" + q.formatted()
            + "\nCURRENT READER ANSWER (may be wrong):\n" + str(current)
            + kgcc_text
            + "\nALREADY SEARCHED:\n" + "\n".join(previous)
            + "\nRECENT NEW EXTERNAL MATERIAL:\n" + last
            + f"\nNEW SEARCH ROUND: {round_no}"
        )
        raw = self.planner.complete(
            GAP_SYSTEM, user, temperature=0.0,
            seed=self.cfg.seed, max_tokens=320)
        outcome = {
            "query": None, "target_relation": None,
            "query_clipped": False, "query_rejected_reason": None,
            "raw": raw,
        }
        try:
            obj = parse_json(raw)
            query, target = obj.get("query"), obj.get("target_relation")
            if not isinstance(query, str) or not isinstance(target, str):
                raise ValueError("missing_query_or_relation")
            query, clipped = bounded_query(query, self.cfg.max_query_chars)
            if len(query) < 5:
                raise ValueError("query_too_short")
            canonical = normalize_query(query)
            if canonical == normalize_query(q.text):
                raise ValueError("duplicated_original_question")
            if canonical in {normalize_query(x) for x in previous}:
                raise ValueError("repeated_query")
            outcome.update({
                "query": query, "target_relation": target.strip()[:500],
                "query_clipped": clipped,
            })
        except ValueError as ex:
            outcome["query_rejected_reason"] = str(ex)
        return outcome

    def _source_answer(self, q: Question, docs: list[Passage]) -> dict:
        user, visible = source_prompt(q, docs, self.cfg.source_excerpt_chars)
        raw = self.reader.complete(
            SOURCE_SYSTEM, user, temperature=self.cfg.temperature,
            seed=self.cfg.seed, max_tokens=650)
        choice, recovered = get_choice(raw, q.labels)
        return {
            "prediction": choice, "parse_recovered": recovered,
            "evidence_quotes": verified_quotes(raw, visible),
            "raw": raw, "prompt_sha256": sha256(user.encode("utf8")).hexdigest(),
            "doc_ids": [x.id for x in docs], "prompt_chars": len(user),
            "visible": visible,
        }

    def _confront(self, q: Question, prior_label: str | None,
                  base: dict, evidence: dict) -> dict:
        # Only displayed source text is eligible for validating citations.
        cited = "\n\n".join(
            f"[{x['id']}]\n{x['quote']}"
            for x in evidence["evidence_quotes"])
        visible = {
            x["id"]: x["quote"] for x in evidence["evidence_quotes"]
        }
        prompt = (
            "ORIGINAL QUESTION:\n" + q.formatted()
            + "\n\nPREVIOUS ANSWER (not a verified fact):\n"
            + str(prior_label)
            + "\n\nORIGINAL MEDRGAG OUTPUT (unverified reasoning):\n"
            + base["raw"][:900]
            + "\n\nINDEPENDENT SOURCE-ONLY ANSWER (not verified):\n"
            + str(evidence["prediction"])
            + "\n\nEXACT CITED SOURCE EXCERPTS:\n" + cited
            + "\n\nOnly the cited passages above can serve as "
              "external evidence. Other facts are prior knowledge."
        )
        raw = self.reader.complete(
            CONFRONT_SYSTEM, prompt, temperature=self.cfg.temperature,
            seed=self.cfg.seed, max_tokens=650)
        label, recovered = get_choice(raw, q.labels)
        return {
            "prediction": label, "parse_recovered": recovered,
            "evidence_quotes": verified_quotes(raw, visible),
            "raw": raw, "prompt_sha256": sha256(prompt.encode("utf8")).hexdigest(),
            "prompt_chars": len(prompt),
        }

    def run(self, q: Question, retrieved: list[Passage],
            generated: list[Passage], *,
            original_selection: tuple[int, ...] | None) -> dict:
        if len(retrieved) < 5 or len(generated) < 5:
            raise ValueError("MedRGAG requires original five retrieved + five KGCC")
        if (original_selection is None
                or not 1 <= len(original_selection) <= 5
                or len(set(original_selection)) != len(original_selection)
                or any(type(x) is not int or x < 0 or x >= 10
                       for x in original_selection)):
            raise ValueError("Invalid original KADS selection")
        cfg = self.cfg
        choices = retrieved[:5] + generated[:5]
        anchor = [choices[i] for i in original_selection]
        rcalls = self.reader.calls
        pcalls = self.planner.calls if self.planner else 0
        searchcalls = self.retriever.calls if self.retriever else 0
        rankercalls = self.ranker.calls
        base = self._original_medrgag(q, anchor)
        current = base["prediction"]
        prefix = [{
            "round": 0, "prediction": current,
            "base_prediction": base["prediction"],
            "reason": "exact_MedRGAG_B0",
            "reader_calls_total": self.reader.calls - rcalls,
            "planner_calls_total": 0, "retrieval_calls_total": 0,
            "selected_doc_ids": base["ids"],
        }]
        initial_ids = {source_key(x) for x in retrieved}
        initial_hashes = {content_fingerprint(x.text) for x in retrieved}
        used_ids = {source_key(x) for x in anchor}
        used_hashes = {content_fingerprint(x.text) for x in anchor}
        old_retrieved = [d for d in retrieved[:5] if d.origin == "retrieved"]
        queries: list[str] = []
        new_views: list[dict] = []
        discovered: list[Passage] = []
        for turn in range(1, cfg.max_rounds + 1):
            plan = self._plan(q, generated, queries, discovered, current, turn)
            stage = {
                "round": turn, "query": plan["query"],
                "target_relation_unverified": plan["target_relation"],
                "raw_planner_output": plan["raw"],
                "query_clipped": plan["query_clipped"],
                "query_rejected_reason": plan["query_rejected_reason"],
                "fresh_source_count": 0,
                "fresh_outside_original_top32": 0,
                "fresh_corpus_documents": [],
                "source_prediction": None,
                "source_citations": [],
                "source_parse_recovered": False,
                "reversed_prediction": None,
                "confront_prediction": None,
                "confront_citations": [],
                "decision": "no_new_source",
                "prediction": None,
            }
            if plan["query"] is not None:
                queries.append(plan["query"])
                candidates = self.retriever.search(
                    plan["query"], k=cfg.retrieval_k)
                unique: list[Passage] = []
                ids_local, hashes_local = set(), set()
                for d in candidates:
                    if d.origin != "retrieved" or not d.text.strip():
                        continue
                    key, digest = source_key(d), content_fingerprint(d.text)
                    if (key in used_ids or digest in used_hashes
                            or key in ids_local or digest in hashes_local):
                        continue
                    unique.append(d)
                    ids_local.add(key)
                    hashes_local.add(digest)
                fresh = (self.ranker.rank(q, unique)[:cfg.source_docs]
                         if unique else [])
                stage["fresh_source_count"] = len(fresh)
                stage["fresh_corpus_documents"] = source_records(fresh)
                stage["fresh_outside_original_top32"] = sum(
                    source_key(d) not in initial_ids
                    and content_fingerprint(d.text) not in initial_hashes
                    for d in fresh)
                if fresh:
                    discovered.extend(fresh)
                    used_ids.update(source_key(d) for d in fresh)
                    used_hashes.update(content_fingerprint(d.text) for d in fresh)
                    docs = list(fresh)
                    if cfg.source_context == "external_anchor":
                        # Explicit ablation of anchor correlation. Still
                        # NEVER includes generated KGCC in source-only view.
                        ids = {x.id for x in docs}
                        docs.extend(x for x in old_retrieved if x.id not in ids)
                        docs = docs[:cfg.source_docs]
                    external = self._source_answer(q, docs)
                    stage.update({
                        "source_prediction": external["prediction"],
                        "source_citations": external["evidence_quotes"],
                        "source_parse_recovered": external["parse_recovered"],
                        "source_doc_ids": external["doc_ids"],
                        "source_prompt_sha256": external["prompt_sha256"],
                        "source_raw_reader_output": external["raw"],
                    })
                    if cfg.policy == "latest":
                        # Explicit optimistic ablation to isolate the cost
                        # of evidence checks and adjudication.
                        if external["prediction"] is not None:
                            current = external["prediction"]
                            stage["decision"] = "latest_unverified_ablation"
                    elif external["prediction"] == current:
                        stage["decision"] = "source_agrees_with_prior"
                    elif external["prediction"] is None:
                        stage["decision"] = "source_abstains_or_unparseable"
                    elif not external["evidence_quotes"]:
                        stage["decision"] = "no_valid_exact_source_quote"
                    else:
                        stable = True
                        if cfg.order_check and len(docs) > 1:
                            reverse = self._source_answer(q, list(reversed(docs)))
                            stage["reversed_prediction"] = reverse["prediction"]
                            stage["reversed_citations"] = reverse["evidence_quotes"]
                            stable = (reverse["prediction"] == external["prediction"]
                                      and bool(reverse["evidence_quotes"]))
                        if not stable:
                            stage["decision"] = "order_sensitive_or_uncited"
                        else:
                            checked = self._confront(q, current, base, external)
                            stage["confront_prediction"] = checked["prediction"]
                            stage["confront_citations"] = checked["evidence_quotes"]
                            stage["confront_raw_reader_output"] = checked["raw"]
                            if (checked["prediction"] == external["prediction"]
                                    and checked["evidence_quotes"]):
                                current = external["prediction"]
                                stage["decision"] = (
                                    "baseline_parse_recovered_via_cited_source"
                                    if base["prediction"] is None
                                    else "cited_confrontation_adopted")
                            else:
                                stage["decision"] = "confrontation_does_not_support_change"
            stage["prediction"] = current
            new_views.append(stage)
            prefix.append({
                "round": turn, "prediction": current,
                "base_prediction": base["prediction"],
                "reason": stage["decision"],
                "reader_calls_total": self.reader.calls - rcalls,
                "planner_calls_total": (self.planner.calls - pcalls
                                        if self.planner else 0),
                "retrieval_calls_total": (self.retriever.calls - searchcalls
                                          if self.retriever else 0),
                "queries_used_total": len(queries),
                "selected_doc_ids": (
                    stage.get("source_doc_ids") if
                    stage["decision"] in {
                        "cited_confrontation_adopted",
                        "baseline_parse_recovered_via_cited_source",
                        "latest_unverified_ablation"} else base["ids"]),
            })
        return {
            "id": q.id, "mode": "evidence_confronted_medrgag",
            "policy": cfg.policy, "planner_mode": cfg.planner_mode,
            "source_context": cfg.source_context,
            "order_check": cfg.order_check,
            "prediction": current, "base_prediction": base["prediction"],
            "baseline_parse_recovered": base["parse_recovered"],
            "baseline_prompt_sha256": base["prompt_sha256"],
            "baseline_prompt_chars": base["prompt_chars"],
            "base_raw_reader_output": base["raw"],
            "base_selected_doc_ids": base["ids"],
            "base_selected_sources": base["sources"],
            "prefixes": prefix, "novel_views": new_views,
            "depth_used": 5 + sum(x["fresh_source_count"] for x in new_views),
            "reader_calls": self.reader.calls - rcalls,
            "auxiliary_calls": (self.planner.calls - pcalls if self.planner else 0),
            "retrieval_calls": (self.retriever.calls - searchcalls
                                if self.retriever else 0),
            "ranker_batches": self.ranker.calls - rankercalls,
            "gold_used_at_inference": False,
            "quotes_are_entailment_proofs": False,
            "model_agreement_guarantees_correctness": False,
            "historical_KGCC_and_KADS_cost_excluded": True,
        }
