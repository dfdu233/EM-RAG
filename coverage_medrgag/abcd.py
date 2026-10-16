"""Four-paper, training-free MedRGAG composition (not an original SOTA method).

A Query2doc (EMNLP'23): original KGCC generation becomes hypothetical BM25
  query expansion; generated text NEVER becomes verified external evidence.
B IRCoT (ACL'23): next source query depends on previous Reader reasoning.
C RECOMP (ICLR'24): deterministic extractive sentence compression, rather
  than the paper's *trained* compressor; original source retained in trace.
D SuRe (ICLR'24): source-only, answer-conditioned evidence notes. Only
  verbatim quotes matching displayed external documents are accepted. This
  is a SuRe-inspired adaptation, NOT a faithful reimplementation.

MedRGAG's KADS original five passages, KGCC, final MedCPT ranking, native
BM25 corpus and frozen Reader remain in place. Budget B=0 has exactly the
original MedRGAG Reader prompt. Every later B adds one actual source query;
there is NO answer vote and NO hidden gold at inference.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    fresh_per_round: int = 3
    max_reader_docs: int = 5
    max_query_chars: int = 300
    pseudo_doc_chars: int = 100
    excerpt_chars: int = 750
    extract_sentences: int = 2
    extract_chars: int = 550
    note_chars: int = 1600
    use_query2doc: bool = True
    use_ircot: bool = True
    use_recomp: bool = True
    use_sure: bool = True
    temperature: float = 0.1
    seed: int = 42
    context_tokens: int = 128000
    max_prompt_chars: int = 150000

    def __post_init__(self):
        if not 0 <= self.max_rounds <= 8:
            raise ValueError("Rounds must be between 0 and 8")
        if not 1 <= self.retrieval_k <= 128:
            raise ValueError("Bad retrieval_k")
        if not 1 <= self.fresh_per_round <= 4:
            raise ValueError("fresh_per_round must be 1..4")
        if self.max_reader_docs != 5:
            raise ValueError("Preserve MedRGAG Reader input budget of five passages")
        if not 64 <= self.max_query_chars <= 300:
            raise ValueError("BM25 query hard cap is 300 characters")
        if not 20 <= self.pseudo_doc_chars <= 200:
            raise ValueError("Invalid pseudo-document excerpt size")
        if not 80 <= self.excerpt_chars <= 4000:
            raise ValueError("Invalid excerpt size")
        if not 1 <= self.extract_sentences <= 8 or self.extract_chars < 80:
            raise ValueError("Invalid extractive compressor budget")
        if self.note_chars < 80 or self.max_prompt_chars < 100:
            raise ValueError("Invalid prompt budgets")


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=;)\s+|\n+")
_WORDS = re.compile(r"[a-z0-9]+", re.I)
_DEFAULT_STOP = {
    "with", "that", "which", "this", "from", "there", "were",
    "been", "into", "also", "have", "does", "could", "would",
    "should", "between", "more", "than", "about", "other",
}


def keywords(text: str) -> set[str]:
    # Preserve negation and conditions such as "not", "without" and "after".
    return {w for w in _WORDS.findall(text.casefold())
            if (len(w) >= 3 and w not in _DEFAULT_STOP)}


def fit_query(text: str, limit: int) -> tuple[str, bool]:
    """A bounded BM25 query with deterministic graceful handling of LLM overflow.

    Source search query may be truncated; SOURCE FACTS are never truncated
    or converted to trusted facts by this function.
    """
    raw = " ".join(str(text).split())
    if len(raw) <= limit:
        return raw, False
    fragment = raw[:limit]
    if " " in fragment:
        fragment = fragment.rsplit(" ", 1)[0]
    return fragment.strip() or raw[:limit], True


def combine_query(query: str, pseudo: str, limit: int) -> tuple[str, bool]:
    """Original question/query gets priority; KGCC pseudo-doc is optional."""
    base, was_clipped = fit_query(query, limit)
    room = max(0, limit - len(base) - 1)
    if not pseudo.strip() or room < 12:
        return base, was_clipped or bool(pseudo.strip())
    suffix, suffix_cut = fit_query(pseudo, room)
    return (base + " " + suffix).strip(), was_clipped or suffix_cut


def compress_source(q: Question, doc: Passage, *, max_sentences: int,
                    max_chars: int) -> Passage:
    """Training-free RECOMP-inspired extractive sentence selection.

    This is NOT the paper's trained summarizer. Every returned character
    (other than joining whitespace) originates from a selected source
    sentence. Compression does not claim to establish entailment or truth.
    """
    if doc.origin != "retrieved":
        raise ValueError("Do not compress generated KGCC as source evidence")
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(doc.text) if s.strip()]
    if not sentences:
        return doc
    question_terms = keywords(q.formatted())
    ranked = sorted(
        enumerate(sentences),
        key=lambda x: (
            -len(keywords(x[1]) & question_terms),
            x[0],
        ))
    chosen_indices = sorted(i for i, _ in ranked[:max_sentences])
    selected = " ".join(sentences[i] for i in chosen_indices)
    if len(selected) > max_chars:
        # Truncate the first sentence if the source has no short segments.
        selected = selected[:max_chars].rstrip()
    return replace(doc, text=selected)


def validate_option_notes(raw: str, q: Question, sources: list[Passage],
                          *, max_quotes_per_choice: int = 2) -> dict:
    """Accept only literal substring quotes from actual retrieved evidence.

    Valid quotation != clinical truth. The LLM's proposed support/refutation
    direction is untrusted and does NOT select the final answer.
    """
    accepted: list[dict] = []
    rejected = 0
    try:
        obj = parse_json(raw)
    except ValueError:
        return {"quotes": [], "rejected": 1, "parse_ok": False}
    by_id = {d.id: d for d in sources if d.origin == "retrieved"}
    entries = obj.get("choices", [])
    if not isinstance(entries, list):
        return {"quotes": [], "rejected": 1, "parse_ok": False}
    for entry in entries:
        if not isinstance(entry, dict):
            rejected += 1
            continue
        choice = str(entry.get("choice", "")).upper().strip()
        if choice not in q.labels:
            rejected += 1
            continue
        evidence = entry.get("quotes", [])
        if not isinstance(evidence, list):
            rejected += 1
            continue
        for record in evidence[:max_quotes_per_choice]:
            if not isinstance(record, dict):
                rejected += 1
                continue
            doc_id = str(record.get("id", ""))
            quote = record.get("quote")
            relation = record.get("relation")
            if (doc_id not in by_id or not isinstance(quote, str)
                    or len(quote.strip()) < 8
                    or not isinstance(relation, str)
                    or relation not in ("supports", "refutes")
                    or quote.strip() not in by_id[doc_id].text):
                rejected += 1
                continue
            accepted.append({
                "choice": choice, "source_id": doc_id,
                "source_original_id": by_id[doc_id].original_id,
                "quote": quote.strip(), "claimed_relation": relation,
                "is_valid_source_substring": True,
                "relation_is_not_verified": True,
            })
    return {"quotes": accepted, "rejected": rejected, "parse_ok": True}


def render_notes(result: dict, *, max_chars: int) -> str:
    """Only literal cited snippets enter Reader, not invented explanations."""
    quotes = result["quotes"]
    if not quotes:
        return ""
    lines = [
        "# OPTION-CONDITIONED EXTERNAL SOURCE EXCERPTS",
        "These are direct excerpts from retrieved source text, not "
        "independently verified medical truths.",
        "Support/refutation labels are hypotheses. Check the original "
        "question's population, condition, negation and comparison.",
        "Do not treat KGCC generated passages as source citations.",
    ]
    for record in quotes:
        line = (
            f"Choice {record['choice']} ({record['claimed_relation']}, "
            f"unverified), source [{record['source_id']}]: "
            + record["quote"].replace("\n", " "))
        if len("\n".join(lines + [line])) > max_chars:
            break
        lines.append(line)
    return "\n".join(lines) if len(lines) > 4 else ""


_PLANNER_SYSTEM = (
    "Produce one NEW English biomedical BM25 search query grounded in "
    "the original question, its qualifiers and currently unresolved facts. "
    "The previous Reader reasoning and KGCC background are fallible hints, "
    "NOT evidence. Seek direct source facts that could distinguish choices, "
    "including contradicting findings and adjusted conditions. "
    "The query must be concise (under 170 characters), without selecting "
    "an answer or referring to gold labels. "
    'Return JSON {"query":"...","target_fact":"..."} only.'
)

_SURE_SYSTEM = (
    "Make separate source-grounded evidence notes for each answer choice. "
    "USE ONLY the supplied external retrieved SOURCE documents. "
    "For every claimed support or contradiction, COPY a contiguous exact "
    "substring of the source (including negation/conditions), at least 8 "
    "characters. No paraphrase inside quotes. If there is no direct support "
    "or contradiction, return an empty list for the choice. "
    "Never cite generated KGCC. Do not assume any answer choice is correct. "
    "Return JSON only: "
    '{"choices":[{"choice":"A","quotes":[{"id":"n1_0",'
    '"quote":"exact source substring","relation":"supports"}]}]}.'
)


class MedRGAGABCD:
    def __init__(self, *, reader: ChatModel, auxiliary: ChatModel | None,
                 retriever: Retriever | None, ranker, rag_template: str,
                 config: Config | None = None, tokenizer=None):
        self.reader = reader
        self.auxiliary = auxiliary
        self.retriever = retriever
        self.ranker = ranker
        self.template = rag_template
        self.cfg = config or Config()
        self.tokenizer = tokenizer
        if self.cfg.max_rounds and retriever is None:
            raise ValueError("Expansion requires live MedRGAG retrieval")
        if (self.cfg.max_rounds and
                (self.cfg.use_ircot or self.cfg.use_sure) and auxiliary is None):
            raise ValueError("IRCoT/SuRe ablations require frozen auxiliary LLM")

    def _answer(self, q: Question, docs: list[Passage],
                *, option_notes: str = "") -> dict:
        chosen = self.ranker.rank(q, list(docs))[:self.cfg.max_reader_docs]
        prompt = reader_prompt(
            q, chosen, self.template, tokenizer=self.tokenizer,
            context_tokens=self.cfg.context_tokens,
            max_prompt_chars=self.cfg.max_prompt_chars)
        if option_notes:
            appendix = ("\n\n" + option_notes + "\n"
                        "Final answer must follow the original JSON answer_choice format.")
            if len(prompt) + len(appendix) > self.cfg.max_prompt_chars:
                # Never remove original question or source to fit notes.
                appendix = ""
            prompt += appendix
        raw = self.reader.complete(
            "You are a helpful assistant.", prompt,
            temperature=self.cfg.temperature, seed=self.cfg.seed,
            max_tokens=512)
        try:
            explanation = parse_json(raw).get("reasoning", "")
        except ValueError:
            explanation = ""
        return {
            "prediction": parse_answer(raw, q.labels),
            "reasoning": explanation[:700] if isinstance(explanation, str) else "",
            "raw_reader_output": raw,
            "selected_doc_ids": [d.id for d in chosen],
            "selected_sources": [d.origin for d in chosen],
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "option_note_chars": len(option_notes),
        }

    def _query(self, q: Question, generated: list[Passage], *,
               turn: int, previous: list[str], previous_reasoning: str,
               recent_sources: list[Passage]) -> dict:
        cfg = self.cfg
        raw_response = ""
        planned = q.text
        target = ""
        invalid = False
        if cfg.use_ircot:
            source_view = "\n".join(
                f"[{d.id}] {d.text[:220]}" for d in recent_sources[-3:])
            hint = " | ".join(g.text[:180] for g in generated[:2])
            ask = (
                "Question and options:\n" + q.formatted()
                + "\nPrevious Reader reasoning (NOT verified):\n"
                + previous_reasoning[:500]
                + "\nOriginal MedRGAG KGCC hypotheses (NOT sources):\n"
                + hint[:400]
                + "\nRecently retrieved external sources:\n"
                + source_view
                + "\nPrevious queries:\n" + "\n".join(previous)
                + f"\nRound: {turn}\nReturn one NEW query and target_fact.")
            raw_response = self.auxiliary.complete(
                _PLANNER_SYSTEM, ask, temperature=0.0,
                seed=cfg.seed, max_tokens=256)
            try:
                obj = parse_json(raw_response)
                proposed = obj.get("query")
                if not isinstance(proposed, str) or not proposed.strip():
                    raise ValueError("Empty planned query")
                planned = proposed.strip()
                target = str(obj.get("target_fact", ""))[:260]
            except ValueError:
                invalid = True
                planned = q.text
        # Query2doc adaptation: reuse *cached* MedRGAG KGCC as the pseudo-doc,
        # not an additional generated answer and not independent verification.
        pseudo = (generated[(turn - 1) % min(5, len(generated))].text[
            :cfg.pseudo_doc_chars] if cfg.use_query2doc else "")
        query, clipped = combine_query(
            planned, pseudo, cfg.max_query_chars)
        previous_set = {normalize_query(x) for x in previous}
        if normalize_query(query) in previous_set:
            # Allow a unique question-only backoff; never repeat a BM25 call.
            backoff, _ = fit_query(q.text, cfg.max_query_chars)
            if normalize_query(backoff) in previous_set:
                query = None
            else:
                query = backoff
        return {
            "query": query, "planned_query": planned,
            "pseudo_doc_expansion": bool(pseudo),
            "query_was_truncated": clipped,
            "planning_fallback": invalid,
            "target_fact_unverified": target,
            "raw_plan": raw_response,
        }

    def _option_notes(self, q: Question, docs: list[Passage]) -> dict:
        sources = [d for d in docs if d.origin == "retrieved"]
        if not sources:
            return {"quotes": [], "rejected": 0, "parse_ok": True, "text": ""}
        view = "\n".join(
            f"[{d.id}] {d.text[:self.cfg.excerpt_chars]}" for d in sources)
        raw = self.auxiliary.complete(
            _SURE_SYSTEM,
            "Question and alternatives:\n" + q.formatted()
            + "\nEXTERNAL SOURCE TEXT ONLY:\n" + view,
            temperature=0.0, seed=self.cfg.seed, max_tokens=950)
        # Validate the exact displayed excerpt, not text beyond that excerpt.
        excerpts = [replace(d, text=d.text[:self.cfg.excerpt_chars])
                    for d in sources]
        result = validate_option_notes(raw, q, excerpts)
        result["text"] = render_notes(result, max_chars=self.cfg.note_chars)
        return result

    def run(self, q: Question, retrieved: list[Passage],
            generated: list[Passage], *,
            original_selection: tuple[int, ...] | None) -> dict:
        cfg = self.cfg
        if len(retrieved) < 5 or len(generated) < 5:
            raise ValueError("Original MedRGAG needs first 5 retrieved and KGCC")
        if (original_selection is None
                or not 1 <= len(original_selection) <= 5
                or len(set(original_selection)) != len(original_selection)
                or any(type(x) is not int or not 0 <= x < 10
                       for x in original_selection)):
            raise ValueError("Original KADS final_list must be valid")
        base_candidates = retrieved[:5] + generated[:5]
        anchor = [base_candidates[x] for x in original_selection]
        reader_before = self.reader.calls
        aux_before = self.auxiliary.calls if self.auxiliary else 0
        retrieval_before = self.retriever.calls if self.retriever else 0
        rank_before = self.ranker.calls
        # Exactly ONE original MedRGAG Reader call at B0 with NO new prompts.
        original = self._answer(q, anchor)
        anchor_ids = {x.id: x for x in anchor}
        stable_anchor = [anchor_ids[x] for x in original["selected_doc_ids"]]
        prefixes = [{
            "round": 0, "prediction": original["prediction"],
            "base_prediction": original["prediction"],
            "reason": "original_KADS_MedCPT_and_Reader",
            "reader_calls_total": self.reader.calls - reader_before,
            "planner_calls_total": 0, "retrieval_calls_total": 0,
            "selected_doc_ids": original["selected_doc_ids"],
        }]
        stages = []
        seen_ids = {source_key(p) for p in anchor}
        seen_hashes = {content_fingerprint(p.text) for p in anchor}
        original_ids = {source_key(p) for p in retrieved}
        original_hashes = {content_fingerprint(p.text) for p in retrieved}
        queries: list[str] = []
        fresh_pool: list[Passage] = []
        previous = original
        for turn in range(1, cfg.max_rounds + 1):
            plan = self._query(
                q, generated, turn=turn, previous=queries,
                previous_reasoning=previous["reasoning"],
                recent_sources=fresh_pool)
            query = plan["query"]
            stage = {
                "round": turn, "query": query,
                "query_was_truncated": plan["query_was_truncated"],
                "planning_fallback": plan["planning_fallback"],
                "target_fact_unverified": plan["target_fact_unverified"],
                "pseudo_doc_expansion": plan["pseudo_doc_expansion"],
                "fresh_source_count": 0,
                "fresh_outside_original_top32": 0,
                "fresh_corpus_documents": [],
                "compression_original_chars": 0,
                "compression_output_chars": 0,
                "source_quote_count": 0,
                "rejected_quotes": 0,
                "source_quoted_note": "",
                "prediction": None,
            }
            if query is not None:
                queries.append(query)
                source_list = self.retriever.search(query, k=cfg.retrieval_k)
                seen_this_round = set()
                novel = []
                for doc in source_list:
                    if doc.origin != "retrieved" or not doc.text.strip():
                        continue
                    uid = source_key(doc)
                    hsh = content_fingerprint(doc.text)
                    if (uid in seen_ids or hsh in seen_hashes
                            or uid in seen_this_round or hsh in seen_this_round):
                        continue
                    novel.append(doc)
                    seen_this_round.update((uid, hsh))
                chosen_novel = (
                    self.ranker.rank(q, novel)[:cfg.fresh_per_round]
                    if novel else [])
                stage["fresh_source_count"] = len(chosen_novel)
                stage["fresh_outside_original_top32"] = sum(
                    (source_key(doc) not in original_ids
                     and content_fingerprint(doc.text) not in original_hashes)
                    for doc in chosen_novel)
                stage["fresh_corpus_documents"] = source_records(chosen_novel)
                # Only actually incorporated new docs are marked seen.
                for doc in chosen_novel:
                    seen_ids.add(source_key(doc))
                    seen_hashes.add(content_fingerprint(doc.text))
                if chosen_novel:
                    processed = [
                        compress_source(q, doc,
                                        max_sentences=cfg.extract_sentences,
                                        max_chars=cfg.extract_chars)
                        if cfg.use_recomp else doc
                        for doc in chosen_novel]
                    stage["compression_original_chars"] = sum(
                        len(d.text) for d in chosen_novel)
                    stage["compression_output_chars"] = sum(
                        len(d.text) for d in processed)
                    fresh_pool.extend(processed)
                    # Preserve external AND parametric anchor pathways, but
                    # prevent old KGCC hypotheses from filling all 5 slots.
                    anchored = []
                    first_real = next(
                        (d for d in stable_anchor if d.origin == "retrieved"),
                        None)
                    first_gen = next(
                        (d for d in stable_anchor
                         if d.origin == "parametric_generated"), None)
                    if first_real:
                        anchored.append(first_real)
                    if first_gen:
                        anchored.append(first_gen)
                    for d in stable_anchor:
                        if len(anchored) >= 2:
                            break
                        if d.id not in {x.id for x in anchored}:
                            anchored.append(d)
                    # Always make space for most recently acquired evidence,
                    # then fill remaining source slots from earlier rounds.
                    remaining = cfg.max_reader_docs - len(anchored)
                    fresh_order = processed + [
                        d for d in reversed(fresh_pool)
                        if d.id not in {x.id for x in processed}]
                    view = anchored + fresh_order[:remaining]
                    if len({d.id for d in view}) != len(view):
                        raise AssertionError("Repeated Reader document id")
                    notes = (
                        self._option_notes(q, view)
                        if cfg.use_sure else {
                            "quotes": [], "rejected": 0, "text": ""})
                    result = self._answer(
                        q, view, option_notes=notes["text"])
                    stage.update({
                        "prediction": result["prediction"],
                        "raw_reader_output": result["raw_reader_output"],
                        "selected_doc_ids": result["selected_doc_ids"],
                        "selected_sources": result["selected_sources"],
                        "prompt_sha256": result["prompt_sha256"],
                        "prompt_chars": result["prompt_chars"],
                        "source_quote_count": len(notes["quotes"]),
                        "rejected_quotes": notes["rejected"],
                        "source_quoted_note": notes["text"],
                    })
                    previous = result
            stages.append(stage)
            final_result = previous
            prefixes.append({
                "round": turn, "prediction": final_result["prediction"],
                "base_prediction": original["prediction"],
                "reason": ("new_evidence_reader_answer"
                           if stage["prediction"] is not None
                           else "no_new_evidence_previous_answer"),
                "reader_calls_total": self.reader.calls - reader_before,
                "planner_calls_total": (
                    self.auxiliary.calls - aux_before
                    if self.auxiliary else 0),
                "retrieval_calls_total": (
                    self.retriever.calls - retrieval_before
                    if self.retriever else 0),
                "selected_doc_ids": (
                    stage["selected_doc_ids"]
                    if stage["prediction"] is not None
                    else prefixes[-1]["selected_doc_ids"]),
            })
        return {
            "id": q.id, "mode": "medrgag_abcd",
            "components": {
                "A_query2doc": cfg.use_query2doc,
                "B_ircot": cfg.use_ircot,
                "C_recomp_inspired": cfg.use_recomp,
                "D_sure_inspired": cfg.use_sure,
            },
            "prediction": prefixes[-1]["prediction"],
            "base_prediction": original["prediction"],
            "base_raw_reader_output": original["raw_reader_output"],
            "baseline_prompt_sha256": original["prompt_sha256"],
            "baseline_prompt_chars": original["prompt_chars"],
            "base_selected_doc_ids": original["selected_doc_ids"],
            "base_selected_sources": original["selected_sources"],
            "depth_used": 5 + sum(s["fresh_source_count"] for s in stages),
            "prefixes": prefixes,
            "novel_views": stages,
            "reader_calls": self.reader.calls - reader_before,
            "auxiliary_calls": (
                self.auxiliary.calls - aux_before if self.auxiliary else 0),
            "retrieval_calls": (
                self.retriever.calls - retrieval_before
                if self.retriever else 0),
            "ranker_batches": self.ranker.calls - rank_before,
            "gold_used_at_inference": False,
            "KGCC_as_parametric_hypothesis_not_source": True,
            "source_quote_is_not_truth_verification": True,
            "precomputed_KGCC_KADS_cost_excluded": True,
            "SOTA_or_scaling_guarantee": False,
        }
