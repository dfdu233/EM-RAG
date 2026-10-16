"""Progressive Source Views: a training-free, falsifiable MedRGAG experiment.

Unlike prefix-only CA-MedRGAG/TF-AES this actually issues NEW BM25 searches
whose queries are motivated by the original KGCC knowledge proposals.
Unlike a single concatenated context, every new disjoint source group gets its
OWN 5-document frozen Reader call; we log every prefix decision and compare
source-view agreement against the original KADS answer.

DESIGN CONTRACT:
* B=0 is the exact frozen MedRGAG KADS context and Reader prompt.
* KGCC is a parametric HYPOTHESIS / query aid, never independent verification.
* Each query group must contribute previously unseen external passages.
* Context is capped at five documents, preserving up to three KADS anchors.
* There is no evidence truth oracle, learned gate or calibrated probability.
* Source-disjoint is NOT statistical independence. Agreement can be wrong.
* Only held-out REAL accuracy-cost curves can establish actual scaling.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from hashlib import sha256
import re
from typing import Protocol

from .data import Question, Passage, reader_prompt
from .models import ChatModel, parse_json
from .progressive_validity import fit_query, recover_answer, view_provenance


class SourceRetriever(Protocol):
    calls: int
    def search(self, query: str, *, k: int) -> list[Passage]: ...


@dataclass(frozen=True)
class Config:
    max_rounds: int = 2
    retrieval_k: int = 8
    fresh_docs_per_view: int = 2
    anchor_docs_per_view: int = 3
    max_reader_docs: int = 5
    planner_mode: str = "kgcc"  # ablation: question-only planner
    policy: str = "anchored_vote"  # optional contrastive_vote/source-only
    anchor_mode: str = "original"  # retrieved_only isolates KGCC anchor effects
    paired_source_control: bool = False  # matched old-doc context; extra Reader call
    query_size_policy: str = "compact"  # reject reproduces historical failed plans
    fresh_ranking: str = "question"  # target_relation is a preregistered ablation
    planner_excerpt_chars: int = 700
    context_tokens: int = 128000
    max_prompt_chars: int = 150000
    temperature: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if not 0 <= self.max_rounds <= 8:
            raise ValueError("max_rounds must be in [0,8]")
        if not 1 <= self.retrieval_k <= 128:
            raise ValueError("retrieval_k must be in [1,128]")
        if self.max_reader_docs != 5 or self.fresh_docs_per_view < 1:
            raise ValueError("Original Reader budget must remain five passages")
        if (self.fresh_docs_per_view + self.anchor_docs_per_view > 5 or
                self.anchor_docs_per_view < 1):
            raise ValueError("Anchor + fresh docs must fit within five")
        if self.planner_mode not in ("kgcc", "question"):
            raise ValueError("Unknown planner ablation")
        if self.policy not in ("anchored_vote", "latest", "contrastive_vote"):
            raise ValueError("Unknown aggregation policy")
        if self.anchor_mode not in ("original", "retrieved_only"):
            raise ValueError("Unknown anchor control")
        if self.query_size_policy not in ("compact", "reject"):
            raise ValueError("Unknown query-size strategy")
        if self.fresh_ranking not in ("question", "target_relation"):
            raise ValueError("Unknown fresh-source ranking")
        if self.policy == "contrastive_vote" and not self.paired_source_control:
            raise ValueError("Contrastive vote requires matched source control")
        if self.paired_source_control and (
                self.fresh_docs_per_view + self.anchor_docs_per_view != 5):
            raise ValueError("Matched comparison needs five documents per view")
        if self.context_tokens < 1 or self.max_prompt_chars < 1:
            raise ValueError("Invalid Reader context budget")


def content_fingerprint(text: str) -> str:
    return sha256(" ".join(text.split()).casefold().encode("utf-8")).hexdigest()


def source_key(doc: Passage) -> str:
    """Prefer original corpus IDs; always track text hashes independently."""
    if doc.original_id.strip() and doc.original_id.strip() not in ("None", "null"):
        return "id:" + doc.original_id.strip()
    return "sha:" + content_fingerprint(doc.text)


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def source_records(docs: list[Passage]) -> list[dict]:
    return [{
        "id": d.id, "original_id": d.original_id, "origin": d.origin,
        "title": d.title, "text_sha256": content_fingerprint(d.text),
        "text": d.text,
    } for d in docs]


def aggregate(base: str | None, views: list[str | None],
              *, policy: str = "anchored_vote") -> dict:
    """Non-oracular answer aggregation. No correctness guarantee.

    To override a parseable KADS answer, two NEW source-disjoint views must
    agree on the alternative, with strictly more votes than the current
    baseline and every competing label. On a tie, preserve original KADS.
    """
    if policy == "latest":
        valid = [x for x in views if x is not None]
        return {
            "prediction": valid[-1] if valid else base,
            "counts": dict(Counter(([base] if base else []) + valid)),
            "reason": "latest_view_ablation" if valid else "no_valid_view",
        }
    if policy != "anchored_vote":
        raise ValueError(f"Unknown policy {policy}")
    counts = Counter([x for x in views if x is not None])
    if base is None:
        # The baseline is unparseable: use first valid view for recovery.
        # Mark this explicitly; do not silently substitute a label.
        prediction = next((x for x in views if x is not None), None)
        if counts and counts.most_common(1)[0][1] >= 2:
            best_count = max(counts.values())
            leaders = [label for label, n in counts.items() if n == best_count]
            prediction = leaders[0] if len(leaders) == 1 else prediction
        return {"prediction": prediction, "counts": dict(counts),
                "reason": "unparseable_baseline_fallback"}
    counts[base] += 1
    others = {key: n for key, n in counts.items() if key != base}
    if not others:
        return {"prediction": base, "counts": dict(counts),
                "reason": "no_valid_alternative"}
    best_count = max(others.values())
    winners = [key for key, n in others.items()
               if n == best_count and n >= 2 and n > counts[base]]
    if len(winners) == 1:
        return {"prediction": winners[0], "counts": dict(counts),
                "reason": "two_disjoint_source_views_overrule"}
    return {"prediction": base, "counts": dict(counts),
            "reason": "anchored_ambiguous_or_insufficient_support"}


PLANNER_SYSTEM = (
    "You plan ONE biomedical corpus search. A generated knowledge snippet "
    "is a conjecture, NOT a cited or independently verified fact. "
    "Do NOT assert any answer choice is correct. "
    "Identify one concrete, unresolved task-specific relationship or "
    "study condition that would help distinguish the original options. "
    "Search for actual data about the relationship, including possible "
    "negative findings and confounders. The query MUST be <=240 characters "
    "(hard maximum 300), using precise search terms and preserving important "
    "negation, population and adjustment conditions. "
    "The next query must NOT duplicate "
    "any prior query. Use full-text medical search terminology. "
    'Reply in JSON exactly: {"query":"...","target_relation":"..."}'
)


class ProgressiveSourceViews:
    def __init__(self, *, reader: ChatModel, planner: ChatModel | None,
                 retriever: SourceRetriever | None, ranker,
                 rag_template: str, config: Config | None = None,
                 tokenizer=None):
        self.reader = reader
        self.planner = planner
        self.retriever = retriever
        self.ranker = ranker
        self.rag_template = rag_template
        self.cfg = config or Config()
        self.tokenizer = tokenizer
        if self.cfg.max_rounds and (planner is None or retriever is None):
            raise ValueError("Live rounds require planner and real/replayed retriever")

    def _answer(self, q: Question, docs: list[Passage]) -> dict:
        chosen = self.ranker.rank(q, list(docs))[:self.cfg.max_reader_docs]
        prompt = reader_prompt(
            q, chosen, self.rag_template, tokenizer=self.tokenizer,
            context_tokens=self.cfg.context_tokens,
            max_prompt_chars=self.cfg.max_prompt_chars)
        raw = self.reader.complete(
            "You are a helpful assistant.", prompt,
            temperature=self.cfg.temperature, seed=self.cfg.seed, max_tokens=512)
        parsed = recover_answer(raw, q.labels)
        return {
            "prediction": parsed.prediction,
            "parse_method": parsed.method,
            "raw_reader_output": raw,
            "selected_doc_ids": [d.id for d in chosen],
            "selected_original_ids": [d.original_id for d in chosen],
            "selected_sources": [d.origin for d in chosen],
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
        }

    def _plan_query(self, q: Question, kgcc: list[Passage],
                    previous_queries: list[str],
                    previous_sources: list[Passage], turn: int,
                    baseline_answer: str | None,
                    prior_view_answers: list[str | None]
                    ) -> tuple[str | None, str | None, str, dict]:
        if self.planner is None:
            raise AssertionError("Planner not supplied")
        background = ""
        if self.cfg.planner_mode == "kgcc":
            background = "\n\nPARAMETRIC KGCC IDEAS (NOT verified evidence):\n"
            background += "\n".join(
                f"- {g.text[:self.cfg.planner_excerpt_chars]}"
                for g in kgcc[:5])
        source_text = "\n".join(
            f"[{x.original_id}] {x.text[:self.cfg.planner_excerpt_chars]}"
            for x in previous_sources[-4:])
        divergent = sorted({
            label for label in prior_view_answers
            if label is not None and label != baseline_answer
        })
        disagreement = (
            "\nUNVERIFIED READER ANSWER DISAGREEMENT (not truth): "
            + f"reference={baseline_answer}; alternate_views={divergent}. "
            + "Seek independently retrieved facts that could distinguish "
              "the competing choices WITHOUT presuming either correct.\n"
            if divergent else
            "\nNo current answer disagreement; this may still be a correlated error.\n"
        )
        prompt = (
            "MEDICAL QUESTION AND OPTIONS:\n" + q.formatted()
            + disagreement
            + background
            + "\nPRIOR CORPUS QUERIES:\n"
            + "\n".join(previous_queries)
            + "\nNEWLY DISCOVERED EXTERNAL SOURCES (untrusted):\n"
            + (source_text or "None yet")
            + f"\nROUND: {turn}\n"
            + "Select exactly one NEW search query of no more than 240 "
              "characters, which preserves the medically decisive condition. "
              "Do not use the gold answer, assume a diagnosis, or copy "
              "background conjectures as fact.")
        raw = self.planner.complete(
            PLANNER_SYSTEM, prompt, temperature=0.0,
            seed=self.cfg.seed, max_tokens=320)
        try:
            obj = parse_json(raw)
            query = obj.get("query")
            relation = obj.get("target_relation")
            if not isinstance(query, str):
                raise ValueError("query_field_missing_or_invalid")
            if not isinstance(relation, str) or not relation.strip():
                raise ValueError("target_relation_missing")
            normalized = fit_query(
                query, relation=relation, question=q.text,
                max_chars=300, strategy=self.cfg.query_size_policy)
            info = {
                "status": normalized.status,
                "original_query": normalized.original_query,
                "original_length": normalized.original_length,
                "final_length": normalized.final_length,
                "max_chars": 300,
            }
            if normalized.query is None:
                return None, None, raw, info
            if (normalize_query(normalized.query) == normalize_query(q.text) or
                    normalize_query(normalized.query) in {
                        normalize_query(x) for x in previous_queries}):
                info["status"] = "duplicate_query_after_repair"
                return None, None, raw, info
            return normalized.query, relation.strip(), raw, info
        except ValueError as exc:
            return None, None, raw, {
                "status": f"malformed_plan:{str(exc)}",
                "original_query": None,
                "original_length": 0, "final_length": 0,
                "max_chars": 300,
            }

    def run(self, q: Question, retrieved: list[Passage],
            generated: list[Passage], *,
            original_selection: tuple[int, ...] | None) -> dict:
        cfg = self.cfg
        if len(retrieved) < 5 or len(generated) < 5:
            raise ValueError("MedRGAG cache must contain 5 retrieved + 5 KGCC")
        if (original_selection is None or
                not 1 <= len(original_selection) <= 5 or
                len(set(original_selection)) != len(original_selection) or
                any(type(i) is not int or not 0 <= i < 10
                    for i in original_selection)):
            raise ValueError("Original KADS final_list is required and must be valid")
        base_candidates = retrieved[:5] + generated[:5]
        anchor = [base_candidates[i] for i in original_selection]
        if len({doc.id for doc in anchor}) != len(anchor):
            raise ValueError("Duplicate original anchor document IDs")
        rcalls = self.reader.calls
        pcalls = self.planner.calls if self.planner else 0
        retrieval_calls = self.retriever.calls if self.retriever else 0
        ranker_calls = self.ranker.calls
        base = self._answer(q, anchor)
        # Use the ORIGINAL final MedCPT ordering as the stable anchor,
        # rather than the raw historical KADS index ordering.
        by_id = {doc.id: doc for doc in anchor}
        stable_anchor = [by_id[key] for key in base["selected_doc_ids"]]
        if cfg.anchor_mode == "retrieved_only":
            # Remove KGCC from the NEW view, not from B0, to identify whether
            # shared generated background explains correlated Reader errors.
            candidates = [d for d in stable_anchor if d.origin == "retrieved"]
            candidates += [d for d in retrieved[:5]
                           if d.origin == "retrieved"]
            anchor_docs = []
            seen_anchor = set()
            for d in candidates:
                if d.id not in seen_anchor:
                    anchor_docs.append(d)
                    seen_anchor.add(d.id)
                if len(anchor_docs) == cfg.anchor_docs_per_view:
                    break
        else:
            anchor_docs = stable_anchor[:cfg.anchor_docs_per_view]
        if len(anchor_docs) != cfg.anchor_docs_per_view:
            raise ValueError("Insufficient original source anchors for this view")
        # Fixed-old-source counterfactual for the same anchor, document count
        # and frozen Reader. With an original KADS anchor this is exactly B0.
        control_fill = []
        seen_control = {x.id for x in anchor_docs}
        for candidate in stable_anchor + retrieved[:5] + generated[:5]:
            if candidate.id not in seen_control:
                control_fill.append(candidate)
                seen_control.add(candidate.id)
            if len(control_fill) >= cfg.fresh_docs_per_view:
                break
        control_docs = anchor_docs + control_fill
        matched_control = None
        prefixes = [{
            "round": 0, "prediction": base["prediction"],
            "base_prediction": base["prediction"],
            "base_parse_method": base["parse_method"],
            "reason": "original_KADS_reference",
            "reader_calls_total": self.reader.calls - rcalls,
            "planner_calls_total": 0, "retrieval_calls_total": 0,
            "selected_doc_ids": base["selected_doc_ids"],
        }]
        queries: list[str] = []
        new_views: list[dict] = []
        ever_seen_ids = {source_key(d) for d in anchor}
        ever_seen_hashes = {content_fingerprint(d.text) for d in anchor}
        original_32_ids = {source_key(d) for d in retrieved}
        original_32_hashes = {content_fingerprint(d.text) for d in retrieved}
        discovered: list[Passage] = []
        vote_preds: list[str | None] = []
        for turn in range(1, cfg.max_rounds + 1):
            query, target_relation, raw_plan, plan_info = self._plan_query(
                q, generated, queries, discovered, turn,
                base["prediction"], vote_preds)
            stage: dict = {
                "round": turn, "query": query, "query_provenance": plan_info,
                "anchor_mode": cfg.anchor_mode,
                "anchor_provenance": view_provenance(
                    anchor_docs, [], stable_anchor),
                "target_relation_unverified": target_relation,
                "raw_planner_output": raw_plan,
                "fresh_source_count": 0,
                "fresh_outside_original_top32": 0,
                "fresh_corpus_documents": [],
                "prediction": None,
            }
            if query is not None:
                queries.append(query)
                raw_documents = self.retriever.search(query, k=cfg.retrieval_k)
                # Rank only NEW real retrieved documents for the current view.
                # Duplicate source IDs and duplicate normalized full text count
                # as previously seen, even when another query returns them.
                unique: list[Passage] = []
                local_ids: set[str] = set()
                local_hashes: set[str] = set()
                for d in raw_documents:
                    if not d.text.strip() or d.origin != "retrieved":
                        continue
                    key = source_key(d)
                    hsh = content_fingerprint(d.text)
                    if (key in ever_seen_ids or hsh in ever_seen_hashes
                            or key in local_ids or hsh in local_hashes):
                        continue
                    unique.append(d)
                    local_ids.add(key)
                    local_hashes.add(hsh)
                scoring_query = (
                    Question(q.id, q.text + "\nTarget relation: " +
                             target_relation[:240], q.options)
                    if cfg.fresh_ranking == "target_relation" and target_relation
                    else q)
                ranked_fresh = self.ranker.rank(scoring_query, unique) if unique else []
                fresh = ranked_fresh[:cfg.fresh_docs_per_view]
                stage["fresh_ranking"] = cfg.fresh_ranking
                stage["fresh_source_count"] = len(fresh)
                stage["fresh_outside_original_top32"] = sum(
                    source_key(d) not in original_32_ids and
                    content_fingerprint(d.text) not in original_32_hashes
                    for d in fresh)
                stage["fresh_corpus_documents"] = source_records(fresh)
                if fresh:
                    # Every view is source-disjoint from previous NEW views.
                    # Explicitly preserve up to 3 original KADS documents,
                    # including its parametric knowledge when KADS chose it.
                    view_context = anchor_docs + fresh
                    if len(view_context) > cfg.max_reader_docs:
                        raise AssertionError("Context budget exceeded")
                    stage["anchor_provenance"] = view_provenance(
                        anchor_docs, fresh, stable_anchor)
                    # Only run a matched control when BOTH inputs have five
                    # passages. Otherwise count differences confound evidence.
                    if (cfg.paired_source_control and matched_control is None
                            and len(view_context) == 5 and len(control_docs) == 5):
                        if [d.id for d in control_docs] == [
                                d.id for d in stable_anchor]:
                            matched_control = dict(base)
                            matched_control["reused_B0"] = True
                        else:
                            matched_control = self._answer(q, control_docs)
                            matched_control["reused_B0"] = False
                        matched_control["anchor_doc_ids"] = [
                            d.id for d in anchor_docs]
                        matched_control["old_fill_doc_ids"] = [
                            d.id for d in control_fill]
                    result = self._answer(q, view_context)
                    stage.update({
                        "parse_method": result["parse_method"],
                        "control_prediction": (
                            matched_control["prediction"] if matched_control
                            else None),
                        "differs_from_matched_control": (
                            result["prediction"] != matched_control["prediction"]
                            if matched_control and result["prediction"] is not None
                            and matched_control["prediction"] is not None
                            else None),
                        "prediction": result["prediction"],
                        "raw_reader_output": result["raw_reader_output"],
                        "selected_doc_ids": result["selected_doc_ids"],
                        "selected_original_ids": result["selected_original_ids"],
                        "selected_sources": result["selected_sources"],
                        "prompt_sha256": result["prompt_sha256"],
                        "prompt_chars": result["prompt_chars"],
                    })
                    vote_preds.append(result["prediction"])
                    discovered.extend(fresh)
                    ever_seen_ids.update(source_key(d) for d in fresh)
                    ever_seen_hashes.update(
                        content_fingerprint(d.text) for d in fresh)
            new_views.append(stage)
            voted = aggregate(
                base["prediction"], vote_preds,
                policy=("anchored_vote" if cfg.policy == "contrastive_vote"
                        else cfg.policy))
            if (cfg.policy == "contrastive_vote" and base["prediction"] is not None
                    and voted["prediction"] != base["prediction"]):
                if (matched_control is None or
                        matched_control["prediction"] is None):
                    voted = dict(voted, prediction=base["prediction"],
                                 reason="matched_control_unavailable")
                elif matched_control["prediction"] == voted["prediction"]:
                    voted = dict(voted, prediction=base["prediction"],
                                 reason="change_also_present_without_new_sources")
                else:
                    voted = dict(voted, reason="source_contrast_consensus")
            prefixes.append({
                "round": turn,
                "prediction": voted["prediction"],
                "base_prediction": base["prediction"],
                "reason": voted["reason"],
                "vote_counts": voted["counts"],
                "valid_view_count": sum(x is not None for x in vote_preds),
                "queries_used_total": len(queries),
                "reader_calls_total": self.reader.calls - rcalls,
                "planner_calls_total": self.planner.calls - pcalls,
                "retrieval_calls_total": self.retriever.calls - retrieval_calls,
                "outside_original_top32_cumulative": sum(
                    x["fresh_outside_original_top32"] for x in new_views),
                "selected_doc_ids": (
                    base["selected_doc_ids"] if voted["prediction"] ==
                    base["prediction"] else
                    [x.get("selected_doc_ids", []) for x in new_views
                     if x.get("prediction") == voted["prediction"]][-1]),
            })
        final = prefixes[-1]
        return {
            "id": q.id,
            "mode": "progressive_source_views",
            "policy": cfg.policy,
            "planner_mode": cfg.planner_mode,
            "prediction": final["prediction"],
            "base_prediction": base["prediction"],
            "baseline_prompt_sha256": base["prompt_sha256"],
            "baseline_prompt_chars": base["prompt_chars"],
            "base_raw_reader_output": base["raw_reader_output"],
            "base_parse_method": base["parse_method"],
            "anchor_mode": cfg.anchor_mode,
            "anchor_doc_ids": [d.id for d in anchor_docs],
            "anchor_generated_count": sum(
                d.origin == "parametric_generated" for d in anchor_docs),
            "query_size_policy": cfg.query_size_policy,
            "fresh_ranking": cfg.fresh_ranking,
            "matched_source_control": matched_control,
            "base_selected_doc_ids": base["selected_doc_ids"],
            "base_selected_sources": base["selected_sources"],
            "depth_used": 5 + sum(x["fresh_source_count"] for x in new_views),
            "prefixes": prefixes,
            "novel_views": new_views,
            "reader_calls": self.reader.calls - rcalls,
            "auxiliary_calls": self.planner.calls - pcalls if self.planner else 0,
            "retrieval_calls": (
                self.retriever.calls - retrieval_calls if self.retriever else 0),
            "ranker_batches": self.ranker.calls - ranker_calls,
            "source_groups_are_independent": False,
            "new_sources_can_be_false_or_misleading": True,
            "actual_scaling_not_guaranteed": True,
            "gold_used_at_inference": False,
            "precomputed_KGCC_KADS_cost_excluded": True,
        }
