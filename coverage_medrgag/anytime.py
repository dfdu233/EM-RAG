"""Anytime evidence selection; a SINGLE training-free MedRGAG modification.

Original MedRGAG retrieval cache + KGCC documents + ORIGINAL KADS selection
are unchanged. At K=5 the selected context is EXACTLY original KADS.
As more ranked sources become available, a fixed question-conditioned
coverage utility proposes alternative 5-doc sets. A retained incumbent
prevents utility regression, without an answer Gate, extra LLM Judge, new
training or repeated KGCC generation. Final Reader executes exactly once
by default; optional prefix answers are only for scientific diagnosis.

A NONDECREASING UTILITY SCORE DOES NOT GUARANTEE NONDECREASING ACCURACY.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Callable

from .data import Passage, Question, reader_prompt
from .models import ChatModel, parse_answer
from .utility import FacetScorer, greedy_coverage, utility


@dataclass(frozen=True)
class Config:
    depths: tuple[int, ...] = (5, 10, 20, 32)
    max_context_docs: int = 5
    read_each_depth: bool = False
    initial_kads_anchor: bool = True
    context_tokens: int = 128000
    max_reader_prompt_chars: int = 150000
    reader_temperature: float = 0.1
    reader_seed: int = 42

    def __post_init__(self):
        if (not self.depths or self.depths[0] != 5 or
                tuple(sorted(set(self.depths))) != self.depths):
            raise ValueError("Depths must be strictly increasing, starting at 5")
        if any(d < 5 for d in self.depths):
            raise ValueError("Depth budgets must be >= 5")
        if self.max_context_docs != 5:
            raise ValueError("Preserve MedRGAG Reader budget of exactly five passages")
        if self.context_tokens < 1 or self.max_reader_prompt_chars < 1:
            raise ValueError("Input budget must be positive")


class AnytimeMedRGAG:
    def __init__(self, *, reader: ChatModel, ranker,
                 rag_template: str,
                 scorer_factory: Callable[[Question], FacetScorer],
                 config: Config | None = None, tokenizer=None):
        self.reader = reader
        self.ranker = ranker
        self.rag_template = rag_template
        self.scorer_factory = scorer_factory
        self.config = config or Config()
        self.tokenizer = tokenizer

    def run(self, q: Question, retrieved: list[Passage],
            generated: list[Passage],
            *, original_selection: tuple[int, ...] | None = None) -> dict:
        cfg = self.config
        if len(retrieved) < 5 or len(generated) < 5:
            raise ValueError("Original MedRGAG needs 5 retrieved + 5 KGCC candidates")
        if len({d.id for d in retrieved + generated[:5]}) != len(retrieved) + 5:
            raise ValueError("Every candidate must have a unique doc ID")
        if cfg.initial_kads_anchor and original_selection is None:
            raise ValueError("MedRGAG original KADS cache is REQUIRED for the anchor")
        if cfg.initial_kads_anchor:
            if (not 1 <= len(original_selection) <= 5 or
                    len(set(original_selection)) != len(original_selection) or
                    any(type(i) is not int or i not in range(10)
                        for i in original_selection)):
                raise ValueError("Invalid original KADS selection")
        scorer = self.scorer_factory(q)
        facets = scorer.facets
        doc_scores: dict[str, tuple[float, ...]] = {}
        reader_start = self.reader.calls
        rank_start = self.ranker.calls
        incumbent: list[Passage] = []
        incumbent_utility = -1.0
        observed = 0
        trace = []
        prefix_predictions = []
        last_proposal_utility = None

        for k_requested in cfg.depths:
            k = min(len(retrieved), k_requested)
            if k <= observed:
                break
            current_retrieved = retrieved[:k]
            available = current_retrieved + generated[:5]
            fresh = [d for d in available if d.id not in doc_scores]
            new_values = scorer.score(fresh)
            if set(new_values) != {d.id for d in fresh}:
                raise ValueError("Facet scorer must score all and only new documents")
            if any(len(vector) != len(facets) for vector in new_values.values()):
                raise ValueError("Facet scorer changed dimension across rounds")
            if any(not all(0 <= v <= 1 for v in vector)
                   for vector in new_values.values()):
                raise ValueError("Facet similarity must be a finite value in [0,1]")
            doc_scores.update(new_values)
            if cfg.initial_kads_anchor and not incumbent:
                initial = retrieved[:5] + generated[:5]
                incumbent = [initial[i] for i in original_selection]
                incumbent_utility = utility(incumbent, doc_scores)

            # K=5 has strict reference parity when original KADS is anchored:
            # extra improvement at K=5 would mix new selection and scaling.
            proposed = (list(incumbent)
                        if cfg.initial_kads_anchor and k_requested == 5
                        else greedy_coverage(
                            available, doc_scores,
                            capacity=cfg.max_context_docs))
            proposal_score = utility(proposed, doc_scores)
            last_proposal_utility = proposal_score
            previous = incumbent_utility
            improved = not incumbent or proposal_score > incumbent_utility + 1e-12
            if improved:
                incumbent = list(proposed)
                incumbent_utility = proposal_score
            assert incumbent_utility + 1e-12 >= previous
            if not incumbent or len(incumbent) > cfg.max_context_docs:
                raise AssertionError("Invalid best-context memory")
            if any(d.id not in {x.id for x in available} for d in incumbent):
                raise AssertionError("Incumbent must be a subset of available documents")
            beyond_five = [
                doc.id for doc in incumbent
                if doc.origin == "retrieved" and doc.id not in {
                    x.id for x in retrieved[:5]}
            ]
            record = {
                "k": k,
                "proposed_doc_ids": [d.id for d in proposed],
                "proposed_utility": proposal_score,
                "best_doc_ids": [d.id for d in incumbent],
                "best_utility": incumbent_utility,
                "incumbent_updated": improved,
                "new_source_doc_ids": beyond_five,
                "scored_new_documents": len(fresh),
            }
            trace.append(record)
            observed = k

            if cfg.read_each_depth:
                output = self._answer(q, incumbent)
                prefix_predictions.append({
                    "k": k, "prediction": output["prediction"],
                    "context_doc_ids": [d.id for d in incumbent],
                    "utility": incumbent_utility,
                    "prompt_sha256": output["prompt_sha256"],
                    "prompt_chars": output["prompt_chars"],
                    "raw_reader_output": output["raw_reader_output"],
                })

        if not trace:
            raise ValueError("No depth stage evaluated")
        final = (self._answer(q, incumbent) if not prefix_predictions else {
            "prediction": prefix_predictions[-1]["prediction"],
            "raw_reader_output": prefix_predictions[-1]["raw_reader_output"],
            "prompt_sha256": prefix_predictions[-1]["prompt_sha256"],
            "prompt_chars": prefix_predictions[-1]["prompt_chars"],
            "ordered_ids": prefix_predictions[-1]["context_doc_ids"],
        })
        previous_scores = [r["best_utility"] for r in trace]
        if any(y + 1e-12 < x for x, y in
               zip(previous_scores, previous_scores[1:])):
            raise AssertionError("Non-monotone best-context utility: invariant violated")

        result = {
            "id": q.id,
            "mode": "anytime_training_free",
            "scorer": getattr(scorer, "name", "unknown"),
            "facets": list(facets),
            "depth_used": trace[-1]["k"],
            "depth_path": trace,
            "estimated_best_utility": incumbent_utility,
            "estimated_best_utility_nondecreasing": True,
            "best_utility_has_correctness_guarantee": False,
            "best_doc_ids_unordered": [d.id for d in incumbent],
            "selected_doc_ids": final["ordered_ids"],
            "selected_sources": [
                next(d.origin for d in incumbent if d.id == doc_id)
                for doc_id in final["ordered_ids"]
            ],
            "selected_original_ids": [
                next(d.original_id for d in incumbent if d.id == doc_id)
                for doc_id in final["ordered_ids"]
            ],
            "new_source_docs_in_reader": trace[-1]["new_source_doc_ids"],
            "scorer_calls": scorer.calls,
            "ranker_batches": self.ranker.calls - rank_start,
            "auxiliary_calls": 0,
            "reader_calls": self.reader.calls - reader_start,
            "read_each_depth": cfg.read_each_depth,
            "prefix_reader_results": prefix_predictions,
            "prediction": final["prediction"],
            "parse_ok": final["prediction"] is not None,
            "raw_reader_output": final["raw_reader_output"],
            "reader_prompt_chars": final["prompt_chars"],
            "reader_prompt_sha256": final["prompt_sha256"],
            "truth_sufficiency": "unknown",
        }
        return result

    def _answer(self, q: Question, selection: list[Passage]) -> dict:
        # This is the ORIGINAL MedRGAG last-stage MedCPT ranking + prompt.
        ranked = self.ranker.rank(q, list(selection))
        prompt = reader_prompt(
            q, ranked, self.rag_template, tokenizer=self.tokenizer,
            context_tokens=self.config.context_tokens,
            max_prompt_chars=self.config.max_reader_prompt_chars)
        raw = self.reader.complete(
            "You are a helpful assistant.", prompt,
            temperature=self.config.reader_temperature,
            seed=self.config.reader_seed, max_tokens=512)
        return {
            "prediction": parse_answer(raw, q.labels),
            "raw_reader_output": raw,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "ordered_ids": [d.id for d in ranked],
        }
