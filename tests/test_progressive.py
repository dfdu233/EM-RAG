"""Scientific regression tests: real new queries, KGCC provenance, no oracle.

All models, BM25 results and reranker are FAKE. This only checks protocol
integrity, not whether the algorithm improves real biomedical answers.
"""
import pytest

from coverage_medrgag.data import Passage, Question
from coverage_medrgag.models import IdentityRanker
from coverage_medrgag.progressive import (
    Config, ProgressiveSourceViews, aggregate, content_fingerprint,
    source_key,
)


class MockPlanner:
    def __init__(self, *, duplicate=False, malformed=False):
        self.calls = 0
        self.requests = []
        self.duplicate = duplicate
        self.malformed = malformed

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.requests.append((system, user))
        if self.malformed:
            return "not-json"
        if self.duplicate:
            return '{"query":"precise unusual controlled relationship round one", "target_relation":"r"}'
        return (
            '{"query":"precise unusual controlled relationship round '
            + str(self.calls) + '", "target_relation":"adjusted independent effect"}')


class MockRetriever:
    def __init__(self, *, no_new=False):
        self.calls = 0
        self.queries = []
        self.no_new = no_new

    def search(self, query, *, k):
        self.calls += 1
        self.queries.append((query, k))
        if self.no_new:
            return [Passage("old", "original document 0", "retrieved", "old0")]
        return [
            Passage(f"n{self.calls}_0",
                    f"PROOF_B_round_{self.calls}_first has condition support",
                    "retrieved", f"source{self.calls}a"),
            Passage(f"n{self.calls}_1",
                    f"PROOF_B_round_{self.calls}_second has condition support",
                    "retrieved", f"source{self.calls}b"),
            # Duplicate first document, different internal index.
            Passage(f"n{self.calls}_2",
                    f"PROOF_B_round_{self.calls}_first has condition support",
                    "retrieved", f"another_doc_id{self.calls}"),
        ]


class MockReader:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.prompts.append(user)
        choice = "B" if "PROOF_B_round" in user else "A"
        return '{"answer_choice":"' + choice + '"}'


def inputs():
    q = Question("ex1", "Does fictional risk remain after controlling for age?",
                 {"A": "Yes", "B": "No"})
    docs = [Passage(f"r{i}", f"original document {i}",
                    "retrieved", f"old{i}") for i in range(32)]
    kgcc = [Passage(f"g{i}", f"Generated KGCC hypothesis_{i}",
                    "parametric_generated", f"gen{i}") for i in range(5)]
    return q, docs, kgcc


def make_system(*, rounds=2, policy="anchored_vote", planner_mode="kgcc",
                duplicate=False, no_new=False, malformed=False):
    reader, planner = MockReader(), MockPlanner(
        duplicate=duplicate, malformed=malformed)
    retrieval = MockRetriever(no_new=no_new)
    engine = ProgressiveSourceViews(
        reader=reader,
        planner=planner if rounds else None,
        retriever=retrieval if rounds else None,
        ranker=IdentityRanker(),
        rag_template="Question: {question}\nRetrieved: {documents}",
        config=Config(max_rounds=rounds, policy=policy,
                      planner_mode=planner_mode))
    return engine, reader, planner, retrieval


def test_kads_reference_exact_at_zero_budget_without_new_calls():
    engine, reader, planner, retrieval = make_system(rounds=0)
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert result["prediction"] == "A"
    assert result["base_selected_doc_ids"] == ["r0", "g0", "r1", "g1", "r2"]
    assert result["reader_calls"] == 1
    assert result["auxiliary_calls"] == 0
    assert result["retrieval_calls"] == 0
    assert len(result["prefixes"]) == 1
    assert result["prefixes"][0]["round"] == 0
    assert "Generated KGCC hypothesis_0" in reader.prompts[0]


def test_actual_new_source_views_need_two_votes_before_overriding():
    engine, reader, planner, retrieval = make_system()
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert len(result["prefixes"]) == 3
    assert [x["prediction"] for x in result["prefixes"]] == ["A", "A", "B"]
    assert result["prediction"] == "B"
    assert result["reader_calls"] == 3
    assert result["auxiliary_calls"] == 2
    assert result["retrieval_calls"] == 2
    assert result["novel_views"][0]["fresh_source_count"] == 2
    assert result["novel_views"][1]["fresh_source_count"] == 2
    assert result["novel_views"][0]["fresh_outside_original_top32"] == 2
    assert result["novel_views"][1]["fresh_outside_original_top32"] == 2
    assert result["source_groups_are_independent"] is False
    assert len(set(
        d["original_id"]
        for stage in result["novel_views"]
        for d in stage["fresh_corpus_documents"])) == 4
    assert all("parametric" not in d["origin"] for stage in result["novel_views"]
               for d in stage["fresh_corpus_documents"])
    assert len(result["novel_views"][0]["selected_doc_ids"]) == 5
    assert "PARAMETRIC KGCC IDEAS" in planner.requests[0][1]
    assert "Generated KGCC hypothesis" in planner.requests[0][1]
    assert "UNVERIFIED READER ANSWER DISAGREEMENT" in planner.requests[1][1]
    assert "alternate_views=['B']" in planner.requests[1][1]
    assert "ORIGINAL QUESTION" not in planner.requests[0][1]  # no hidden gold
    assert retrieval.queries[0][1] == 8


def test_question_only_ablation_does_not_get_kgcc_query_hints():
    engine, reader, planner, retrieval = make_system(planner_mode="question")
    q, ret, gen = inputs()
    engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert "PARAMETRIC KGCC IDEAS" not in planner.requests[0][1]
    assert "Generated KGCC hypothesis" not in planner.requests[0][1]


def test_failed_query_planning_keeps_original_without_fake_extra_evidence():
    engine, reader, planner, retrieval = make_system(malformed=True)
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert result["prediction"] == result["base_prediction"] == "A"
    assert result["retrieval_calls"] == 0
    assert result["reader_calls"] == 1
    assert [v["query"] for v in result["novel_views"]] == [None, None]


def test_repeat_query_is_not_retrieved_twice():
    engine, reader, planner, retrieval = make_system(duplicate=True)
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert result["retrieval_calls"] == 1
    assert result["reader_calls"] == 2
    assert result["novel_views"][-1]["query"] is None
    assert result["prediction"] == "A"


def test_identical_results_with_no_new_source_skip_extra_reader_calls():
    engine, reader, planner, retrieval = make_system(no_new=True)
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert result["reader_calls"] == 1
    assert result["retrieval_calls"] == 2
    assert all(v["fresh_source_count"] == 0 for v in result["novel_views"])
    assert result["prediction"] == "A"


def test_latest_policy_is_a_separate_optimistic_ablation():
    engine, reader, planner, retrieval = make_system(policy="latest")
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen, original_selection=(0, 5, 1, 6, 2))
    assert [x["prediction"] for x in result["prefixes"]] == ["A", "B", "B"]


def test_aggregation_is_not_a_correctness_oracle():
    assert aggregate("A", ["B"], policy="anchored_vote")["prediction"] == "A"
    assert aggregate("A", ["B", "B"], policy="anchored_vote")["prediction"] == "B"
    assert aggregate("A", ["B", "A", "B"], policy="anchored_vote")["prediction"] == "A"
    assert aggregate("A", [None, None])["prediction"] == "A"
    assert aggregate(None, ["B"])["prediction"] == "B"
    assert aggregate("A", [], policy="latest")["prediction"] == "A"


def test_invalid_config_and_gold_free_protocol():
    with pytest.raises(ValueError):
        Config(fresh_docs_per_view=4, anchor_docs_per_view=3)
    with pytest.raises(ValueError):
        Config(max_rounds=9)
    engine, *_ = make_system()
    q, ret, gen = inputs()
    with pytest.raises(ValueError, match="KADS"):
        engine.run(q, ret, gen, original_selection=None)
    assert source_key(ret[0]) == "id:old0"
    assert content_fingerprint(ret[0].text) == content_fingerprint(
        " original   document 0  ")
