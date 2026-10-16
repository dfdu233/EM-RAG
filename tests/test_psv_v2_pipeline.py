"""End-to-end PSV v2 mock tests. No API, corpus download, or training."""
import json
import pytest

from coverage_medrgag.progressive import Config, ProgressiveSourceViews
from coverage_medrgag.progressive_reparse import reparse_record
from coverage_medrgag.data import Question
from coverage_medrgag.models import IdentityRanker
from tests.test_progressive import inputs, MockPlanner, MockRetriever, MockReader


def build(config, *, reader=None, planner=None, ranker=None):
    reader = reader or MockReader()
    planner = planner or MockPlanner()
    retriever = MockRetriever()
    return ProgressiveSourceViews(
        reader=reader, planner=planner, retriever=retriever,
        ranker=ranker or IdentityRanker(),
        rag_template="Question: {question}\nRetrieved: {documents}",
        config=config), reader, planner, retriever


def test_source_only_anchor_keeps_exact_original_B0_and_matched_budget():
    engine, reader, planner, retriever = build(Config(
        max_rounds=2, anchor_mode="retrieved_only",
        policy="contrastive_vote", paired_source_control=True))
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen,
                        original_selection=(0, 5, 1, 6, 2))
    assert result["base_selected_doc_ids"] == [
        "r0", "g0", "r1", "g1", "r2"]
    assert result["base_prediction"] == "A"
    assert result["anchor_doc_ids"] == ["r0", "r1", "r2"]
    assert result["anchor_generated_count"] == 0
    assert result["matched_source_control"]["selected_doc_ids"] == [
        "r0", "r1", "r2", "r3", "r4"]
    assert result["matched_source_control"]["prediction"] == "A"
    assert result["matched_source_control"]["reused_B0"] is False
    assert all(s["anchor_provenance"]["anchor_generated_count"] == 0
               for s in result["novel_views"])
    assert all(len(s["selected_doc_ids"]) == 5 for s in result["novel_views"])
    assert [x["prediction"] for x in result["prefixes"]] == ["A", "A", "B"]
    assert result["prefixes"][-1]["reason"] == "source_contrast_consensus"
    assert result["reader_calls"] == 4  # B0 + one control + two source views
    assert result["retrieval_calls"] == 2
    assert result["auxiliary_calls"] == 2


class ModelWithSharedBackgroundError(MockReader):
    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.prompts.append(user)
        return ('{"answer_choice":"A"}'
                if "Generated KGCC hypothesis_" in user
                else '{"answer_choice":"B"}')


def test_contrastive_rule_rejects_agreement_also_found_without_new_evidence():
    reader = ModelWithSharedBackgroundError()
    engine, *_ = build(Config(max_rounds=2,
                              anchor_mode="retrieved_only",
                              paired_source_control=True,
                              policy="contrastive_vote"), reader=reader)
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen,
                        original_selection=(0, 5, 1, 6, 2))
    assert result["base_prediction"] == "A"
    assert result["matched_source_control"]["prediction"] == "B"
    assert [stage["prediction"] for stage in result["novel_views"]] == ["B", "B"]
    assert result["prediction"] == "A"
    assert result["prefixes"][-1]["reason"] == (
        "change_also_present_without_new_sources")
    assert result["reader_calls"] == 4


def test_historical_original_anchor_reuses_B0_without_extra_reader():
    engine, reader, planner, retrieval = build(Config(
        max_rounds=2, anchor_mode="original", paired_source_control=True))
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen,
                        original_selection=(0, 5, 1, 6, 2))
    assert result["matched_source_control"]["reused_B0"] is True
    assert result["reader_calls"] == 3
    assert result["anchor_generated_count"] == 1


class LongPlanner(MockPlanner):
    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.requests.append((system, user))
        return json.dumps({
            "query": "clinical trial general context unrelated " * 13
                     + "adjusted risk ratio after controlling grade",
            "target_relation": "adjusted effect controlling grade"})


def test_query_repair_uses_same_planner_call_but_preserves_audit():
    engine, reader, planner, retriever = build(
        Config(max_rounds=1, query_size_policy="compact"),
        planner=LongPlanner())
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen,
                        original_selection=(0, 5, 1, 6, 2))
    stage = result["novel_views"][0]
    assert stage["query_provenance"]["status"] == "compacted_unverified"
    assert len(stage["query"]) <= 300
    assert "adjusted" in stage["query"]
    assert "controlling" in stage["query"]
    assert result["auxiliary_calls"] == 1
    assert result["retrieval_calls"] == 1

    rejection, *_ = build(
        Config(max_rounds=1, query_size_policy="reject"), planner=LongPlanner())
    declined = rejection.run(q, ret, gen,
                             original_selection=(0, 5, 1, 6, 2))
    assert declined["retrieval_calls"] == 0
    assert declined["novel_views"][0]["query_provenance"]["status"] == (
        "over_limit_rejected")


class LogRanker(IdentityRanker):
    def __init__(self):
        super().__init__()
        self.query_texts = []

    def rank(self, q, docs):
        self.query_texts.append(q.text)
        return super().rank(q, docs)


def test_target_relation_ranking_is_opt_in_and_reader_prompt_unchanged():
    ranker = LogRanker()
    engine, *_ = build(Config(max_rounds=1, fresh_ranking="target_relation"),
                       ranker=ranker)
    q, ret, gen = inputs()
    result = engine.run(q, ret, gen,
                        original_selection=(0, 5, 1, 6, 2))
    assert any("Target relation:" in x for x in ranker.query_texts)
    assert result["fresh_ranking"] == "target_relation"
    assert result["novel_views"][0]["fresh_ranking"] == "target_relation"


def test_no_gold_frozen_reparse_recomputes_all_vote_prefixes():
    engine, reader, planner, retriever = build(Config(max_rounds=2))
    q, ret, gen = inputs()
    row = engine.run(q, ret, gen,
                     original_selection=(0, 5, 1, 6, 2))
    original_prompt_hash = row["baseline_prompt_sha256"]
    row["base_raw_reader_output"] = "Final Answer: B"
    # Both additional views vote A; original mock outputs B.
    for stage in row["novel_views"]:
        stage["raw_reader_output"] = "Final Answer: A"
    reparsed = reparse_record(row, labels=q.labels)
    assert reparsed["base_prediction"] == "B"
    assert [x["prediction"] for x in reparsed["prefixes"]] == ["B", "B", "A"]
    assert reparsed["prediction"] == "A"
    assert reparsed["baseline_prompt_sha256"] == original_prompt_hash
    assert reparsed["reader_calls"] == row["reader_calls"]
    assert row["base_prediction"] == "A"  # original frozen record untouched


def test_source_control_requires_matched_document_count():
    with pytest.raises(ValueError, match="five documents"):
        Config(fresh_docs_per_view=1, anchor_docs_per_view=3,
               paired_source_control=True)
    with pytest.raises(ValueError, match="requires matched"):
        Config(policy="contrastive_vote")
    with pytest.raises(ValueError, match="anchor"):
        Config(anchor_mode="nonsense")
