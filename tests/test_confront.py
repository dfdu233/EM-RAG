"""Counterexamples and scientific invariants for the literature-fusion baseline.

The mocks are not evidence of medical accuracy; they test quote provenance,
query safety, prior/source independence and the exact B0 MedRGAG anchor.
"""
import json
import re
import pytest

from coverage_medrgag.confront import (
    Config, EvidenceConfrontedMedRGAG, CONFRONT_SYSTEM,
    SOURCE_SYSTEM, bounded_query, get_choice, verified_quotes,
)
from coverage_medrgag.data import Passage, Question
from coverage_medrgag.models import IdentityRanker


Q = Question("q", "Does therapy retain effect after adjustment?",
             {"A": "Yes", "B": "No"})
PASSAGES = [
    Passage(f"r{i}", f"Old source passage number {i} general medicine",
            "retrieved", f"original{i}") for i in range(32)
]
GENERATED = [
    Passage(f"g{i}", f"KGCC GENERATED HYPOTHESIS_{i}",
            "parametric_generated", f"generated{i}") for i in range(5)
]
FACT = "B supports the adjusted effect after covariate control."


class MockPlanner:
    def __init__(self, long=False):
        self.calls = 0
        self.requests = []
        self.long = long

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.requests.append((system, user))
        text = ("medical trial adjusted effect research" + " extra" * 80
                if self.long else f"adjusted condition effect clinical round {self.calls}")
        return json.dumps({
            "query": text,
            "target_relation": "adjusted effect", "candidate_answer": "B"})


class MockRetriever:
    def __init__(self, *, no_new=False):
        self.calls = 0
        self.queries = []
        self.no_new = no_new

    def search(self, query, *, k):
        assert len(query) <= 300
        self.calls += 1
        self.queries.append(query)
        if self.no_new:
            return [PASSAGES[0]]
        return [
            Passage(f"new{self.calls}_{i}",
                    FACT + f" This is passage {i} in query {self.calls}.",
                    "retrieved", f"external_trial_{self.calls}_{i}")
            for i in range(2)
        ]


class MockReader:
    def __init__(self, *, invalid_quote=False, reversed_flips=False,
                 confront_choice="B", baseline_malformed=False):
        self.calls = 0
        self.requests = []
        self.invalid_quote = invalid_quote
        self.reversed_flips = reversed_flips
        self.confront_choice = confront_choice
        self.baseline_malformed = baseline_malformed

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.requests.append((system, user))
        if system == SOURCE_SYSTEM:
            assert "KGCC GENERATED HYPOTHESIS" not in user
            ids = re.findall(r"\[(new\d+_\d+)\]", user)
            assert ids
            if self.reversed_flips and ids[0].endswith("_1"):
                chosen = "A"
            else:
                chosen = "B"
            return json.dumps({
                "answer_choice": chosen,
                "evidence": [{
                    "id": "g0" if self.invalid_quote else ids[0],
                    "quote": FACT
                }],
                "reasoning": "Source views are hypotheses"})
        if system == CONFRONT_SYSTEM:
            cited = re.findall(r"\[(new\d+_\d+)\]", user)
            assert cited
            return json.dumps({
                "answer_choice": self.confront_choice,
                "evidence": [{"id": cited[0], "quote": FACT}],
                "reasoning": "conditional effect"})
        if self.baseline_malformed:
            return '{"reasoning": unescaped text, "answer_choice": "A"'
        return '{"answer_choice":"A", "reasoning":"baseline"}'


def engine(*, max_rounds=2, reader=None, long_query=False,
           no_new=False, **kwargs):
    rdr = reader or MockReader()
    plan, retr = MockPlanner(long_query), MockRetriever(no_new=no_new)
    model = EvidenceConfrontedMedRGAG(
        reader=rdr,
        planner=plan if max_rounds else None,
        retriever=retr if max_rounds else None,
        ranker=IdentityRanker(),
        rag_template="Question: {question}\nRetrieved: {documents}",
        config=Config(max_rounds=max_rounds, source_docs=2, **kwargs))
    return model, rdr, plan, retr


def run(model):
    return model.run(Q, PASSAGES, GENERATED,
                     original_selection=(0,5,1,6,2))


def test_b0_original_is_same_cached_KADS_and_no_new_calls():
    obj, rdr, plan, retr = engine(max_rounds=0)
    res = run(obj)
    assert res["base_selected_doc_ids"] == ["r0","g0","r1","g1","r2"]
    assert res["prediction"] == res["base_prediction"] == "A"
    assert "KGCC GENERATED HYPOTHESIS_0" in rdr.requests[0][1]
    assert res["reader_calls"] == 1
    assert res["retrieval_calls"] == res["auxiliary_calls"] == 0
    assert len(res["prefixes"]) == 1


def test_three_paper_mechanisms_plus_order_intervention_are_called():
    obj, rdr, plan, retr = engine(max_rounds=1)
    res = run(obj)
    assert res["base_prediction"] == "A"
    assert res["prediction"] == "B"
    stage = res["novel_views"][0]
    assert stage["decision"] == "cited_confrontation_adopted"
    assert stage["source_prediction"] == "B"
    assert stage["reversed_prediction"] == "B"
    assert stage["confront_prediction"] == "B"
    assert stage["source_citations"][0]["id"] == "new1_0"
    assert stage["confront_citations"][0]["id"] == "new1_0"
    assert res["reader_calls"] == 4  # baseline+source+reverse+adjudication
    assert res["auxiliary_calls"] == 1
    assert res["retrieval_calls"] == 1
    assert stage["fresh_outside_original_top32"] == 2
    assert "KGCC GENERATED HYPOTHESIS" in plan.requests[0][1]
    assert res["prefixes"][1]["prediction"] == "B"
    assert res["quotes_are_entailment_proofs"] is False


def test_invalid_quotes_cannot_override_without_reconciliation():
    obj, rdr, plan, retr = engine(
        max_rounds=1, reader=MockReader(invalid_quote=True))
    row = run(obj)
    assert row["prediction"] == "A"
    assert row["novel_views"][0]["decision"] == "no_valid_exact_source_quote"
    assert row["reader_calls"] == 2
    assert not any(x[0] == CONFRONT_SYSTEM for x in rdr.requests)


def test_order_instability_does_not_pass():
    obj, rdr, plan, retr = engine(
        max_rounds=1, reader=MockReader(reversed_flips=True))
    row = run(obj)
    assert row["prediction"] == "A"
    assert row["novel_views"][0]["decision"] == "order_sensitive_or_uncited"
    assert row["reader_calls"] == 3


def test_confrontation_can_reject_a_wrong_new_evidence_view():
    obj, rdr, plan, retr = engine(
        max_rounds=1, reader=MockReader(confront_choice="A"))
    row = run(obj)
    assert row["prediction"] == "A"
    assert row["novel_views"][0]["decision"] == "confrontation_does_not_support_change"


def test_query_overlength_is_clipped_before_live_retrieval():
    obj, rdr, plan, retr = engine(max_rounds=1, long_query=True)
    row = run(obj)
    assert len(retr.queries) == 1
    assert 5 <= len(retr.queries[0]) <= 300
    assert row["novel_views"][0]["query_clipped"] is True
    assert row["novel_views"][0]["query_rejected_reason"] is None
    assert bounded_query(" z " * 500)[1] is True


def test_parse_recovery_is_logged_independently_from_knowledge():
    obj, rdr, plan, retr = engine(
        max_rounds=0, reader=MockReader(baseline_malformed=True))
    row = run(obj)
    assert row["prediction"] == "A"
    assert row["baseline_parse_recovered"] is True
    assert get_choice('{"reasoning": bad, "answer_choice":"A"', ("A","B")) == ("A", True)
    assert get_choice('answer_choice: A\nanswer_choice: B', ("A","B")) == (None, False)


def test_query_and_exact_source_validation():
    assert verified_quotes('{"evidence":[{"id":"r0","quote":"a long unrelated string"}]}',
                           {"r0": "evidence not present"}) == []
    assert verified_quotes('{"evidence":[{"id":"r0","quote":"Sufficient textual proof"}]}',
                           {"r0": "Text: Sufficient textual proof"}) == [
                               {"id":"r0", "quote":"Sufficient textual proof"}]
    with pytest.raises(ValueError):
        Config(source_docs=6)
    with pytest.raises(ValueError):
        Config(max_query_chars=301)


def test_no_new_sources_means_no_more_reader_calls():
    obj, rdr, plan, retr = engine(max_rounds=2, no_new=True)
    row = run(obj)
    assert row["prediction"] == "A"
    assert row["reader_calls"] == 1
    assert row["retrieval_calls"] == 2
    assert all(v["fresh_source_count"] == 0 for v in row["novel_views"])


def test_question_only_and_latest_are_explicit_ablation():
    obj, rdr, plan, retr = engine(
        max_rounds=1, planner_mode="question",
        policy="latest", order_check=False)
    row = run(obj)
    assert row["prediction"] == "B"
    assert row["reader_calls"] == 2
    assert "KGCC GENERATED HYPOTHESIS" not in plan.requests[0][1]
    assert row["novel_views"][0]["decision"] == "latest_unverified_ablation"
