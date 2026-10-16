"""Scientific unit tests for MedRGAG A+B+C+D composition; no GPU/API/index.

Do not mistake dummy predictions for real medical benchmark scores.
"""
import json
import pytest

from coverage_medrgag.abcd import (
    Config, MedRGAGABCD, combine_query, compress_source,
    fit_query, render_notes, validate_option_notes,
)
from coverage_medrgag.data import Passage, Question
from coverage_medrgag.models import IdentityRanker


class Reader:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.prompts.append(user)
        return '{"reasoning":"Need adjusted results from independent source.", "answer_choice":"A"}'


class Auxiliary:
    def __init__(self, *, overflow=False):
        self.calls = 0
        self.prompts = []
        self.overflow = overflow

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.prompts.append((system, user))
        if "one NEW English biomedical BM25" in system:
            query = ("independent adjusted prognostic effect " + "risk " * 150
                     if self.overflow else
                     f"adjusted multivariate survival association round {self.calls}")
            return json.dumps({
                "query": query, "target_fact": "adjusted risk independent"})
        if "Make separate source-grounded" in system:
            # A valid exact source quote, plus three deliberately invalid
            # attempts at hallucinated source evidence.
            return json.dumps({"choices": [
                {"choice": "A", "quotes": [
                    {"id":"new1_0", "quote":"After adjustment, effect persisted.",
                     "relation":"supports"},
                    {"id":"g0", "quote":"Generated KGCC story is proof.",
                     "relation":"supports"},
                ]},
                {"choice": "B", "quotes": [
                    {"id":"new1_0", "quote":"Invented effect conclusion.",
                     "relation":"supports"},
                    {"id":"new1_0", "quote":"After adjustment, effect persisted.",
                     "relation":"yes"},
                ]}
            ]})
        raise AssertionError("Unexpected auxiliary call")


class Retriever:
    def __init__(self, *, no_new=False):
        self.calls = 0
        self.queries = []
        self.no_new = no_new

    def search(self, query, *, k):
        self.calls += 1
        self.queries.append(query)
        if self.no_new:
            return [Passage("old", "Original document zero", "retrieved", "p0")]
        return [
            Passage(f"new{self.calls}_0",
                    "After adjustment, effect persisted. Another background.",
                    "retrieved", f"outside{self.calls}"),
            Passage(f"new{self.calls}_1",
                    "No change occurred among the comparison population.",
                    "retrieved", f"another{self.calls}"),
        ]


def synthetic():
    q = Question("x", "Does depth predict survival after adjustment?",
                 {"A":"Yes", "B":"No"})
    retrieval = [
        Passage(f"r{i}", f"Original document {i}", "retrieved", f"p{i}")
        for i in range(32)]
    generated = [
        Passage(f"g{i}",
                f"Generated KGCC hypothesis {i} about statistical survival adjustment",
                "parametric_generated", f"gen{i}")
        for i in range(5)]
    return q, retrieval, generated


def build(config):
    r, a, search = Reader(), Auxiliary(), Retriever()
    return MedRGAGABCD(
        reader=r, auxiliary=a if config.use_ircot or config.use_sure else None,
        retriever=search if config.max_rounds else None,
        ranker=IdentityRanker(),
        rag_template="Question: {question}\nRetrieved: {documents}",
        config=config), r, a, search


def test_query_limit_enforced_and_query2doc_is_not_a_truth_source():
    query, clipped = combine_query("research question", "pseudodoc " * 100, 100)
    assert len(query) <= 100
    assert clipped
    assert query.startswith("research question")
    assert len(fit_query("very long phrase " * 100, 300)[0]) <= 300
    with pytest.raises(ValueError, match="300"):
        Config(max_query_chars=301)


def test_extractive_compression_and_choice_citations_reject_hallucinations():
    q, _, _ = synthetic()
    d = Passage("new1_0",
                "Unrelated opening. After adjustment, effect persisted. "
                "The comparator remained without effect.",
                "retrieved", "source000")
    short = compress_source(q, d, max_sentences=2, max_chars=200)
    assert len(short.text) < len(d.text)
    assert short.origin == "retrieved"
    assert short.original_id == d.original_id
    with pytest.raises(ValueError):
        compress_source(q, Passage("g0", "Hallucination", "parametric_generated"),
                        max_sentences=2, max_chars=200)
    notes = validate_option_notes(json.dumps({"choices":[
        {"choice":"A","quotes":[
            {"id":"new1_0","quote":"After adjustment, effect persisted.",
             "relation":"supports"},
            {"id":"g0","quote":"Generated KGCC story is proof.",
             "relation":"supports"}]},
        {"choice":"B","quotes":[
            {"id":"new1_0","quote":"Invented effect conclusion.",
             "relation":"supports"}]},
    ]}), q, [d])
    assert len(notes["quotes"]) == 1
    assert notes["rejected"] == 2
    assert "After adjustment" in render_notes(notes, max_chars=1400)


def test_budget_zero_matches_original_kads_medcpt_order_and_no_extra_calls():
    q, retrieved, generated = synthetic()
    config = Config(max_rounds=0)
    engine, r, a, search = build(config)
    value = engine.run(q, retrieved, generated,
                       original_selection=(0,5,1,6,2))
    assert value["base_selected_doc_ids"] == ["r0","g0","r1","g1","r2"]
    assert value["prediction"] == "A"
    assert len(value["prefixes"]) == 1
    assert value["reader_calls"] == 1
    assert value["auxiliary_calls"] == 0
    assert value["retrieval_calls"] == 0
    assert "Generated KGCC hypothesis 0" in r.prompts[0]
    assert "# OPTION-CONDITIONED" not in r.prompts[0]


def test_all_four_components_and_budget_curve_and_quote_audit():
    q, retrieved, generated = synthetic()
    engine, r, a, search = build(Config(max_rounds=2))
    value = engine.run(q, retrieved, generated,
                       original_selection=(0,5,1,6,2))
    assert len(value["prefixes"]) == 3
    assert value["reader_calls"] == 3
    assert value["retrieval_calls"] == 2
    assert value["auxiliary_calls"] == 4
    assert all(len(x) <= 300 for x in search.queries)
    assert value["novel_views"][0]["fresh_source_count"] == 2
    assert value["novel_views"][0]["source_quote_count"] == 1
    assert value["novel_views"][0]["rejected_quotes"] == 3
    assert "Generated KGCC story is proof" not in r.prompts[1]
    assert "After adjustment, effect persisted." in r.prompts[1]
    assert value["novel_views"][0]["compression_output_chars"] <= (
        value["novel_views"][0]["compression_original_chars"])
    assert value["prefixes"][-1]["reader_calls_total"] == 3
    assert value["gold_used_at_inference"] is False


def test_long_planner_query_is_capped_and_still_retrieved():
    q, ret, gen = synthetic()
    cfg = Config(max_rounds=1, use_sure=False)
    reader, aux, search = Reader(), Auxiliary(overflow=True), Retriever()
    engine = MedRGAGABCD(
        reader=reader, auxiliary=aux, retriever=search,
        ranker=IdentityRanker(),
        rag_template="Question: {question}\nRetrieved: {documents}",
        config=cfg)
    value = engine.run(q, ret, gen, original_selection=(0,5,1,6,2))
    assert value["novel_views"][0]["query_was_truncated"] is True
    assert len(search.queries[0]) <= 300
    assert value["retrieval_calls"] == 1


@pytest.mark.parametrize("disabled", [
    "use_query2doc", "use_ircot", "use_recomp", "use_sure"])
def test_each_component_can_be_ablated_separately(disabled):
    q, ret, gen = synthetic()
    flags = {"max_rounds": 1, disabled: False}
    config = Config(**flags)
    engine, reader, aux, retrieval = build(config)
    result = engine.run(q, ret, gen,
                        original_selection=(0,5,1,6,2))
    assert result["components"][
        {"use_query2doc":"A_query2doc",
         "use_ircot":"B_ircot",
         "use_recomp":"C_recomp_inspired",
         "use_sure":"D_sure_inspired"}[disabled]] is False
    assert result["retrieval_calls"] == 1
    assert result["reader_calls"] == 2
    assert result["base_prediction"] == "A"


def test_no_source_does_not_invent_reader_answer_or_sources():
    q, ret, gen = synthetic()
    reader, aux, search = Reader(), Auxiliary(), Retriever(no_new=True)
    engine = MedRGAGABCD(
        reader=reader, auxiliary=aux, retriever=search,
        ranker=IdentityRanker(),
        rag_template="Question: {question}\nRetrieved: {documents}",
        config=Config(max_rounds=1))
    result = engine.run(q, ret, gen,
                        original_selection=(0,5,1,6,2))
    assert result["reader_calls"] == 1
    assert result["retrieval_calls"] == 1
    assert result["prediction"] == result["base_prediction"]
    assert result["novel_views"][0]["fresh_source_count"] == 0
