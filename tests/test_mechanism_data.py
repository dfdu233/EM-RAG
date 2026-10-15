import json
import pytest
from rag_mechanisms.data import (
    load_benchmark, load_ranked_cache, load_generated_cache, manifest, read_manifest,
    hash_file, Passage, Question, write_jsonl,
)
from rag_mechanisms.design import build_conditions, census


def sample(tmp_path):
    bench = tmp_path / "benchmark.json"
    bench.write_text(json.dumps({"medqa": {
        str(i): {"id": str(i), "question": "What is fictional Y?",
                 "options": {"A": "alpha", "B": "beta"},
                 "answer": "A" if i % 2 else "B"}
        for i in range(10)}}), encoding="utf-8")
    cache = tmp_path / "snippets.json"
    cache.write_text(json.dumps({
        str(i): {
            "snippets": [
                {"id": f"{i}-{j}", "contents": f"Reference {i}:{j} "
                 + ("x" * 90), "title": "Textbook"}
                for j in range(10)],
            "scores": [float(j) for j in range(10)]
        } for i in range(10)}), encoding="utf-8")
    return bench, cache


def test_gold_never_in_question_or_sample_selection(tmp_path):
    bench, cache_path = sample(tmp_path)
    questions, gold = load_benchmark(bench, "medqa")
    assert not hasattr(questions["1"], "answer")
    assert gold["1"] == "A"
    assert questions["1"].medrgag_query().endswith("A. alpha\nB. beta")
    candidates = load_ranked_cache(cache_path)
    sample1 = manifest(questions, dataset="medqa", seed=123, count=5,
                       eligible_ids=set(candidates))
    sample2 = manifest(questions, dataset="medqa", seed=123, count=5,
                       eligible_ids=set(candidates))
    assert sample1 == sample2
    mf = tmp_path / "manifest.jsonl"
    write_jsonl(mf, sample1)
    assert read_manifest(mf, "medqa") == [x["id"] for x in sample1]
    assert len(hash_file(bench)) == 64


def test_depth_conditions_and_negative_control_have_equal_text_lengths(tmp_path):
    bench, cache_path = sample(tmp_path)
    retrieval = load_ranked_cache(cache_path)
    gen = [Passage(id=f"g{k}", text=f"Generated {k}", source="MODEL")
           for k in range(5)]
    interventions = build_conditions(
        "0", retrieval["0"], gen, retrieval,
        depths=(0, 1, 3, 5, 10),
        include_generation=True, include_negative_control=True, seed=43)
    lookup = {c.name: c for c in interventions}
    assert len(lookup["retrieved_k0"].passages) == 0
    assert len(lookup["retrieved_k5"].passages) == 5
    assert len(lookup["generated_k5"].passages) == 5
    assert len(lookup["retrieved3_generated2"].passages) == 5
    assert sum(len(d.text) for d in lookup["prefix3_own_tail2"].passages) == (
        sum(len(d.text) for d in
            lookup["prefix3_other_question_tail2"].passages))
    assert all(x.source == "OTHER_QUESTION_CACHE_CONTROL"
               for x in lookup["prefix3_other_question_tail2"].passages[3:])
    assert census({"0": Question("0", "Q?", {"A": "a"})}, retrieval,
                  ["0"], (0, 5))["depth_census"]["5"]["available_questions"] == 1


def test_generated_cache_keeps_generated_only(tmp_path):
    p = tmp_path / "generated.json"
    p.write_text(json.dumps({"1": {"snippets": [
        {"id": "Llama", "contents": "A generated background."},
        {"id": "empty", "contents": ""}
    ]}}), encoding="utf-8")
    assert len(load_generated_cache(p)["1"]) == 1
    assert load_generated_cache(p)["1"][0].source == "PARAMETRIC_GENERATED"


def test_manifest_error(tmp_path):
    p = tmp_path / "manifest.jsonl"
    write_jsonl(p, [{"dataset": "medqa", "id": "1"},
                    {"dataset": "medqa", "id": "1"}])
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(p, "medqa")
