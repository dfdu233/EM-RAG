"""End-to-end cached MedRGAG -> new BM25 -> source-only -> cited adjudication.

ALL services are mocked; this checks protocol and resumability, not accuracy.
"""
import json
import re

import pytest

from coverage_medrgag import confront_cli
from coverage_medrgag.confront import SOURCE_SYSTEM, CONFRONT_SYSTEM
from coverage_medrgag.data import Passage
from tests.test_progressive_cli import fake_orig_medrgag


TEXT = "A sufficient-length fake statement about adjusted clinical outcomes."


class MockChat:
    def __init__(self, model, **kwargs):
        self.model = model
        self.calls = 0
        self.messages = []

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        self.messages.append((system, user))
        if self.model == "planner":
            return json.dumps({
                "query": "trial adjusted endpoint effect conditions " + str(self.calls),
                "target_relation": "controlled effect"})
        if system == SOURCE_SYSTEM:
            id_ = re.search(r"\[(new\d+_\d+)\]", user).group(1)
            return json.dumps({
                "answer_choice": "B",
                "evidence": [{"id": id_, "quote": TEXT}]})
        if system == CONFRONT_SYSTEM:
            id_ = re.search(r"\[(new\d+_\d+)\]", user).group(1)
            return json.dumps({
                "answer_choice": "B",
                "evidence": [{"id": id_, "quote": TEXT}]})
        return '{"answer_choice":"A"}'


class MockBM25:
    def __init__(self, **kwargs):
        self.calls = 0

    def search(self, query, *, k):
        assert len(query) <= 300
        self.calls += 1
        return [
            Passage(f"new{self.calls}_{i}",
                    TEXT + f" Unique source {self.calls}:{i}.",
                    "retrieved", f"external{self.calls}{i}") for i in range(2)
        ]


def argv(root, out):
    return [
        "run", "--medrgag-root", str(root), "--dataset", "pubmedqa",
        "--reader-model", "reader", "--planner-model", "planner",
        "--no-rerank", "--max-rounds", "1", "--source-docs", "2",
        "--limit", "2", "--max-new-examples", "2", "--output", str(out)
    ]


def test_end_to_end_and_safe_resume(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    out = tmp_path / "fusion"
    monkeypatch.setattr(confront_cli, "OpenAIChat", MockChat)
    monkeypatch.setattr(confront_cli, "NativeMedRGAGBM25", MockBM25)
    cmd = argv(root, out)
    assert confront_cli.main(cmd) == 0
    rows = [json.loads(x) for x in
            (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(x["base_prediction"] == "A" for x in rows)
    assert all(x["prediction"] == "B" for x in rows)
    assert all(x["reader_calls"] == 4 for x in rows)
    assert all(x["retrieval_calls"] == 1 for x in rows)
    assert all(x["novel_views"][0]["decision"] ==
               "cited_confrontation_adopted" for x in rows)
    assert all(x["gold_used_at_inference"] is False for x in rows)

    assert confront_cli.main(cmd) == 0
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="Refusing to mix"):
        confront_cli.main(cmd + ["--no-order-check"])

    report = tmp_path / "eval.json"
    assert confront_cli.main([
        "evaluate", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(out / "predictions.jsonl"),
        "--bootstrap", "40", "--output", str(report)
    ]) == 0
    data = json.loads(report.read_text())
    assert data["n"] == 2
    assert data["by_budget"]["0"]["accuracy"] == .5
    assert data["by_budget"]["1"]["accuracy"] == .5
    assert data["transitions"]["B0_to_B1"]["rescue"] == 1
    assert data["transitions"]["B0_to_B1"]["harm"] == 1
    extra = data["confrontation_diagnostics"]
    assert extra["order_checked_count"] == 2
    assert extra["final_answer_changed"] == 2
    assert extra["source_quoted_verbatim_count"] == 2
    assert extra["comparison_on_parseable_baseline_only"]["paired_n"] == 2


def test_exact_original_b0_without_extra_services(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    out = tmp_path / "b0"
    monkeypatch.setattr(confront_cli, "OpenAIChat", MockChat)
    cmd = argv(root, out) + ["--max-rounds", "0"]
    assert confront_cli.main(cmd) == 0
    rows = [json.loads(x) for x in
            (out / "predictions.jsonl").read_text().splitlines()]
    assert all(x["prediction"] == "A" for x in rows)
    assert all(x["reader_calls"] == 1 and x["auxiliary_calls"] == 0
               and x["retrieval_calls"] == 0 for x in rows)


def test_gold_never_loaded_for_inference(tmp_path):
    source = tmp_path / "arbitrary_gold.json"
    source.write_text(json.dumps({"pubmedqa": {"x": {
        "question":"Ask something?", "options":{"A":"Yes","B":"No"},
        "answer":"INVALID_DO_NOT_USE"}}}), encoding="utf-8")
    info = confront_cli._questions_without_gold(source, "pubmedqa")
    assert info["x"].labels == ("A", "B")
