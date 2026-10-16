"""End-to-end ABCD command: synthetic MedRGAG cache, no actual API/BM25/GPU."""
import json
import pytest
from coverage_medrgag import abcd_cli
from tests.test_progressive_cli import fake_orig_medrgag, MockNativeBM25


class FakeChat:
    def __init__(self, model, **kwargs):
        self.model, self.calls = model, 0

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        if self.model == "aux":
            if "Make separate source-grounded" in system:
                return json.dumps({"choices":[
                    {"choice":"A","quotes":[]},
                    {"choice":"B","quotes":[]},
                    {"choice":"C","quotes":[]}]})
            return json.dumps({
                "query":f"independent adjusted association round {self.calls}",
                "target_fact":"adjustment"})
        return '{"answer_choice":"B"}' if "FRESH_CORPUS_EXCERPT" in user else '{"answer_choice":"A"}'


def args(root, out):
    return [
        "run", "--medrgag-root", str(root), "--dataset", "pubmedqa",
        "--reader-model", "reader", "--planner-model", "aux",
        "--no-rerank", "--max-rounds", "2",
        "--limit", "2", "--max-new-examples", "4",
        "--output", str(out),
    ]


def test_abcd_cli_rounds_resume_evaluation_and_component_flags(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    out = tmp_path / "ABCD"
    monkeypatch.setattr(abcd_cli, "OpenAIChat", FakeChat)
    monkeypatch.setattr(abcd_cli, "NativeMedRGAGBM25", MockNativeBM25)
    argv = args(root, out)
    assert abcd_cli.main(argv) == 0
    traces = [json.loads(x) for x in
              (out / "predictions.jsonl").read_text().splitlines()]
    assert len(traces) == 2
    assert all(len(x["prefixes"]) == 3 for x in traces)
    assert all(x["base_prediction"] == "A" for x in traces)
    assert all(x["reader_calls"] == 3 for x in traces)
    assert all(x["retrieval_calls"] == 2 for x in traces)
    assert all(x["auxiliary_calls"] == 4 for x in traces)
    assert all(x["baseline_prompt_sha256"] for x in traces)
    assert all(x["components"]["A_query2doc"] for x in traces)
    assert abcd_cli.main(argv) == 0
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2
    report = tmp_path / "report.json"
    assert abcd_cli.main([
        "evaluate", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(out / "predictions.jsonl"),
        "--bootstrap", "50", "--output", str(report)
    ]) == 0
    summary = json.loads(report.read_text())
    assert summary["by_budget"]["0"]["accuracy"] == .5
    assert summary["by_budget"]["2"]["accuracy"] == .5
    assert summary["transitions"]["B0_to_B2"]["rescue"] == 1
    assert summary["transitions"]["B0_to_B2"]["harm"] == 1
    with pytest.raises(ValueError, match="Refusing to mix"):
        abcd_cli.main(argv + ["--no-recomp"])


def test_abcd_cli_b0_parity_does_not_load_index_or_planner(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    monkeypatch.setattr(abcd_cli, "OpenAIChat", FakeChat)
    argv = args(root, tmp_path / "B0") + ["--max-rounds","0"]
    assert abcd_cli.main(argv) == 0
    rows = [json.loads(x) for x in
            (tmp_path / "B0" / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["reader_calls"] == 1 and row["auxiliary_calls"] == 0
               and row["retrieval_calls"] == 0 for row in rows)


def test_abcd_cli_can_disable_all_four_without_auxiliary(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    monkeypatch.setattr(abcd_cli, "OpenAIChat", FakeChat)
    monkeypatch.setattr(abcd_cli, "NativeMedRGAGBM25", MockNativeBM25)
    argv = args(root, tmp_path / "bare") + [
        "--no-query2doc", "--no-ircot", "--no-recomp", "--no-sure"]
    assert abcd_cli.main(argv) == 0
    rows = [json.loads(x) for x in
            (tmp_path / "bare" / "predictions.jsonl").read_text().splitlines()]
    assert all(row["auxiliary_calls"] == 0 for row in rows)
    assert all(row["components"]["D_sure_inspired"] is False for row in rows)
    assert all(row["components"]["B_ircot"] is False for row in rows)
