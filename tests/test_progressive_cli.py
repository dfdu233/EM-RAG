"""End-to-end runner/CLI and offline evaluation using MOCK API and BM25.

No actual model accuracy is tested here. All evidence and answers synthetic.
"""
import json
import pytest

from coverage_medrgag import progressive_cli
from coverage_medrgag.data import Passage


def fake_orig_medrgag(tmp_path):
    root = tmp_path / "MedRGAG"
    (root / "src").mkdir(parents=True)
    base = root / "outputs" / "pubmedqa" / "docs"
    (base / "retrieval").mkdir(parents=True)
    (base / "generate").mkdir(parents=True)
    (root / "benchmark.json").write_text(
        json.dumps({"pubmedqa": {
            "q1": {"question": "Synthetic question 1",
                   "options": {"A": "yes", "B": "no", "C": "maybe"},
                   "answer": "A"},
            "q2": {"question": "Synthetic question 2",
                   "options": {"A": "yes", "B": "no", "C": "maybe"},
                   "answer": "B"},
        }}), encoding="utf-8")
    (root / "src" / "template.py").write_text(
        'rag_template = "Question: {question}\\nDocuments: {documents}"\n'
        'llm_selection_prompt = "Choose {question} {documents}"\n',
        encoding="utf-8")
    retrieved, generated, selected = {}, {}, {}
    for qid in ("q1", "q2"):
        retrieved[qid] = {
            "snippets": [
                {"id": f"{qid}_old_{i}",
                 "contents": f"OLD_{qid}_passage_{i} regular research"}
                for i in range(32)]}
        generated[qid] = {
            "snippets": [
                {"id": f"{qid}_kgcc_{i}",
                 "contents": f"KGCC_{qid}_hypothesis_{i}"}
                for i in range(5)]}
        selected[qid] = {"final_list": [0, 5, 1, 6, 2]}
    (base / "retrieval" / "snippets.json").write_text(
        json.dumps(retrieved), encoding="utf-8")
    (base / "generate" / "snippets.json").write_text(
        json.dumps(generated), encoding="utf-8")
    (base / "generate" / "snippets_final_select.json").write_text(
        json.dumps(selected), encoding="utf-8")
    return root


class MockOpenAIChat:
    def __init__(self, model, **kwargs):
        self.calls = 0
        self.model = model

    def complete(self, system, user, *, temperature, seed, max_tokens):
        self.calls += 1
        if self.model == "planner":
            return json.dumps({
                "query": f"unseen-medical-relationship-round-{self.calls}",
                "target_relation": "a specific comparison"})
        return ('{"answer_choice":"B"}' if "FRESH_CORPUS_EXCERPT" in user
                else '{"answer_choice":"A"}')


class MockNativeBM25:
    def __init__(self, **kwargs):
        self.calls = 0

    def search(self, query, *, k):
        self.calls += 1
        return [
            Passage(f"new{self.calls}_{i}",
                    f"FRESH_CORPUS_EXCERPT_{self.calls}_{i}",
                    "retrieved", f"article{self.calls}_{i}")
            for i in range(2)
        ]


def common_args(root, out):
    return [
        "run", "--medrgag-root", str(root), "--dataset", "pubmedqa",
        "--reader-model", "reader", "--planner-model", "planner",
        "--no-rerank", "--max-rounds", "2", "--limit", "2",
        "--max-new-examples", "5", "--output", str(out),
    ]


def test_runner_is_resumable_and_has_real_budget_curves(tmp_path, monkeypatch):
    root, out = fake_orig_medrgag(tmp_path), tmp_path / "experimental"
    monkeypatch.setattr(progressive_cli, "OpenAIChat", MockOpenAIChat)
    monkeypatch.setattr(progressive_cli, "NativeMedRGAGBM25", MockNativeBM25)
    args = common_args(root, out)
    assert progressive_cli.main(args) == 0
    rows = [
        json.loads(x) for x in
        (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["base_prediction"] == "A" for r in rows)
    assert all(r["prediction"] == "B" for r in rows)
    assert all(r["reader_calls"] == 3 for r in rows)
    assert all(r["auxiliary_calls"] == 2 for r in rows)
    assert all(r["retrieval_calls"] == 2 for r in rows)
    assert all([p["prediction"] for p in row["prefixes"]] ==
               ["A", "A", "B"] for row in rows)
    assert progressive_cli.main(args) == 0
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2

    report = tmp_path / "report.json"
    assert progressive_cli.main([
        "evaluate", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(out / "predictions.jsonl"),
        "--bootstrap", "100", "--output", str(report)
    ]) == 0
    metrics = json.loads(report.read_text(encoding="utf-8"))
    assert metrics["by_budget"]["0"]["accuracy"] == .5
    assert metrics["by_budget"]["1"]["accuracy"] == .5
    assert metrics["by_budget"]["2"]["accuracy"] == .5
    assert metrics["transitions"]["B0_to_B2"]["rescue"] == 1
    assert metrics["transitions"]["B0_to_B2"]["harm"] == 1
    assert metrics["by_budget"]["2"]["mean_real_retrieval_calls"] == 2
    assert metrics["by_budget"]["2"]["mean_planner_calls"] == 2
    assert metrics["transitions"]["B0_to_B2"]["n_selected_outside_original_top32"] == 2*2*2
    with pytest.raises(ValueError, match="Refusing to mix"):
        progressive_cli.main(args + ["--max-rounds", "3"])


def test_zero_rounds_uses_no_extra_models_or_corpus(tmp_path, monkeypatch):
    root, out = fake_orig_medrgag(tmp_path), tmp_path / "reference"
    monkeypatch.setattr(progressive_cli, "OpenAIChat", MockOpenAIChat)
    args = common_args(root, out)
    # Changing max-rounds to 0 means the mock BM25 retriever is not used.
    assert progressive_cli.main(args + ["--max-rounds", "0"]) == 0
    rows = [json.loads(x) for x in (out / "predictions.jsonl").read_text().splitlines()]
    assert all(x["prediction"] == "A" and x["reader_calls"] == 1
               and x["retrieval_calls"] == 0 and x["auxiliary_calls"] == 0
               and len(x["prefixes"]) == 1 for x in rows)


def test_real_retriever_never_auto_builds_or_clones(tmp_path):
    from coverage_medrgag.live_retrieval import NativeMedRGAGBM25
    with pytest.raises(FileNotFoundError, match="Refusing automatic"):
        NativeMedRGAGBM25(medrgag_root=tmp_path, corpus_name="MedText")


def test_inference_loader_never_interprets_gold_answers(tmp_path):
    path = tmp_path / "invalid_gold.json"
    path.write_text(json.dumps({"pubmedqa": {"x": {
        "question": "What is known?", "options": {"A": "Yes", "B": "No"},
        "answer": "THIS_IS_NOT_A_VALID_ANSWER_AND_MUST_NOT_BE_ACCESSED"
    }}}), encoding="utf-8")
    result = progressive_cli._questions_without_gold(path, "pubmedqa")
    assert result["x"].text == "What is known?"
    assert result["x"].labels == ("A", "B")
