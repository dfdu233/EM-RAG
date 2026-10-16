"""End-to-end training-free cached MedRGAG runner without real API/GPU."""
import json
import pytest

from coverage_medrgag import anytime_cli


def mock_medrgag(tmp_path):
    root = tmp_path / "MedRGAG"
    (root / "src").mkdir(parents=True)
    base = root / "outputs" / "pubmedqa" / "docs"
    (base / "retrieval").mkdir(parents=True)
    (base / "generate").mkdir(parents=True)
    benchmark = {
        "pubmedqa": {
            "1": {"question":"Synthetic trial after adjustment?",
                  "options":{"A":"yes","B":"no","C":"maybe"},"answer":"A"},
            "2": {"question":"Different artificial study?",
                  "options":{"A":"yes","B":"no","C":"maybe"},"answer":"B"},
        }
    }
    (root / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    (root / "src" / "template.py").write_text(
        'rag_template = "Question: {question}\\nRetrieved: {documents}"\n'
        'llm_selection_prompt = "Select five {question} {documents}"\n',
        encoding="utf-8")
    sources = {}
    generated = {}
    selected = {}
    for qid in ("1", "2"):
        sources[qid] = {
            "snippets": [
                {"id":f"pub_{qid}_{i}",
                 "contents":f"Synthetic document {i} after trial adjustment comparison."}
                for i in range(32)]
        }
        generated[qid] = {
            "snippets": [
                {"id":f"kgcc_{qid}_{i}", "contents":f"Background {i} after trial."}
                for i in range(5)]
        }
        selected[qid] = {"final_list":[0,5,1,6,2]}
    (base / "retrieval" / "snippets.json").write_text(
        json.dumps(sources), encoding="utf-8")
    (base / "generate" / "snippets.json").write_text(
        json.dumps(generated), encoding="utf-8")
    (base / "generate" / "snippets_final_select.json").write_text(
        json.dumps(selected), encoding="utf-8")
    return root


class FakeModel:
    def __init__(self, model, **kwargs):
        self.calls = 0
        self.model = model
        self.seen = []

    def complete(self, system, user, *, temperature=0, seed=None,
                 max_tokens=512):
        self.calls += 1
        self.seen.append(user)
        return '{"answer_choice":"A"}'


def test_live_interface_with_cached_medrgag_inputs(tmp_path, monkeypatch):
    root = mock_medrgag(tmp_path)
    monkeypatch.setattr(anytime_cli, "OpenAIChat", FakeModel)
    out = tmp_path / "run"
    argv = [
        "run", "--medrgag-root", str(root),
        "--dataset", "pubmedqa",
        "--reader-model", "mock-reader",
        "--utility", "lexical", "--no-rerank",
        "--depths", "5", "10", "20", "32",
        "--limit", "2", "--output", str(out),
    ]
    assert anytime_cli.main(argv) == 0
    rows = [json.loads(x) for x in
            (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["reader_calls"] == 1 for r in rows)
    assert all(r["auxiliary_calls"] == 0 for r in rows)
    assert all(r["scorer"] == "lexical" for r in rows)
    assert all(len(r["depth_path"]) == 4 for r in rows)
    assert all(r["depth_path"][0]["best_doc_ids"] ==
               ["r0","g0","r1","g1","r2"] for r in rows)
    assert anytime_cli.main(argv) == 0
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 2
    report = tmp_path / "report.json"
    assert anytime_cli.main([
        "evaluate", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(out / "predictions.jsonl"),
        "--bootstrap", "20", "--output", str(report)
    ]) == 0
    stats = json.loads(report.read_text())
    assert stats["summary"]["accuracy"] == .5
    assert stats["summary"]["preprocessing_KGCC_cost_not_included"] is True


def test_prefix_curve_and_provenance_guard(tmp_path, monkeypatch):
    root = mock_medrgag(tmp_path)
    monkeypatch.setattr(anytime_cli, "OpenAIChat", FakeModel)
    out = tmp_path / "prefix"
    args = [
        "run", "--medrgag-root", str(root),
        "--dataset", "pubmedqa",
        "--reader-model", "mock-reader",
        "--utility", "lexical", "--no-rerank",
        "--depths", "5", "10", "--read-each-depth",
        "--limit", "2", "--output", str(out),
    ]
    assert anytime_cli.main(args) == 0
    rows = [json.loads(x) for x in
            (out / "predictions.jsonl").read_text().splitlines()]
    assert all(r["reader_calls"] == 2 for r in rows)
    summary = tmp_path / "curve.json"
    assert anytime_cli.main([
        "evaluate", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(out / "predictions.jsonl"),
        "--bootstrap", "20", "--include-prefix-curve",
        "--output", str(summary)
    ]) == 0
    curve = json.loads(summary.read_text())["prefix_curve"]
    assert curve["by_depth"]["5"]["n"] == 2
    assert curve["by_depth"]["10"]["n"] == 2
    assert "5_to_10" in curve["paired_transitions"]

    # Changing a budget after starting must FAIL instead of merging outputs.
    with pytest.raises(ValueError, match="Refusing to mix"):
        anytime_cli.main(args + ["--max-facets", "6"])


def test_medcpt_requires_reranker(tmp_path):
    root = mock_medrgag(tmp_path)
    with pytest.raises(ValueError, match="MedCPT utility requires"):
        anytime_cli.main([
            "run", "--medrgag-root", str(root),
            "--dataset", "pubmedqa", "--reader-model", "fixture",
            "--no-rerank", "--limit", "1", "--output", str(tmp_path / "bad")
        ])
