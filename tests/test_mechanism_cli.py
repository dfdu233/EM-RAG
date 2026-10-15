"""Offline end-to-end smoke test: real MedRGAG-format files, mocked Reader."""
import json
from pathlib import Path

from rag_mechanisms import cli
from rag_mechanisms.data import read_jsonl


def test_plan_run_resume_report_and_audit_without_model_or_gold_leak(tmp_path, monkeypatch):
    root = tmp_path / "MedRGAG"
    (root / "src").mkdir(parents=True)
    (root / "src" / "template.py").write_text(
        "rag_template = 'QUESTION: {question}\\nDOCUMENTS: {documents}'\n",
        encoding="utf-8")
    benchmark = root / "benchmark.json"
    benchmark.write_text(json.dumps({"pubmedqa": {
        "101": {"question": "Synthetic study: what is X?",
                "options": {"A": "Alpha", "B": "Beta"}, "answer": "B"},
    }}), encoding="utf-8")
    retrieval = root / "snippets.json"
    retrieval.write_text(json.dumps({"101": {
        "snippets": [
            {"id": str(i), "contents": f"Fictional document {i}"}
            for i in range(8)],
        "scores": [0.1] * 8,
    }}), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    cli.main([
        "plan", "--benchmark", str(benchmark), "--dataset", "pubmedqa",
        "--retrieval-cache", str(retrieval), "--count", "1",
        "--output", str(manifest),
    ])
    assert len(read_jsonl(manifest)) == 1

    prompts = []

    class OfflineReader:
        def __init__(self, model, **kwargs):
            self.model = model
            self.temperature = kwargs["temperature"]
            self.max_output_tokens = kwargs["max_output_tokens"]
            self.seed = kwargs["seed"]
            self.calls = 0

        def answer(self, prompt, choices):
            self.calls += 1
            prompts.append(prompt.text)
            return {
                "prediction": "A", "raw_output": '{"answer_choice":"A"}',
                "prompt_tokens_billed": 123,
                "completion_tokens_billed": 8,
            }

    monkeypatch.setattr(cli, "FrozenReader", OfflineReader)

    out = tmp_path / "run"
    argv = [
        "run", "--benchmark", str(benchmark), "--dataset", "pubmedqa",
        "--retrieval-cache", str(retrieval),
        "--template", str(root / "src" / "template.py"),
        "--manifest", str(manifest), "--output", str(out),
        "--model", "mock-reader", "--depths", "0", "1", "3", "5",
        "--max-examples", "1",
    ]
    cli.main(argv + ["--max-calls", "2"])
    first = read_jsonl(out / "predictions.jsonl")
    assert len(first) == 2
    cli.main(argv + ["--max-calls", "10"])
    results = read_jsonl(out / "predictions.jsonl")
    assert len(results) == 4
    assert len({r["condition"] for r in results}) == 4
    assert all(r["prediction"] == "A" for r in results)
    assert len(prompts) == 4
    assert all("Synthetic study" in p for p in prompts)
    assert all('"answer": "B"' not in p for p in prompts)

    rep = tmp_path / "report.json"
    cli.main([
        "report", "--benchmark", str(benchmark), "--dataset", "pubmedqa",
        "--predictions", str(out / "predictions.jsonl"),
        "--bootstrap-samples", "50", "--output", str(rep),
    ])
    report = json.loads(rep.read_text())
    assert report["summary"]["conditions"]["retrieved_k0"]["accuracy"] == 0.0
    assert report["summary"]["no_manual_evidence_sufficiency_claim"] is True

    audit = tmp_path / "audit"
    cli.main([
        "audit", "--benchmark", str(benchmark), "--dataset", "pubmedqa",
        "--predictions", str(out / "predictions.jsonl"),
        "--retrieval-cache", str(retrieval), "--output", str(audit),
        "--per-category", "1",
    ])
    assert (audit / "casebook.md").exists()
    labels = read_jsonl(audit / "annotations_template.jsonl")
    assert labels[0]["sufficient_k5"] == "unknown"
