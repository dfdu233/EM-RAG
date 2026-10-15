"""Executable offline integration: cached MedRGAG 5+5 => KADS => frozen Reader.

The real model is mocked; this validates caching, gold isolation, output
resumption, original/reference semantics, and paired evaluation. Real medical
accuracy and model weights are not tested by this fixture.
"""
import json
from pathlib import Path

import pytest

from coverage_medrgag import cli


def write_medrgag(tmp_path):
    root = tmp_path / "MedRGAG"
    (root / "src").mkdir(parents=True)
    dataset = "medqa"
    base = root / "outputs" / dataset / "docs"
    (base / "retrieval").mkdir(parents=True)
    (base / "generate").mkdir(parents=True)
    (root / "benchmark.json").write_text(json.dumps({
        "medqa": {
            "q1": {"question": "Synthetic medical question?",
                   "options": {"A": "Alpha", "B": "Beta"}, "answer": "B"},
            "q2": {"question": "Another synthetic question?",
                   "options": {"A": "Alpha", "B": "Beta"}, "answer": "A"},
        }
    }), encoding="utf-8")
    (root / "src" / "template.py").write_text(
        'rag_template = "Question:\\n{question}\\nDocuments:\\n{documents}"\n'
        'llm_selection_prompt = "Pick five from ten. {question}\\n{documents}"\n',
        encoding="utf-8")
    retrieval = {}
    generation = {}
    selections = {}
    for qid in ("q1", "q2"):
        retrieval[qid] = {
            "snippets": [
                {"id": f"{qid}_r{i}", "contents": f"Fictional retrieved {qid} source {i}."}
                for i in range(32)],
            "scores": [1/(1+i) for i in range(32)]}
        generation[qid] = {
            "snippets": [
                {"id": f"{qid}_g{i}", "contents": f"Fictional KGCC {qid} knowledge {i}."}
                for i in range(5)]}
        selections[qid] = {"final_list": [0,1,5,6,7]}
    (base / "retrieval" / "snippets.json").write_text(
        json.dumps(retrieval), encoding="utf-8")
    (base / "generate" / "snippets.json").write_text(
        json.dumps(generation), encoding="utf-8")
    (base / "generate" / "snippets_final_select.json").write_text(
        json.dumps(selections), encoding="utf-8")
    return root


class MockChat:
    prompts = []
    def __init__(self, model, **kwargs):
        self.model = model
        self.calls = 0

    def complete(self, system, user, *, temperature=0, seed=None, max_tokens=512):
        self.calls += 1
        self.prompts.append((self.model, system, user))
        if system.startswith("You assess whether"):
            # Source-only early stopping with independently inspectable quote.
            return json.dumps({
                "sufficient": True,
                "supported_choice": "A",
                "key_evidence": [{
                    "doc_id":"r0",
                    "quote":"Fictional retrieved "
                }],
                "missing_facts": []})
        if system.startswith("Select up to five"):
            return "- Final Selection: [0] [1] [5] [6] [7]"
        return '{"reasoning":"fixture only","answer_choice":"A"}'


def test_three_mode_cli_and_gold_isolation(tmp_path, monkeypatch):
    root = write_medrgag(tmp_path)
    monkeypatch.setattr(cli, "OpenAIChat", MockChat)
    sample_args = [
        "--medrgag-root", str(root), "--dataset", "medqa",
        "--reader-model", "fixed-reader",
        "--no-rerank", "--limit", "2",
    ]
    outs = {}
    for mode in ("reference", "static", "adaptive"):
        folder = tmp_path / mode
        outs[mode] = folder
        args = [
            "run", "--mode", mode, "--output", str(folder),
            *sample_args
        ]
        if mode != "reference":
            args += ["--aux-model", "fixture-helper"]
        assert cli.main(args) == 0
        rows = [
            json.loads(x) for x in
            (folder / "predictions.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
        assert all(r["prediction"] == "A" for r in rows)
        assert all(r["reader_calls"] == 1 for r in rows)
        assert rows[0]["depth_used"] == 5
        if mode == "reference":
            assert rows[0]["auxiliary_calls"] == 0
        if mode == "adaptive":
            assert rows[0]["source_proxy_sufficient"] is True
            assert rows[0]["auxiliary_calls"] == 2
        # Exact repeat must skip completed records and not double-write.
        assert cli.main(args) == 0
        assert len((folder / "predictions.jsonl").read_text().splitlines()) == 2
    # LLM saw question/options but never the gold answer label as oracle input.
    # q1 gold is B, but the stub always emitted A because no gold was provided.
    assert all('"answer": "B"' not in x[2] for x in MockChat.prompts)
    report = tmp_path / "report.json"
    assert cli.main([
        "evaluate", "--dataset", "medqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(outs["adaptive"] / "predictions.jsonl"),
        "--compare", str(outs["static"] / "predictions.jsonl"),
        "--bootstrap", "50", "--output", str(report)
    ]) == 0
    data = json.loads(report.read_text())
    assert data["summary"]["accuracy"] == .5
    assert data["comparisons"]["vs_other_run"]["paired_n"] == 2


def test_refuse_changed_data_or_config(tmp_path, monkeypatch):
    root = write_medrgag(tmp_path)
    monkeypatch.setattr(cli, "OpenAIChat", MockChat)
    folder = tmp_path / "run"
    args = ["run", "--mode", "static",
            "--medrgag-root", str(root), "--dataset", "medqa",
            "--reader-model", "reader", "--aux-model", "helper",
            "--no-rerank", "--limit", "1", "--output", str(folder)]
    assert cli.main(args) == 0
    with pytest.raises(ValueError, match="Refusing to mix"):
        cli.main(args + ["--seed", "99"])
