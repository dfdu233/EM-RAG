import json
from pathlib import Path
import pytest

from rag_mechanisms.data import Question, Passage, write_jsonl
from rag_mechanisms.analysis import report, paired_interval
from rag_mechanisms.audit import prepare_casebook, annotation_template


def rows():
    records = []
    # q1: baseline correct -> k1 wrong -> k5 correct
    # q2: baseline wrong -> k1 correct -> k5 wrong
    data = {
        "1": ("A", "B", "A", "A"),
        "2": ("B", "A", "B", "A"),
    }
    for qid, (a0, a1, a5, agen) in data.items():
        for name, pred, k in (("retrieved_k0", a0, 0),
                              ("retrieved_k1", a1, 1),
                              ("retrieved_k5", a5, 5),
                              ("generated_k5", agen, 5)):
            records.append({
                "dataset": "medqa", "id": qid, "reader": "dummy",
                "condition": name, "status": "ok", "prediction": pred,
                "source_type": "external_retrieved",
                "document_count": k, "document_characters": k * 50,
                "prompt_chars": 200 + k * 50,
            })
    return records


def test_paired_report_and_nonmonotonicity(tmp_path):
    f = tmp_path / "predictions.jsonl"
    write_jsonl(f, rows())
    stats = report(str(f), "medqa", {"1": "A", "2": "A"},
                   seed=7, bootstrap_samples=100)
    p = stats["paired_comparisons"]["retrieved_k0 -> retrieved_k1"]
    assert p["n"] == 2
    assert p["rescue_count"] == 1
    assert p["harm_count"] == 1
    assert p["mean_delta"] == 0
    assert stats["depth_behaviour"]["n_fully_paired_across_depths"] == 2
    assert stats["depth_behaviour"]["oracle_for_diagnosis_only"] is True
    assert stats["summary"]["no_emrag_modules_compared"] is True


def test_report_refuses_duplicate_or_mismatched_dataset(tmp_path):
    f = tmp_path / "pred.jsonl"
    write_jsonl(f, rows() + [rows()[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        report(str(f), "medqa", {"1": "A", "2": "A"})


def test_manual_audit_never_claims_auto_sufficiency(tmp_path):
    file = tmp_path / "predictions.jsonl"
    write_jsonl(file, rows())
    questions = {
        "1": Question("1", "Question 1?", {"A": "a", "B": "b"}),
        "2": Question("2", "Question 2?", {"A": "a", "B": "b"}),
    }
    docs = {q: [Passage(str(i), f"Source {q} {i} text") for i in range(20)]
            for q in questions}
    cases, casebook = prepare_casebook(
        str(file), "medqa", questions,
        {"1": "A", "2": "A"}, docs, per_category=5)
    assert cases
    labels = annotation_template(cases)
    assert all(x["sufficient_k20"] == "unknown" for x in labels)
    assert "MedRGAG" in casebook
    assert "sufficient_k20" in casebook
