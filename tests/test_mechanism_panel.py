import pytest

from rag_mechanisms.panel import panel
from rag_mechanisms.annotation_report import summarize_annotations


def test_cross_method_common_harm_and_rescue():
    # q1 correct closed-book; two retrieval methods corrupt it.
    # q2 wrong closed-book; two augmentation methods repair it.
    gold = {"q1": "A", "q2": "A", "q3": "B"}
    refs = {
        "DirectResponse": {"q1": "A", "q2": "B", "q3": "B"},
        "VanillaRAG": {"q1": "B", "q2": "A", "q3": "B"},
        "MedRAG": {"q1": "B", "q2": "A", "q3": "B"},
        "MedRGAG": {"q1": "A", "q2": "B", "q3": "B"},
    }
    result = panel(gold, ["q1", "q2", "q3"], refs, bootstrap_samples=20)
    assert result["n_cases_harmed_by_multiple_methods"] == 1
    assert result["n_cases_rescued_by_multiple_methods"] == 1
    assert result["methods"]["VanillaRAG"]["harm_count"] == 1
    assert result["methods"]["VanillaRAG"]["rescue_count"] == 1
    assert result["methods"]["VanillaRAG"]["mean_delta"] == 0


def test_annotation_separates_evidence_supply_from_reader_failure():
    annotations = [{
        "dataset": "medqa", "id": "q1",
        "sufficient_k5": "no", "sufficient_k20": "yes",
        "decisive_new_fact_5_to_20": "yes",
    }, {
        "dataset": "medqa", "id": "q2",
        "sufficient_k5": "yes", "sufficient_k20": "yes",
        "decisive_new_fact_5_to_20": "no",
    }]
    preds = [
        {"dataset": "medqa", "id": "q1", "condition": "retrieved_k5",
         "status": "ok", "prediction": "B"},
        {"dataset": "medqa", "id": "q1", "condition": "retrieved_k20",
         "status": "ok", "prediction": "B"},
        {"dataset": "medqa", "id": "q2", "condition": "retrieved_k5",
         "status": "ok", "prediction": "A"},
        {"dataset": "medqa", "id": "q2", "condition": "retrieved_k20",
         "status": "ok", "prediction": "A"},
    ]
    data = summarize_annotations(annotations, preds, {"q1": "A", "q2": "A"}, "medqa")
    assert data["evidence_sufficiency_vs_reader_performance"]["20"]["yes"]["n"] == 2
    assert data["evidence_sufficiency_vs_reader_performance"]["20"]["yes"]["reader_failure_despite_human_sufficiency"] == 1
    assert data["became_sufficient_between_5_and_20"]["still_wrong_despite_new_sufficiency"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_annotations(annotations + [annotations[0]], preds,
                              {"q1": "A", "q2": "A"}, "medqa")
