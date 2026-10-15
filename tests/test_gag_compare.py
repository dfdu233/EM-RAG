"""Offline, gold-isolated four-arm comparison regression tests."""
from copy import deepcopy
import pytest
from scripts.compare_gag_ablation import compare, summarize_variant


def row(qid, initial, proposed, selected, docs=("doc1",)):
    return {
        "id": qid, "model_calls": 5,
        "traces": [
            {"round": 0, "selected_choice": initial, "proposed_choice": initial},
            {"round": 1, "selected_choice": selected, "proposed_choice": proposed,
             "retrieved_doc_ids": list(docs)},
        ],
    }


def four_arms():
    # q1: only GAG proposes correct B, external Gate deliberately blocks it.
    # q2: GAG leaves existing correct A intact.
    return {
        "without_gag_gate": {
            "q1": row("q1", "A", "A", "A"),
            "q2": row("q2", "A", "A", "A"),
        },
        "with_gag_gate": {
            "q1": row("q1", "A", "B", "A"),
            "q2": row("q2", "A", "A", "A"),
        },
        "without_gag_latest": {
            "q1": row("q1", "A", "A", "A"),
            "q2": row("q2", "A", "A", "A"),
        },
        "with_gag_latest": {
            "q1": row("q1", "A", "B", "B"),
            "q2": row("q2", "A", "A", "A"),
        },
    }


def test_separates_candidate_gain_from_gate_discrimination():
    result = compare(four_arms(), {"q1": "B", "q2": "A"})
    stats = result["variants"]
    assert stats["without_gag_gate"]["reader_candidate_accuracy"] == .5
    assert stats["with_gag_gate"]["reader_candidate_accuracy"] == 1
    assert stats["with_gag_gate"]["final_accuracy"] == .5
    assert stats["with_gag_gate"]["correct_candidates_blocked"] == 1
    assert stats["with_gag_latest"]["final_accuracy"] == 1
    assert result["diagnostic_deltas"]["gag_candidate_accuracy_vs_no_gag_under_gate"] == .5
    assert result["diagnostic_deltas"]["gag_final_accuracy_vs_no_gag_under_gate"] == 0.0
    assert result["candidate_disagreement_between_gate_and_latest_repeats"]["gag"] == 0


def test_comparison_refuses_mismatched_retrieval():
    variants = four_arms()
    variants["with_gag_gate"]["q1"]["traces"][1]["retrieved_doc_ids"] = ["another_doc"]
    with pytest.raises(ValueError, match="Retrieved documents differ"):
        compare(variants, {"q1": "B", "q2": "A"})


def test_comparison_refuses_different_baseline():
    variants = four_arms()
    variants["with_gag_gate"]["q1"]["traces"][0]["selected_choice"] = "B"
    with pytest.raises(ValueError, match="Baseline changed"):
        compare(variants, {"q1": "B", "q2": "A"})
