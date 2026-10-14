"""Regression tests for retrospective gate discrimination and fixed auditing."""
import json
from pathlib import Path
import pytest

from emrag.diagnostics import (
    auc_pairwise, blocking_reasons, read_traces, report_cases,
    root_cause_signals, stratified_sample, annotation_template, audit_casebook,
)


def mk_case(qid, before, proposed, after, gold, support=(), sufficient=False,
            raw_docs=False):
    assert before in {"A", "B"} and proposed in {"A", "B"}
    score_a = {
        "sufficient": sufficient if proposed == "A" else False,
        "support_ids": list(support) if proposed == "A" else [],
        "refute_ids": [], "missing_facts": [] if sufficient and proposed == "A" else ["gap"],
        "score": 2 if proposed == "A" else -1,
    }
    score_b = {
        "sufficient": sufficient if proposed == "B" else False,
        "support_ids": list(support) if proposed == "B" else [],
        "refute_ids": [], "missing_facts": [] if sufficient and proposed == "B" else ["gap"],
        "score": 2 if proposed == "B" else -1,
    }
    return {
        "id": qid, "question": f"Question {qid}?", "options": {"A": "alpha", "B": "beta"},
        "baseline": before, "selected": after, "status": "unverified",
        "evidence": [{"id": "e1", "quote": "A source quote", "source": "demo"}],
        "documents": [{"id": "d1", "contents": "A source quote"}] if raw_docs else [],
        "traces": [
            {"round": 0, "proposed_choice": before,
             "selected_choice": before, "status": "unverified"},
            {"round": 1, "proposed_choice": proposed, "selected_choice": after,
             "status": "verified" if after != before else "unverified",
             "scores": {"A": score_a, "B": score_b},
             "action": "preserved" if before == after else "accepted"},
        ],
    }


def demo_rows():
    return [
        mk_case("rescue_blocked", "B", "A", "B", "A"),
        mk_case("harm_blocked", "A", "B", "A", "A"),
        mk_case("rescue_accepted", "B", "A", "A", "A", ("e1",), True, True),
        mk_case("harm_accepted", "A", "B", "B", "A", ("e1",), True),
        mk_case("reader_failure", "B", "B", "B", "A"),
    ]


def test_proposal_gate_confusion_matrix_and_oracle():
    rows = demo_rows()
    gold = {r["id"]: "A" for r in rows}
    stats, cases = report_cases(rows, gold)
    counts = stats["proposal_transition_counts"]
    assert counts["rescue_blocked"] == 1
    assert counts["harm_blocked"] == 1
    assert counts["rescue_adopted"] == 1
    assert counts["harm_adopted"] == 1
    assert stats["reader_never_proposed_gold_count"] == 1
    assert stats["candidate_oracle_accuracy"] == pytest.approx(4 / 5)
    assert stats["baseline_accuracy"] == pytest.approx(2 / 5)
    assert stats["final_accuracy"] == pytest.approx(2 / 5)
    case = {c["id"]: c for c in cases}
    assert case["rescue_blocked"]["stratum"] == "rescue_blocked"
    assert "judge_says_insufficient" in case["rescue_blocked"]["events"][0]["blocking_reasons"]
    assert stats["independently_annotated_retrieval_cases"] == 0
    assert stats["cases_with_raw_documents"] == 1


def test_deterministic_sample_and_casebook_do_not_fake_raw_context():
    rows = demo_rows()
    _, cases = report_cases(rows, {r["id"]: "A" for r in rows})
    selection1 = stratified_sample(cases, per_stratum=1, seed=45)
    selection2 = stratified_sample(list(reversed(cases)), per_stratum=1, seed=45)
    assert [c["id"] for c in selection1] == [c["id"] for c in selection2]
    md = audit_casebook(rows, selection1)
    assert "Not recorded" in md
    assert "Gold" in md
    annotation = annotation_template(selection1[0])
    assert annotation["retrieved_sufficient"] == "unknown"


def test_independent_annotations_required_for_causal_diagnosis():
    _, cases = report_cases(demo_rows(), {r["id"]: "A" for r in demo_rows()})
    failing = next(c for c in cases if c["id"] == "reader_failure")
    assert root_cause_signals(failing, None) == [
        "independent_evidence_annotation_missing"]
    independent = {"retrieved_sufficient": "yes",
                   "active_context_sufficient": "yes"}
    assert "reader_no_correct_candidate_given_annotated_context" in (
        root_cause_signals(failing, independent))
    missing = {"retrieved_sufficient": "no",
               "active_context_sufficient": "unknown"}
    assert root_cause_signals(failing, missing) == ["retrieval_not_sufficient"]


def test_auc_ties_and_original_margin_explanations():
    assert auc_pairwise([0.0, 1.0], [0.0, -1.0]) == 0.875
    assert auc_pairwise([], [1.0]) is None
    assert blocking_reasons({"scores": {}}, "B", "A") == ["scorecard_unavailable"]


def test_trace_reader_validates_duplicate_id(tmp_path):
    file = tmp_path / "traces.jsonl"
    data = demo_rows()[0]
    file.write_text(json.dumps(data) + "\n" + json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate trace"):
        read_traces(file)
