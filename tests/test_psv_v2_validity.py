"""No gold / no API interface regressions: query budget and answer parsing."""
import pytest

from coverage_medrgag.progressive_validity import (
    fit_query, recover_answer, view_provenance,
)
from coverage_medrgag.data import Passage


def test_queries_are_under_hard_limit_and_keep_late_conditions():
    original = ("general context staging survival statistics and studies " * 12
                + " independently adjusted hazard ratio after controlling grade")
    fixed = fit_query(
        original, relation="adjusted hazard ratio independent outcome",
        question="Does effect remain after adjustment for grade?")
    assert len(original) > 300
    assert fixed.status == "compacted_unverified"
    assert 5 <= len(fixed.query) <= 300
    assert "adjusted" in fixed.query.lower()
    assert "after" in fixed.query.lower()
    assert "controlling" in fixed.query.lower()
    assert fixed.original_query == original
    assert fixed.original_length == len(original)
    rejected = fit_query(original, relation="independent outcome",
                         question="Effect?", strategy="reject")
    assert rejected.query is None
    assert rejected.status == "over_limit_rejected"
    assert fit_query("alpha study query", relation="study", question="alpha"
                     ).status == "unchanged"


def test_only_unambiguous_explicit_answer_is_recovered():
    labels = ("A", "B", "C")
    assert recover_answer(
        '{"reasoning":"A is less likely\\nBut B fits","answer_choice":"B"}',
        labels).prediction == "B"
    assert recover_answer(
        '{"reasoning":"multiple options","answer":"C"}',
        labels).method == "alternate_json_field"
    assert recover_answer("Final Answer: B", labels).prediction == "B"
    assert recover_answer("A is plausible, B is not", labels).prediction is None
    assert recover_answer("1. A: yes\n2. B: no", labels).prediction is None
    assert recover_answer("Final Answer: B\nFinal Answer: A",
                          labels).prediction is None
    assert recover_answer('{"answer_choice":"A","answer":"B"}',
                          labels).prediction is None
    assert recover_answer('{"reasoning":"B is true"}',
                          labels).prediction is None
    assert recover_answer("No label", labels).prediction is None


def test_source_provenance_distinguishes_generated_and_retrieved():
    anchors = [
        Passage("r0", "actual source", "retrieved"),
        Passage("g0", "unverified generated", "parametric_generated"),
    ]
    fresh = [Passage("n0", "new source", "retrieved")]
    data = view_provenance(anchors, fresh, anchors)
    assert data["anchor_generated_count"] == 1
    assert data["fresh_generated_count"] == 0
    assert data["fresh_ids"] == ["n0"]


def test_invalid_query_repair_policy_fails_explicitly():
    with pytest.raises(ValueError):
        fit_query("test query", relation="x", question="x",
                  strategy="unknown")
