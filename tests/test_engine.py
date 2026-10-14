import json
import re
from emrag.core import Document, Question, ScoreCard
from emrag.engine import EMRAG, EMRAGConfig


class OneDocRetriever:
    def search(self, query, k, *, question_id=""):
        return [Document("source1",
                         "Drug X is contraindicated for patients with condition Y.",
                         source="synthetic")]


class AuditFixtureModel:
    """Simulates the response protocol; does not check medical correctness."""

    def __init__(self, inject_bad_quote=False):
        self.calls = 0
        self.inject_bad_quote = inject_bad_quote

    def complete(self, system, user, *, temperature=0.0):
        self.calls += 1
        if system.startswith("You extract ATOMIC"):
            match = re.search(r"DOCUMENT_ID (d-[a-f0-9]{16})", user)
            quote = ("Fabricated quote" if self.inject_bad_quote else
                     "Drug X is contraindicated for patients with condition Y.")
            return json.dumps({"claims": [{
                "doc_id": match.group(1), "claim": "Drug X contraindicated in Y",
                "quote": quote,
            }]})
        if system.startswith("Act as an evidence-only"):
            ids = re.findall(r"\[(e-[0-9a-f]{16})\]", user)
            return json.dumps({"cards": {
                "A": {"support_ids": ids, "refute_ids": [],
                      "missing_facts": [], "sufficient": bool(ids)},
                "B": {"support_ids": [], "refute_ids": [],
                      "missing_facts": ["contraindication"], "sufficient": False},
            }})
        if system.startswith("Answer the multiple-choice"):
            choice = ("A" if "Verbatim source quote: Drug X is contraindicated" in user
                      else "B")
            return json.dumps({"choice": choice, "reason": "fixture"})
        return json.dumps({"missing_facts": [], "disputes": [], "queries": []})


def question():
    return Question("q", "Which drug is contraindicated for condition Y?",
                    {"A": "Drug X", "B": "Drug Q"})


def test_end_to_end_accepts_only_with_quoted_support():
    model = AuditFixtureModel()
    result = EMRAG(model, OneDocRetriever(), EMRAGConfig(
        max_rounds=1, queries_per_round=1)).run(question())
    assert result.baseline == "B"
    assert result.selected == "A"
    assert result.status == "verified"
    assert result.traces[1].action == "accepted"
    assert len(result.evidence) == 1
    assert result.model_calls == 4  # baseline, extract, propose, judge


def test_forged_extraction_does_not_update_answer():
    model = AuditFixtureModel(inject_bad_quote=True)
    result = EMRAG(model, OneDocRetriever(), EMRAGConfig(
        max_rounds=1, queries_per_round=1)).run(question())
    assert result.baseline == "B"
    assert result.selected == "B"
    assert result.status == "unverified"
    assert not result.evidence


def test_gate_preserves_and_can_abstain_when_refuted():
    engine = EMRAG(AuditFixtureModel(), OneDocRetriever())
    cards = {
        "A": ScoreCard("A", support_ids=("e1",), sufficient=True),
        "B": ScoreCard("B", sufficient=False),
    }
    assert engine._decide("B", "unverified", cards, {"A", "B"})[:3] == (
        "A", "verified", "accepted")
    tied = {
        "A": ScoreCard("A", support_ids=("e1",), sufficient=True),
        "B": ScoreCard("B", support_ids=("e2",), sufficient=True),
    }
    assert engine._decide("B", "verified", tied, {"A", "B"})[:3] == (
        "B", "verified", "preserved")
    refuted = {
        "A": ScoreCard("A", refute_ids=("e2",)),
        "B": ScoreCard("B", sufficient=False),
    }
    assert engine._decide("A", "verified", refuted, {"A", "B"})[:3] == (
        None, "abstained", "invalidated_by_contradiction")


def test_abstain_mode_without_evidence():
    result = EMRAG(AuditFixtureModel(inject_bad_quote=True), OneDocRetriever(),
                   EMRAGConfig(max_rounds=1, queries_per_round=1,
                               unverified_policy="abstain")).run(question())
    assert result.selected is None
    assert result.status == "abstained"
