"""Reader-only GAG ablation tests; no GPU/API/model training."""
import json
import re

from emrag.core import Document, EvidenceLedger, Question
from emrag.engine import EMRAG, EMRAGConfig
from emrag.gag import load_generated_snippets, bounded_background
from emrag.prompts import answer_prompt


class FakeRetriever:
    def search(self, query, k, *, question_id=""):
        return [Document("source1", "In the fictional example, Option A is supported.",
                         source="synthetic")]


class CaptureModel:
    """Make the Reader sensitive to GAG while keeping the Judge source-only."""

    def __init__(self):
        self.records = []

    def complete(self, system, user, *, temperature=0.0):
        self.records.append((system, user))
        if system.startswith("You extract ATOMIC"):
            doc_id = re.search(r"DOCUMENT_ID (d-[a-f0-9]{16})", user).group(1)
            return json.dumps({"claims": [{
                "doc_id": doc_id,
                "claim": "Option A is supported in the fictional example",
                "quote": "In the fictional example, Option A is supported.",
            }]})
        if system.startswith("Act as an evidence-only"):
            # A hard invariant: generated text is not a Judge evidence source.
            assert "UNVERIFIED MODEL-GENERATED BACKGROUND" not in user
            assert "Parametric B hint" not in user
            evidence_ids = re.findall(r"\[(e-[0-9a-f]{16})\]", user)
            return json.dumps({"cards": {
                "A": {"support_ids": evidence_ids, "refute_ids": [],
                      "missing_facts": [], "sufficient": bool(evidence_ids)},
                "B": {"support_ids": [], "refute_ids": [],
                      "missing_facts": ["No external support for B"],
                      "sufficient": False},
            }})
        if system.startswith("Answer the multiple-choice"):
            return json.dumps({"choice": (
                "B" if "Parametric B hint" in user else "A")})
        return json.dumps({"missing_facts": [], "disputes": [], "queries": []})


def test_gag_off_preserves_original_v1_reader_prompt():
    q = Question("q", "Which option?", {"A": "A label", "B": "B label"})
    expected = (q.formatted() + "\n\nEXTERNAL QUOTED EVIDENCE "
                "(may be incomplete):\n(none)\nOutput a single choice JSON.")
    assert answer_prompt(q, []) == expected
    assert "UNVERIFIED" not in answer_prompt(q, [])


def test_medrgag_generated_cache_and_bound(tmp_path):
    path = tmp_path / "snippets.json"
    path.write_text(json.dumps({"q1": {"snippets": [
        {"id": "g1", "contents": "Generated text one 123"},
        {"id": "g2", "contents": "Generated text one 123"},
        {"id": "g3", "contents": "Generated text two"},
    ]}, "q2": {"snippets": []}}), encoding="utf-8")
    passages = load_generated_snippets(path)["q1"]
    assert passages == ["Generated text one 123", "Generated text two"]
    assert len(bounded_background(passages, max_docs=2, max_chars=22)) == 2
    assert sum(map(len, bounded_background(passages, max_docs=2, max_chars=22))) <= 22
    assert load_generated_snippets(path)["q2"] == []


def test_generated_text_never_becomes_retrieved_evidence():
    ledger = EvidenceLedger()
    assert ledger.add_documents([Document(
        "g1", "Parametric B hint", kind="generated")], round_id=1) == []
    assert ledger.evidence == {}
    assert ledger.documents == {}


def test_reader_only_background_off_evidence_gate_unchanged():
    question = Question("q", "In this fictional example, which option?",
                        {"A": "Option A", "B": "Option B"})
    model = CaptureModel()
    result = EMRAG(
        model, FakeRetriever(),
        EMRAGConfig(max_rounds=1, queries_per_round=1, update_policy="evidence")
    ).run(question, gag_background=["Parametric B hint"])
    assert result.baseline == "A"  # GAG must NOT affect round-0 baseline
    assert result.traces[1].proposed_choice == "B"
    assert result.selected == "A"  # no external support for B
    assert result.status == "verified"
    assert all("Parametric B hint" not in e.quote for e in result.evidence)
    assert all("Parametric B hint" not in e.statement for e in result.evidence)
    assert sum("Parametric B hint" in user for _, user in model.records) == 1


def test_latest_policy_exposes_candidate_generation_gain_without_gate():
    question = Question("q", "In this fictional example, which option?",
                        {"A": "Option A", "B": "Option B"})
    model = CaptureModel()
    result = EMRAG(
        model, FakeRetriever(),
        EMRAGConfig(max_rounds=1, queries_per_round=1, update_policy="latest")
    ).run(question, gag_background=["Parametric B hint"])
    assert result.traces[1].proposed_choice == "B"
    assert result.selected == "B"
    assert result.status == "unverified"


def test_absent_gag_keeps_candidate_same_as_original():
    question = Question("q", "In this fictional example, which option?",
                        {"A": "Option A", "B": "Option B"})
    model = CaptureModel()
    result = EMRAG(
        model, FakeRetriever(),
        EMRAGConfig(max_rounds=1, queries_per_round=1, update_policy="latest")
    ).run(question)
    assert result.traces[1].proposed_choice == "A"
    assert result.selected == "A"
