import json

from emrag.core import Document, EvidenceLedger, Question
from emrag.engine import EMRAG, EMRAGConfig
from emrag.model import ScriptedModel


def test_two_quotes_from_one_passage_are_not_two_support_votes():
    ledger = EvidenceLedger()
    doc = ledger.add_documents(
        [Document("doc", "Alpha fact is true. Beta fact is true.")], 1)[0]
    evidence = ledger.add_claims([
        {"doc_id": doc.id, "quote": "Alpha fact is true.", "claim": "Alpha true"},
        {"doc_id": doc.id, "quote": "Beta fact is true.", "claim": "Beta true"},
    ])
    payload = {"cards": {"A": {
        "support_ids": [x.id for x in evidence],
        "refute_ids": [], "missing_facts": [], "sufficient": True,
    }}}
    model = ScriptedModel([json.dumps(payload)])
    engine = EMRAG(model, retriever=None)
    q = Question("q", "Which option?", {"A": "true", "B": "false"})
    cards = engine._judge(q, evidence)
    assert len(cards["A"].support_ids) == 1
    assert cards["A"].score == 2


class DiversifiedSearch:
    def search(self, query, k, *, question_id=""):
        return [Document(f"{query}-{i}", f"{query} passage {i}") for i in range(k)]


def test_targeted_query_hits_interleave_before_doc_budget():
    engine = EMRAG(ScriptedModel([]), DiversifiedSearch(), EMRAGConfig(
        queries_per_round=2, docs_per_query=2, max_new_docs=3))
    q = Question("q", "Synthetic question?", {"A": "a", "B": "b"})
    docs = engine._search(q, ["queryA", "queryB"])
    assert [x.id for x in docs[:3]] == ["queryA-0", "queryB-0", "queryA-1"]


def test_evidence_context_prefers_distinct_documents():
    ledger = EvidenceLedger()
    first = ledger.add_documents([Document("A", "A fact one. A fact two.")], 1)[0]
    second = ledger.add_documents([Document("B", "B fact three.")], 2)[0]
    ledger.add_claims([
        {"doc_id": first.id, "claim": "A fact one holds", "quote": "A fact one."},
        {"doc_id": first.id, "claim": "A fact two holds", "quote": "A fact two."},
        {"doc_id": second.id, "claim": "B fact three holds", "quote": "B fact three."},
    ])
    selected = ledger.select(max_items=2, max_chars=9999)
    assert len({x.doc_id for x in selected}) == 2
