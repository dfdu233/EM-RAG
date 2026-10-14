import pytest
from emrag.core import Document, EvidenceLedger, ScoreCard, sanitize_card
from emrag.model import parse_json
from emrag.prompts import answer_choice


def test_exact_source_quote_and_no_generated_evidence():
    book = EvidenceLedger()
    fresh = book.add_documents([
        Document("x", "Drug X is not indicated for population Y.", source="medical"),
        Document("y", "Drug X is not indicated for population Y.", source="medical"),
        Document("z", "Generated advice", kind="generated"),
    ], round_id=1)
    assert len(fresh) == 1
    assert len(book.documents) == 1
    doc = fresh[0]
    claims = book.add_claims([
        {"doc_id": doc.id, "claim": "not indicated for Y",
         "quote": "Drug X is not indicated for population Y."},
        {"doc_id": doc.id, "claim": "fake", "quote": "not in the source"},
        {"doc_id": "fabricated", "claim": "made up", "quote": "made up"},
    ])
    assert len(claims) == 1
    assert claims[0].quote in book.documents[claims[0].doc_id].contents


def test_no_invented_citation_and_conflict():
    card = sanitize_card("B", {
        "support_ids": ["e-real", "e-forged", "e-real", "e-shared"],
        "refute_ids": ["e-shared"],
        "missing_facts": [], "sufficient": True,
    }, {"e-real", "e-shared"})
    assert card.support_ids == ("e-real",)
    assert card.refute_ids == ("e-shared",)
    assert not card.eligible()
    assert card.score == -2


def test_protected_older_evidence_not_evicted():
    ledger = EvidenceLedger()
    d1 = ledger.add_documents([Document("a", "Old correct source.")], 1)[0]
    old = ledger.add_claims([{"doc_id": d1.id, "quote": "Old correct source.",
                              "claim": "Old source states X."}])[0]
    d2 = ledger.add_documents([Document("b", "New source text.")], 2)[0]
    new = ledger.add_claims([{"doc_id": d2.id, "quote": "New source text.",
                              "claim": "New source states Y."}])[0]
    assert ledger.select(1, 9000, {old.id})[0].id == old.id
    assert ledger.select(1, 9000)[0].id == new.id


def test_invalid_model_output_does_not_default_to_A():
    assert answer_choice({"choice": "A, B"}, ("A", "B")) is None
    assert parse_json('prefix {"choice":"B"} suffix') == {"choice": "B"}
    with pytest.raises(ValueError):
        parse_json("not json")
