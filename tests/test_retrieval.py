import json
from emrag.retrieval import JSONLBM25Retriever, MedRGAGSeedRetriever


def test_local_bm25(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    rows = [
        {"id": "1", "contents": "The blue molecule inhibits kinase alpha"},
        {"id": "2", "contents": "The red molecule inhibits kinase beta"},
    ]
    corpus.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    r = JSONLBM25Retriever([corpus])
    assert r.search("kinase beta", 1)[0].id == "2"
    assert r.search("unmatchedword", 5) == []


def test_precomputed_only_first_query(tmp_path):
    saved = tmp_path / "cached.json"
    saved.write_text(json.dumps({"q1": {"snippets": [
        {"id": "doc1", "contents": "Some background"}
    ], "scores": [4.0]}}), encoding="utf-8")
    r = MedRGAGSeedRetriever(str(saved))
    assert len(r.search("original", 5, question_id="q1")) == 1
    assert r.search("new query", 5, question_id="q1") == []
