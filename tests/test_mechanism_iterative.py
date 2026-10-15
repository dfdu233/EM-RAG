import pytest
from rag_mechanisms.iterative import LiveMedRGAG
from rag_mechanisms.iterative_report import report


class StubCorpus:
    def __init__(self, name):
        self.name = name

    def get_relevant_documents(self, query, k=5):
        return ([{"id": f"{self.name}-{i}",
                  "contents": f"{self.name} {query} unique {i} content"}
                 for i in range(k)],
                [float(k-i) for i in range(k)])


def test_live_retriever_interleaves_distinct_corpora():
    retriever = object.__new__(LiveMedRGAG)
    retriever.system = type("Stub", (), {
        "retrievers": [StubCorpus("textbooks"), StubCorpus("wikipedia")]})()
    retriever.corpus = "MedText"
    retriever.calls = 0
    docs = retriever.search("medical question", 4)
    assert len(docs) == 4
    assert [x.source for x in docs] == [
        "MedText:0", "MedText:1", "MedText:0", "MedText:1"]
    assert retriever.calls == 2


def test_iterative_report_paired_correct_and_wrong():
    rows = [{
        "id": "q1", "steps": [
            {"condition": "static", "round": 1, "status": "ok",
             "prediction": "A", "document_count": 4, "prompt_chars": 500},
            {"condition": "adaptive", "round": 1, "status": "ok",
             "prediction": "B", "document_count": 4, "prompt_chars": 500},
        ]},
        {"id": "q2", "steps": [
            {"condition": "static", "round": 1, "status": "ok",
             "prediction": "B", "document_count": 4, "prompt_chars": 500},
            {"condition": "adaptive", "round": 1, "status": "ok",
             "prediction": "A", "document_count": 4, "prompt_chars": 500},
        ]}
    ]
    result = report(rows, {"q1": "A", "q2": "A"}, boots=100)
    assert result["rounds"]["1"]["adaptive_extra_rescue"] == 1
    assert result["rounds"]["1"]["adaptive_extra_harm"] == 1
    assert result["rounds"]["1"]["mean_delta"] == 0
