import json
import pytest

from coverage_medrgag.data import Passage, Question
from coverage_medrgag.models import IdentityRanker
from coverage_medrgag.pipeline import (
    Config, CoverageAdaptiveMedRGAG, _key_documents,
    choose_retrieved_docs
)


class ScriptedChat:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.requests = []

    def complete(self, system, user, *, temperature=0.0, seed=None, max_tokens=512):
        self.calls += 1
        self.requests.append((system, user, temperature, seed))
        return self.outputs.pop(0)


def q():
    return Question("x", "A synthetic question with two choices?",
                    {"A":"alpha","B":"beta"})


def docs():
    external = [Passage(f"r{i}", f"Source {i} says fabricated fact number {i}.",
                        "retrieved") for i in range(32)]
    kgcc = [Passage(f"g{i}", f"Generated background {i}.",
                    "parametric_generated") for i in range(5)]
    return external, kgcc


def build(mode, reader, auxiliary):
    return CoverageAdaptiveMedRGAG(
        reader=reader, auxiliary=auxiliary, ranker=IdentityRanker(),
        rag_template="Question: {question}\nRetrieved Documents: {documents}",
        kads_template="Select 5\nQuestion: {question}\nDocs: {documents}",
        config=Config(mode=mode, depths=(5,10,20,32)))


def test_reference_replays_exact_medrgag_cache_without_auxiliary_or_gate():
    reader = ScriptedChat(['{"reasoning":"this is hypothetical","answer_choice":"A"}'])
    e = build("reference", reader, None)
    ret, gen = docs()
    result = e.run(q(), ret, gen, original_selection=(5, 7, 0, 2, 8))
    assert result["selected_doc_ids"] == ["g0", "g2", "r0", "r2", "g3"]
    assert result["depth_used"] == 5
    assert result["auxiliary_calls"] == 0
    assert result["reader_calls"] == 1
    assert result["source_proxy_sufficient"] is None
    assert result["prediction"] == "A"


def test_static_uses_same_live_original_kads_template():
    aux = ScriptedChat(["Final Selection: [0] [1] [5] [6] [7]"])
    reader = ScriptedChat(['{"answer_choice":"A"}'])
    ret, gen = docs()
    result = build("static", reader, aux).run(q(), ret, gen)
    assert aux.calls == 1
    assert result["depth_used"] == 5
    assert result["selected_sources"] == [
        "retrieved", "retrieved", "parametric_generated",
        "parametric_generated", "parametric_generated"]
    assert result["prediction"] == "A"


def test_adaptive_only_expands_budget_and_uses_one_reader_call():
    ret, gen = docs()
    insufficient = json.dumps({
        "sufficient":False,"supported_choice":None,"key_evidence":[],
        "missing_facts":["missing study condition"]})
    enough = json.dumps({
        "sufficient":True,"supported_choice":"B",
        "key_evidence":[
            {"doc_id":"r8","quote":"Source 8 says fabricated fact number 8."}],
        "missing_facts":[]})
    aux = ScriptedChat([
        insufficient, enough, "- Final Selection: [0] [1] [5] [6] [7]"])
    reader = ScriptedChat(['{"answer_choice":"B"}'])
    result = build("adaptive", reader, aux).run(q(), ret, gen)
    assert result["depth_used"] == 10
    assert result["stop_reason"] == "source_proxy_sufficient"
    assert [p["k"] for p in result["depth_path"]] == [5,10]
    assert result["source_candidates"] == ["r0", "r1", "r2", "r3", "r8"]
    assert result["source_proxy_sufficient"] is True
    assert result["reader_calls"] == 1
    assert result["auxiliary_calls"] == 3
    assert "Source 8" not in reader.requests[0][1] or (
        "Source 8" in reader.requests[0][1])


def test_fabricated_quotes_do_not_stop_retrieval():
    ret, gen = docs()
    invented = json.dumps({
        "sufficient":True,"supported_choice":"A",
        "key_evidence":[{"doc_id":"r4","quote":"an invented supporting quote"}],
        "missing_facts":[]})
    aux = ScriptedChat([
        invented, invented, invented, invented,
        "Final Selection: [0] [1] [5] [6] [7]"])
    reader = ScriptedChat(['{"answer_choice":"A"}'])
    result = build("adaptive", reader, aux).run(q(), ret, gen)
    assert result["depth_used"] == 32
    assert result["source_proxy_sufficient"] is False
    assert result["auxiliary_calls"] == 5
    assert all(x["valid_quote_count"] == 0 for x in result["depth_path"])


def test_key_passage_oversubscribed_and_bad_config():
    ret, _ = docs()
    selected = choose_retrieved_docs(ret[:10], ("r7", "r9"), max_docs=5)
    assert [d.id for d in selected] == ["r0", "r1", "r2", "r7", "r9"]
    signal = _key_documents(5, {
        "sufficient":True, "supported_choice":"A",
        "key_evidence":[{"doc_id":"r999", "quote":"fake"}],
        "missing_facts":[]}, ret[:5], q().labels)
    assert signal.proxy_sufficient is False
    with pytest.raises(ValueError):
        Config(mode="adaptive", depths=(10,5))
