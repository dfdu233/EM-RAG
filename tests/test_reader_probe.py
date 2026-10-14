import json

from emrag.reader_probe import probe_prompt, run_probes, summarize_probes
from emrag.model import ScriptedModel


def fixture():
    return {
        "id": "caseA", "question": "FICTIONAL: Which drug is indicated?",
        "options": {"A": "Drug X", "B": "Drug Q"},
        "evidence": [
            {"id": "e-1", "doc_id": "d-1", "statement": "Drug X indicated",
             "quote": "Drug X is indicated.", "source": "fixture", "title": "demo",
             "round": 1}
        ],
        "documents": [
            {"id": "d-1", "contents": "Drug X is indicated.",
             "title": "demo", "source": "fixture"}
        ],
        "traces": [
            {"round": 0, "selected_choice": "B", "proposed_choice": "B"},
            {"round": 1, "active_evidence_ids": ["e-1"],
             "selected_choice": "B", "proposed_choice": "A"},
        ],
    }


def test_probe_conditions_have_no_gold_or_gate_label_in_prompt():
    row = fixture()
    for condition in ("active_quotes", "all_quotes", "raw_documents"):
        text = probe_prompt(row, condition)
        assert "Which drug is indicated?" in text
        assert "Drug X is indicated." in text
        assert "gold" not in text.lower()
        assert "selected_choice" not in text.lower()


def test_probe_runs_fixed_context_only_without_retrieval():
    row = fixture()
    model = ScriptedModel([
        json.dumps({"choice": "A"}), json.dumps({"choice": "B"}),
        json.dumps({"choice": "A"}),
    ])
    annotation = {
        "caseA": {"retrieved_sufficient": "yes",
                  "active_context_sufficient": "yes"}}
    res = run_probes([row], model, {"caseA": "A"},
                     annotations=annotation, max_calls=3)
    assert model.calls == 3
    assert [r["correct"] for r in res] == [True, False, True]
    summary = summarize_probes(res)
    assert summary["total_calls_attempted"] == 3
    assert summary["conditions"]["active_quotes"]["independently_audited_sufficient_cases"] == 1
    assert summary["paired_interventions"]["active_quotes_to_all_quotes"]["second_breaks_first"] == 1
    assert summary["paired_interventions"]["active_quotes_to_raw_documents"]["paired_accuracy_delta"] == 0


def test_old_cache_does_not_pretend_documents_exist():
    row = fixture()
    row["documents"] = []
    row["traces"][-1].pop("active_evidence_ids")
    model = ScriptedModel([json.dumps({"choice": "B"})])
    res = run_probes([row], model, {"caseA": "A"}, max_calls=1)
    assert res[0]["status"] == "context_unavailable"
    assert res[1]["status"] == "ok"
    assert res[2]["status"] == "context_unavailable"
    assert model.calls == 1


def test_probe_hard_cap_keeps_conditions_paired():
    row = fixture()
    m = ScriptedModel([])
    res = run_probes([row], m, {"caseA": "A"}, max_calls=1)
    assert m.calls == 0
    assert [x["status"] for x in res] == ["skipped_budget"] * 3
    single = ScriptedModel([json.dumps({"choice": "A"})])
    one = run_probes([row], single, {"caseA": "A"}, max_calls=1,
                     conditions=("active_quotes",))
    assert single.calls == 1
    assert one[0]["status"] == "ok"
