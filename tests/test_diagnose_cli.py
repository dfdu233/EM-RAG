import json
from pathlib import Path

from emrag.diagnose_cli import main, _read_fixed_ids
from emrag.diagnostics import hydrate_medrgag_first_turn


def _trace(qid, baseline, proposal, after):
    return {
        "id": qid, "question": "A fictional question?", "options": {
            "A": "alpha", "B": "beta"},
        "baseline": baseline, "selected": after, "status": "unverified",
        "traces": [
            {"round": 0, "selected_choice": baseline,
             "proposed_choice": baseline},
            {"round": 1, "proposed_choice": proposal,
             "selected_choice": after,
             "action": "preserved", "scores": {
                 "A": {"score": 0, "sufficient": False, "support_ids": []},
                 "B": {"score": 0, "sufficient": False, "support_ids": []}
             }}
        ], "evidence": []
    }


def test_offline_cli_no_model_calls_and_stable_manifest(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(_trace("1", "B", "A", "B")) + "\n"
        + json.dumps(_trace("2", "A", "B", "A")) + "\n", encoding="utf-8")
    benchmark = tmp_path / "bench.json"
    benchmark.write_text(json.dumps({"toy": {
        "1": {"question": "A fictional question?",
              "options": {"A": "alpha", "B": "beta"}, "answer": "A"},
        "2": {"question": "A fictional question?",
              "options": {"A": "alpha", "B": "beta"}, "answer": "A"}
    }}), encoding="utf-8")
    out = tmp_path / "diagnosis"
    assert main(["--traces", str(traces), "--benchmark", str(benchmark),
                 "--dataset", "toy", "--output", str(out),
                 "--seed", "100"]) == 0
    stats = json.loads((out / "diagnostic_summary.json").read_text())
    assert stats["no_models_called"] is True
    assert stats["proposal_transition_counts"]["rescue_blocked"] == 1
    assert stats["proposal_transition_counts"]["harm_blocked"] == 1
    assert stats["cases_with_raw_documents"] == 0
    ids = _read_fixed_ids(out / "sample_manifest.jsonl")
    assert set(ids) == {"1", "2"}
    out2 = tmp_path / "second"
    assert main(["--traces", str(traces), "--benchmark", str(benchmark),
                 "--dataset", "toy", "--output", str(out2),
                 "--fixed-manifest", str(out / "sample_manifest.jsonl")]) == 0
    assert _read_fixed_ids(out2 / "sample_manifest.jsonl") == ids


def test_initial_cache_hydration_scope_is_explicit(tmp_path):
    rows = [_trace("1", "B", "A", "B"), _trace("2", "A", "B", "A")]
    cache = tmp_path / "snippets.json"
    cache.write_text(json.dumps({
        "1": {"snippets": [{"id": "foo", "contents": "A useful source quote."}],
              "scores": [1.0]}
    }))
    result = hydrate_medrgag_first_turn(rows, cache)
    assert result["cases_hydrated"] == 1
    assert rows[0]["documents"][0]["contents"] == "A useful source quote."
    assert rows[0]["document_snapshot_scope"] == "cached_first_turn_only"
    assert "documents" not in rows[1]
