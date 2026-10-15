import json
import pytest

from coverage_medrgag.data import (
    MedRGAGCaches, benchmark, reader_prompt, templates, original_candidates,
)
from coverage_medrgag.models import parse_json, parse_answer, parse_selection


def fixture(tmp_path):
    bench = tmp_path / "benchmark.json"
    bench.write_text(json.dumps({"bioasq": {
        "q1": {"question": "Fictional question?",
               "options": {"A": "Yes", "B": "No"}, "answer": "B"}}}), encoding="utf-8")
    retrieval = tmp_path / "retrieved.json"
    retrieval.write_text(json.dumps({"q1": {
        "snippets": [{"id": f"real{i}", "contents": f"Retrieved fact {i}."}
                     for i in range(32)], "scores": list(range(32))
    }}), encoding="utf-8")
    generation = tmp_path / "generated.json"
    generation.write_text(json.dumps({"q1": {
        "snippets": [{"id": f"kgcc{i}", "contents": f"Generated fact {i}."}
                     for i in range(5)]}}), encoding="utf-8")
    selections = tmp_path / "snippets_final_select.json"
    selections.write_text(json.dumps({
        "q1": {"final_list": [0, 5, 8, 2, 6]}}), encoding="utf-8")
    template = tmp_path / "template.py"
    template.write_text(
        'rag_template = "Q:\\n{question}\\nDocs:\\n{documents}\\n'
        'Format {{\\"answer_choice\\": \\"A\\"}}"\n'
        'llm_selection_prompt = "Pick 5 out of 10. {question}\\n{documents}"\n',
        encoding="utf-8")
    return bench, retrieval, generation, selections, template


def test_exact_upstream_cache_selection_order_and_prompt(tmp_path):
    ben, ret, gen, sel, tp = fixture(tmp_path)
    questions, gold = benchmark(ben, "bioasq")
    q = questions["q1"]
    assert not hasattr(q, "answer")
    assert gold["q1"] == "B"
    db = MedRGAGCaches(ret, gen, sel)
    retrieved, generated = db.get("q1")
    assert retrieved[0].id == "r0"
    assert retrieved[0].original_id == "real0"
    assert generated[0].origin == "parametric_generated"
    assert db.official_selection("q1") == (0, 5, 8, 2, 6)
    candidates = original_candidates(retrieved, generated)
    assert [c.id for c in candidates] == [
        "r0", "r1", "r2", "r3", "r4", "g0", "g1", "g2", "g3", "g4"]
    rag, kads = templates(tp)
    prompt = reader_prompt(q, [candidates[i] for i in db.official_selection("q1")], rag)
    assert "Document [0]: Retrieved fact 0." in prompt
    assert "Document [1]: Generated fact 0." in prompt
    assert "A: Yes\nB: No\n" in prompt
    assert '"answer_choice": "A"' in prompt


def test_parser_recovers_answer_field_only_and_legacy_kads_output():
    labels = ("A", "B")
    assert parse_answer('{"reasoning":"unescaped\nline","answer_choice":"B"}', labels) == "B"
    assert parse_answer('{"explanation":"A"}', labels) is None
    assert parse_answer('{"answer_choice":"C"}', labels) is None
    assert parse_selection("- Final Selection: [0] [2] [5] [9] [8]", 10) == (0,2,5,9,8)
    assert parse_selection('{"selected_indices":[0,2,5]}', 10) == (0,2,5)
    for bad in ("Final Selection: [0] [0]", "Final Selection: [10]",
                "Final Selection: [0] [A]", '{"selected_indices":[true,2]}'):
        with pytest.raises(ValueError):
            parse_selection(bad, 10)
