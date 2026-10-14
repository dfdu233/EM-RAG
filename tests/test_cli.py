import json
import pytest
from emrag.cli import load_benchmark


def test_medrgag_format_and_gold_is_separate(tmp_path):
    data = {"medqa": {
        "a": {"id": "a", "question": "Test question?", "options": {
            "A": "yes", "B": "no"}, "answer": "A"},
        "b": {"id": "b", "question": "Second question?", "options": {
            "A": "red", "B": "blue"}, "answer": "blue"},
    }}
    f = tmp_path / "benchmark.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    qs, gold = load_benchmark(str(f), "medqa")
    assert [x.id for x in qs] == ["a", "b"]
    assert gold == {"a": "A", "b": "B"}
    assert not hasattr(qs[0], "answer")


def test_invalid_gold_fails(tmp_path):
    f = tmp_path / "bench.json"
    f.write_text(json.dumps({"medqa": {"x": {"question": "Q?",
                   "options": {"A": "foo", "B": "bar"}, "answer": "C"}}}))
    with pytest.raises(ValueError):
        load_benchmark(str(f), "medqa")
