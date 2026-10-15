import json
from coverage_medrgag.evaluation import (
    _classification, _proxy_decomposition, paired,
    _independent_sufficiency, read_original_predictions
)


def test_paired_accuracy_and_suv_proxy_are_separate(tmp_path):
    gold = {"1":"A", "2":"B", "3":"B", "4":"A"}
    old = {
        "1":{"prediction":"A"}, "2":{"prediction":"A"},
        "3":{"prediction":"B"}, "4":{"prediction":"A"}
    }
    new = {
        "1":{"prediction":"B", "source_proxy_sufficient":True},
        "2":{"prediction":"B", "source_proxy_sufficient":False},
        "3":{"prediction":"B", "source_proxy_sufficient":True},
        "4":{"prediction":"A", "source_proxy_sufficient":False},
    }
    p = paired(old, new, gold, bootstrap=0)
    assert p["harm"] == 1 and p["rescue"] == 1
    assert p["mean_accuracy_delta"] == 0
    summary = _classification(new, gold)
    assert summary["accuracy"] == .75
    assert summary["macro_f1"] is not None
    proxy = _proxy_decomposition(new, gold)
    assert proxy["not_true_S_U_V"] is True
    assert proxy["S_proxy"] == .5
    assert proxy["U_proxy_given_judge_sufficient"] == .5
    assert proxy["V_proxy_given_judge_insufficient"] == 1.0
    original = tmp_path / "original.json"
    original.write_text(json.dumps([
        {"id":"1","model_answer":"A","answer":"B"},
        {"id":"2","model_answer":"B","answer":"A"},
    ]))
    assert read_original_predictions(original)["1"]["prediction"] == "A"


def test_independent_annotations_require_human_review(tmp_path):
    gold = {"1":"A", "2":"B"}
    rows = {"1":{"prediction":"A"}, "2":{"prediction":"A"}}
    p = tmp_path / "audits.jsonl"
    p.write_text(
        '{"id":"1","sufficient":true,"reviewer":"independent_expert"}\n'
        '{"id":"2","sufficient":false,"reviewer":"independent_expert"}\n')
    result = _independent_sufficiency(p, rows, gold)
    assert result["S_reviewed"] == .5
    assert result["U_reviewed"] == 1
    assert result["V_reviewed"] == 0
