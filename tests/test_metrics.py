import pytest
from emrag.core import RunResult, RoundTrace
from emrag.metrics import summarize


def run(qid, baseline, changed):
    steps = [
        RoundTrace(round=0, selected_choice=baseline, model_calls=1),
        RoundTrace(round=1, selected_choice=changed, model_calls=4),
    ]
    return RunResult(qid, "Synthetic item", {"A": "x", "B": "y"},
                     baseline, changed, "unverified", steps, [], 4)


def test_harm_rescue_accounting_and_oracle():
    results = [run("1", "A", "B"), run("2", "A", "B"), run("3", "B", "A")]
    stats = summarize(results, {"1": "A", "2": "B", "3": "A"}, max_rounds=2)
    assert stats["rounds"][0]["accuracy"] == pytest.approx(1 / 3)
    t = stats["rounds"][1]
    assert t["harm_count"] == 1
    assert t["rescue_count"] == 2
    assert t["accuracy_delta"] == pytest.approx(1 / 3)
    assert t["accuracy"] == pytest.approx(2 / 3)
    assert stats["rounds"][2]["accuracy"] == t["accuracy"]  # early-stop carry
    assert stats["oracle_best_prefix_accuracy"] == 1.0


def test_gold_required_and_no_duplicate_ids():
    with pytest.raises(KeyError):
        summarize([run("1", "A", "B")], {})
    with pytest.raises(ValueError):
        summarize([run("1", "A", "B"), run("1", "A", "B")], {"1": "A"})
