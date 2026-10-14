"""Gold-label-isolated anytime metrics for risk, rescue, coverage and budget."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping
from .core import RunResult


def summarize(results: list[RunResult], gold: Mapping[str, str],
              max_rounds: int | None = None) -> dict:
    """Tracks last available answer if a system stopped before budget t.

    Harm_t = P(correct at t-1, wrong at t), Rescue_t = P(wrong, correct).
    For forced-choice (or counting abstentions as incorrect):
    Acc_t - Acc_(t-1) = Rescue_t - Harm_t.
    """
    if not results:
        return {"n": 0, "rounds": []}
    ids = [str(r.id) for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation IDs")
    missing = set(ids) - set(str(k) for k in gold)
    if missing:
        raise KeyError(f"Missing gold labels for IDs: {sorted(missing)[:8]}")
    gold_map = {str(k): str(v).strip().upper() for k, v in gold.items()}
    for r in results:
        if gold_map[r.id] not in {k.upper() for k in r.options}:
            raise ValueError(f"Gold label is not an option for {r.id}")

    steps = max_rounds if max_rounds is not None else max(len(r.traces) - 1 for r in results)
    metrics: list[dict] = []
    previous: list[bool] | None = None
    n = len(results)

    for step in range(steps + 1):
        decisions = [next((t for t in reversed(r.traces) if t.round <= step),
                          r.traces[0]) for r in results]
        correct = [t.selected_choice == gold_map[r.id] for r, t in zip(results, decisions)]
        covered = [t.selected_choice is not None for t in decisions]
        correct_count = sum(correct)
        covered_count = sum(covered)
        point = {
            "round": step, "n": n,
            "accuracy": correct_count / n,
            "coverage": covered_count / n,
            "selective_accuracy": correct_count / covered_count if covered_count else None,
            "verified_coverage": sum(t.status == "verified" for t in decisions) / n,
            "mean_model_calls": sum(t.model_calls for t in decisions) / n,
        }
        if previous is None:
            point.update({"harm_count": 0, "rescue_count": 0,
                          "harm_rate": 0.0, "rescue_rate": 0.0,
                          "harm_conditional": None, "rescue_conditional": None,
                          "accuracy_delta": 0.0})
        else:
            harm = sum(p and not c for p, c in zip(previous, correct))
            rescue = sum(not p and c for p, c in zip(previous, correct))
            prev_correct = sum(previous)
            point.update({
                "harm_count": harm, "rescue_count": rescue,
                "harm_rate": harm / n, "rescue_rate": rescue / n,
                "harm_conditional": harm / prev_correct if prev_correct else None,
                "rescue_conditional": rescue / (n - prev_correct) if n != prev_correct else None,
                "accuracy_delta": (rescue - harm) / n,
            })
            assert abs((sum(correct) - sum(previous)) / n
                       - point["accuracy_delta"]) < 1e-12
        metrics.append(point)
        previous = correct

    return {
        "n": n, "rounds": metrics,
        "final_accuracy": metrics[-1]["accuracy"],
        "final_coverage": metrics[-1]["coverage"],
        "final_verified_coverage": metrics[-1]["verified_coverage"],
        "oracle_best_prefix_accuracy": sum(
            any(t.selected_choice == gold_map[r.id] for t in r.traces)
            for r in results) / n,
        "oracle_is_evaluation_only": True,
    }


def summarize_actions(results: list[RunResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        for step in r.traces:
            counts[step.action] += 1
    return dict(counts)
