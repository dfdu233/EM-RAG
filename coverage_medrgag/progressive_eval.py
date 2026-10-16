"""Gold-isolated AFTER-inference progressive Accuracy–Budget/Harm–Rescue curves.

The report never selects a policy or optimizes a hyperparameter. It only
measures whether NEW retrieval/Reader compute yields real net corrections.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from .data import benchmark
from .evaluation import _classification, paired, read_jsonl


def evaluate_progressive(predictions: str | Path, benchmark_path: str | Path,
                         dataset: str, *, bootstrap: int = 1000,
                         seed: int = 2026) -> dict:
    _, gold = benchmark(benchmark_path, dataset)
    rows = read_jsonl(predictions)
    if any(x.get("dataset") != dataset for x in rows.values()):
        raise ValueError("Prediction file has incorrect dataset")
    if len({x.get("reader") for x in rows.values()}) != 1:
        raise ValueError("Cannot mix different Readers")
    expected_rounds = set()
    budget: dict[int, dict[str, dict]] = defaultdict(dict)
    cost: dict[int, list[dict]] = defaultdict(list)
    quality: dict[int, list[dict]] = defaultdict(list)
    for qid, result in rows.items():
        if qid not in gold:
            raise KeyError(f"Unknown evaluation id {qid}")
        trajectory = result.get("prefixes", [])
        if not trajectory or trajectory[0].get("round") != 0:
            raise ValueError("Missing original KADS reference B0")
        numbers = tuple(x.get("round") for x in trajectory)
        if numbers != tuple(range(len(numbers))):
            raise ValueError(f"Non-contiguous budget path: {qid}")
        if not expected_rounds:
            expected_rounds = set(numbers)
        elif set(numbers) != expected_rounds:
            raise ValueError("Cannot compare different budget trajectories")
        if trajectory[0].get("prediction") != result.get("base_prediction"):
            raise ValueError("B0 differs from cached-reference prediction")
        for entry in trajectory:
            r = entry["round"]
            budget[r][qid] = {"prediction": entry.get("prediction")}
            cost[r].append({
                "reader": int(entry.get("reader_calls_total", 0)),
                "planner": int(entry.get("planner_calls_total", 0)),
                "retrieval": int(entry.get("retrieval_calls_total", 0)),
            })
        views = result.get("novel_views", [])
        for stage in views:
            r = stage["round"]
            if r not in expected_rounds:
                raise ValueError("Unexpected source-view round")
            quality[r].append(stage)
    result_by_budget: dict[str, dict] = {}
    transitions: dict[str, dict] = {}
    first = budget[0]
    for round_no in sorted(budget):
        classification = _classification(budget[round_no], gold)
        result_by_budget[str(round_no)] = {
            **classification,
            "mean_reader_calls": mean(x["reader"] for x in cost[round_no]),
            "mean_planner_calls": mean(x["planner"] for x in cost[round_no]),
            "mean_real_retrieval_calls": mean(x["retrieval"] for x in cost[round_no]),
            "mean_extra_reader_calls_vs_B0": mean(x["reader"]-1 for x in cost[round_no]),
        }
        if round_no:
            direct = paired(
                first, budget[round_no], gold, bootstrap=bootstrap,
                seed=seed + round_no)
            direct["net_rescue_minus_harm"] = direct["rescue"] - direct["harm"]
            this_quality = quality[round_no]
            direct["n_new_source_views"] = sum(
                item["fresh_source_count"] > 0 for item in this_quality)
            direct["n_selected_new_source_passages"] = sum(
                item["fresh_source_count"] for item in this_quality)
            direct["n_selected_outside_original_top32"] = sum(
                item["fresh_outside_original_top32"] for item in this_quality)
            direct["n_failed_plans"] = sum(
                item["query"] is None for item in this_quality)
            transitions[f"B0_to_B{round_no}"] = direct
    incremental = {}
    for before, after in zip(sorted(budget), sorted(budget)[1:]):
        delta = paired(budget[before], budget[after], gold,
                       bootstrap=bootstrap, seed=seed + after + 101)
        delta["net_rescue_minus_harm"] = delta["rescue"] - delta["harm"]
        incremental[f"B{before}_to_B{after}"] = delta
    return {
        "dataset": dataset,
        "n": len(rows),
        "reader": next(iter(rows.values())).get("reader"),
        "by_budget": result_by_budget,
        "transitions": transitions,
        "incremental": incremental,
        "no_gold_used_in_inference": True,
        "warning": (
            "Source-view votes are NOT probabilistically independent and "
            "can agree on a wrong answer. More source queries may increase "
            "both Harm and Rescue. A positive curve requires paired held-out "
            "tests with accurate total compute accounting. Original "
            "MedRGAG KGCC/KADS cache generation costs were excluded; "
            "the current experiment does NOT directly scale corpus V."),
    }
