"""Gold-isolated AFTER-inference progressive Accuracy–Budget/Harm–Rescue curves.

The report never selects a policy or optimizes a hyperparameter. It only
measures whether NEW retrieval/Reader compute yields real net corrections.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean

from .data import benchmark
from .evaluation import _classification, paired, read_jsonl


def evaluate_progressive(predictions: str | Path, benchmark_path: str | Path,
                         dataset: str, *, bootstrap: int = 1000,
                         seed: int = 2026, compare: str | Path | None = None) -> dict:
    _, gold = benchmark(benchmark_path, dataset)
    rows = read_jsonl(predictions)
    if any(x.get("dataset") != dataset for x in rows.values()):
        raise ValueError("Prediction file has incorrect dataset")
    if len({x.get("reader") for x in rows.values()}) != 1:
        raise ValueError("Cannot mix different Readers")
    expected_rounds = set()
    plan_stats = Counter()
    parse_stats = Counter()
    anchor_stats = Counter()
    matched_control_results = {}
    matched_view_results = {}
    budget: dict[int, dict[str, dict]] = defaultdict(dict)
    cost: dict[int, list[dict]] = defaultdict(list)
    quality: dict[int, list[dict]] = defaultdict(list)
    for qid, result in rows.items():
        if qid not in gold:
            raise KeyError(f"Unknown evaluation id {qid}")
        trajectory = result.get("prefixes", [])
        parse_stats["baseline_" + result.get("base_parse_method", "historical_unknown")] += 1
        anchor_stats[result.get("anchor_mode", "historical_original")] += 1
        anchor_stats["sum_generated_anchor_docs"] += result.get("anchor_generated_count", 0)
        matched = result.get("matched_source_control")
        if isinstance(matched, dict) and matched.get("prediction") is not None:
            parse_stats["matched_control_" + matched.get("parse_method", "unknown")] += 1
            first_view = next(
                (view for view in result.get("novel_views", [])
                 if view.get("differs_from_matched_control") is not None
                 and view.get("prediction") is not None
                 and view.get("fresh_source_count") == 2), None)
            if first_view is not None:
                matched_control_results[qid] = {"prediction": matched["prediction"]}
                matched_view_results[qid] = {"prediction": first_view["prediction"]}
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
                "replay": bool(result.get("source_views_replayed", False)),
                "replay_queries": (
                    int(entry.get("queries_used_total", 0))
                    if result.get("source_views_replayed", False) else 0),
            })
        views = result.get("novel_views", [])
        for stage in views:
            plan_stats[stage.get("query_provenance", {}).get(
                "status", "historical_unknown")] += 1
            if stage.get("raw_reader_output") is not None:
                parse_stats["view_" + stage.get("parse_method", "historical_unknown")] += 1
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
            "source_replay_fraction": mean(
                int(x["replay"]) for x in cost[round_no]),
            "mean_replayed_source_queries_not_new_searches": mean(
                x["replay_queries"] for x in cost[round_no]),
        }
        if round_no:
            direct = paired(
                first, budget[round_no], gold, bootstrap=bootstrap,
                seed=seed + round_no)
            direct["net_rescue_minus_harm"] = direct["rescue"] - direct["harm"]
            # B0 -> Br is cumulative across ALL completed search rounds.
            # Reporting only the last turn here would undercount the actual
            # expansion and misstate the budget/coverage mechanism.
            previous_quality = [
                view for item in rows.values()
                for view in item.get("novel_views", [])
                if view["round"] <= round_no
            ]
            direct["n_new_source_views"] = sum(
                item["fresh_source_count"] > 0 for item in previous_quality)
            direct["n_selected_new_source_passages"] = sum(
                item["fresh_source_count"] for item in previous_quality)
            direct["n_selected_outside_original_top32"] = sum(
                item["fresh_outside_original_top32"] for item in previous_quality)
            direct["n_failed_plans"] = sum(
                item["query"] is None for item in previous_quality)
            transitions[f"B0_to_B{round_no}"] = direct
    incremental = {}
    for before, after in zip(sorted(budget), sorted(budget)[1:]):
        delta = paired(budget[before], budget[after], gold,
                       bootstrap=bootstrap, seed=seed + after + 101)
        delta["net_rescue_minus_harm"] = delta["rescue"] - delta["harm"]
        incremental[f"B{before}_to_B{after}"] = delta
    diagnostics = {
        "planned_query_status": dict(plan_stats),
        "answer_parse_method_counts": dict(parse_stats),
        "anchor_mode_counts": dict(anchor_stats),
        "matched_control_sample_count": len(matched_control_results),
        "matched_control_vs_first_new_view": (
            paired(matched_control_results, matched_view_results, gold,
                   seed=seed+909, bootstrap=bootstrap)
            if matched_control_results else None),
        "matched_control_caveat": (
            "Within-question fixed-five-doc source substitution is controlled "
            "for anchor and Reader settings but still can differ due to "
            "decoding noise. It is not a calibrated causal truth detector."),
    }
    cross_config = None
    if compare:
        other = read_jsonl(compare)
        if set(other) != set(rows):
            raise ValueError("Cross-run contrast must use the SAME frozen IDs")
        if {x.get("dataset") for x in other.values()} != {dataset}:
            raise ValueError("Cross-run dataset mismatch")
        if {x.get("reader") for x in other.values()} != {
                x.get("reader") for x in rows.values()}:
            raise ValueError("Cross-run Reader mismatch")
        if any(other[qid].get("baseline_prompt_sha256") !=
               rows[qid].get("baseline_prompt_sha256") for qid in rows):
            raise ValueError("Cross-run B0 prompts differ; no fair comparison")
        cross_config = {
            "alternative_path": str(compare),
            "final_budget_paired": paired(
                {qid: {"prediction": other[qid]["prediction"]} for qid in other},
                {qid: {"prediction": rows[qid]["prediction"]} for qid in rows},
                gold, seed=seed+911, bootstrap=bootstrap),
            "notice": "End-point paired accuracy is not compute-matched "
                      "when one arm uses an extra source-control Reader call.",
        }
    return {
        "dataset": dataset,
        "n": len(rows),
        "reader": next(iter(rows.values())).get("reader"),
        "by_budget": result_by_budget,
        "transitions": transitions,
        "incremental": incremental,
        "diagnostics": diagnostics,
        "cross_run": cross_config,
        "no_gold_used_in_inference": True,
        "warning": (
            "Source-view votes are NOT probabilistically independent and "
            "can agree on a wrong answer. More source queries may increase "
            "both Harm and Rescue. A positive curve requires paired held-out "
            "tests with accurate total compute accounting. Original "
            "MedRGAG KGCC/KADS cache generation costs were excluded; "
            "the current experiment does NOT directly scale corpus V. "
            "Exact-replay arms have zero ACTUAL planner/search calls in their "
            "own run, but inherit source-acquisition cost from the original "
            "live experiment. Do not compare replay costs to live budgets."),
    }
