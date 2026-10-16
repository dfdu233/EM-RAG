"""Independent diagnosis: does nondecreasing UTILITY predict answer accuracy?

Only available when --read-each-depth was enabled during INFERENCE.
Gold labels enter ONLY this offline module, never the scorer/selector.
Bootstrap resamples questions (not repeated K stages).
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from .data import benchmark
from .evaluation import paired


def prefix_curve(path: str | Path, benchmark_path: str | Path, dataset: str,
                 *, bootstrap: int = 1000, seed: int = 2026) -> dict:
    _, gold = benchmark(benchmark_path, dataset)
    series: dict[int, dict[str, dict]] = defaultdict(dict)
    count = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != dataset:
                raise ValueError("Mixed dataset in prefix curve")
            if not row.get("read_each_depth"):
                raise ValueError(
                    "Prefix accuracy curves require a --read-each-depth run")
            qid = str(row["id"])
            if qid not in gold:
                raise KeyError(f"No gold for question {qid}")
            count += 1
            prev = -1.0
            for item in row["prefix_reader_results"]:
                k = int(item["k"])
                score = float(item["utility"])
                if score + 1e-12 < prev:
                    raise ValueError("Saved score trajectory is not monotone")
                prev = score
                if qid in series[k]:
                    raise ValueError(f"Duplicate prefix result: {qid}/{k}")
                series[k][qid] = item
    if not count or not series:
        raise ValueError("No stored prefix Reader answers")
    summary = {}
    for k, by_id in sorted(series.items()):
        ids = sorted(by_id)
        summary[str(k)] = {
            "n": len(ids),
            "correct": sum(by_id[qid]["prediction"] == gold[qid] for qid in ids),
            "accuracy": mean(by_id[qid]["prediction"] == gold[qid] for qid in ids),
            "mean_surrogate_utility": mean(by_id[qid]["utility"] for qid in ids),
            "mean_prompt_chars": mean(by_id[qid]["prompt_chars"] for qid in ids),
        }
    pairwise = {}
    steps = sorted(series)
    for before, after in zip(steps, steps[1:]):
        common = sorted(set(series[before]) & set(series[after]))
        if not common:
            continue
        old = {
            qid: {"prediction": series[before][qid]["prediction"]}
            for qid in common
        }
        new = {
            qid: {"prediction": series[after][qid]["prediction"]}
            for qid in common
        }
        pair = paired(old, new, gold, bootstrap=bootstrap, seed=seed + after)
        pair["n_scored_utility_improved"] = sum(
            series[after][qid]["utility"] >
            series[before][qid]["utility"] + 1e-12
            for qid in common)
        pair["n_score_improved_but_answer_harmed"] = sum(
            series[after][qid]["utility"] >
            series[before][qid]["utility"] + 1e-12
            and old[qid]["prediction"] == gold[qid]
            and new[qid]["prediction"] != gold[qid]
            for qid in common)
        pair["n_score_improved_and_answer_rescued"] = sum(
            series[after][qid]["utility"] >
            series[before][qid]["utility"] + 1e-12
            and old[qid]["prediction"] != gold[qid]
            and new[qid]["prediction"] == gold[qid]
            for qid in common)
        pairwise[f"{before}_to_{after}"] = pair
    return {
        "dataset": dataset,
        "examples": count,
        "by_depth": summary,
        "paired_transitions": pairwise,
        "warning": (
            "This is a fixed-SURROGATE (not correctness) anytime certificate. "
            "Source documents may be inaccurate, and pretrained similarity "
            "is not entailment. Utility increases do NOT guarantee answer "
            "correctness or calibrated confidence. Prompt and answer changes "
            "can also reflect decoding stochasticity."),
    }
