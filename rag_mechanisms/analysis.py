"""Paired RAG evidence-scaling analysis: NOT policy development.

Answer keys are loaded here, after Reader predictions were written.
Never call generation, retrieval or a model from this module.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
import random
from typing import Any

from .data import read_jsonl


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot summarize empty bootstrap")
    values = sorted(values)
    index = (len(values) - 1) * p
    lower, upper = int(index), min(len(values) - 1, int(index) + 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def paired_interval(deltas: list[int], *, seed: int, n_boot: int) -> dict:
    """Resample QUESTIONS, not individual rounds; avoids pseudo-replication."""
    if not deltas:
        return {"n": 0, "mean_delta": None, "ci95": None}
    n = len(deltas)
    effect = sum(deltas) / n
    if n_boot < 1:
        return {"n": n, "mean_delta": effect, "ci95": None}
    rng = random.Random(seed)
    distribution = []
    for _ in range(n_boot):
        distribution.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    return {"n": n, "mean_delta": effect,
            "ci95": [_percentile(distribution, 0.025),
                     _percentile(distribution, 0.975)],
            "ci_method": "paired_question_cluster_bootstrap",
            "bootstrap_replications": n_boot}


def _is_eligible(row: dict) -> bool:
    """Malformed answers count incorrect; prompt-budget rejects are unavailable."""
    return row.get("status") in ("ok", "invalid_answer")


def _correct(row: dict, gold: str) -> bool:
    return row.get("prediction") == gold


def _compare(left: dict[str, dict], right: dict[str, dict], gold: dict[str, str],
             *, seed: int, boots: int) -> dict:
    common = sorted(set(left) & set(right) & set(gold))
    if not common:
        return {"n": 0, "warning": "No paired question IDs with valid contexts"}
    changes = []
    rescue = harm = changed = 0
    for qid in common:
        old = _correct(left[qid], gold[qid])
        new = _correct(right[qid], gold[qid])
        changes.append(int(new) - int(old))
        rescue += int(not old and new)
        harm += int(old and not new)
        changed += int(left[qid].get("prediction") != right[qid].get("prediction"))
    stats = paired_interval(changes, seed=seed, n_boot=boots)
    stats.update({
        "left_accuracy": sum(_correct(left[qid], gold[qid]) for qid in common) / len(common),
        "right_accuracy": sum(_correct(right[qid], gold[qid]) for qid in common) / len(common),
        "rescue_count": rescue, "harm_count": harm,
        "rescue_fraction": rescue / len(common), "harm_fraction": harm / len(common),
        "changed_answer_count": changed,
        "identity": "accuracy_delta = rescue_fraction - harm_fraction",
        "claim_limit": "Paired observational intervention, not proof of document truth.",
    })
    assert abs(stats["mean_delta"] - (rescue - harm) / len(common)) < 1e-10
    return stats


def report(path: str, dataset: str, gold: dict[str, str],
           references: dict[str, dict[str, str | None]] | None = None,
           *, seed: int = 2026, bootstrap_samples: int = 1000) -> dict:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("No model predictions found")
    by_condition: dict[str, dict[str, dict]] = defaultdict(dict)
    all_ids: set[str] = set()
    invalid_records = []
    for row in rows:
        if row.get("dataset") != dataset:
            raise ValueError("Dataset mixing detected")
        qid = str(row["id"])
        name = str(row["condition"])
        if qid not in gold:
            raise KeyError(f"Gold key absent: {dataset}/{qid}")
        if qid in by_condition[name]:
            raise ValueError(f"Duplicate recorded condition {qid}/{name}")
        all_ids.add(qid)
        if _is_eligible(row):
            by_condition[name][qid] = row
        else:
            invalid_records.append({
                "id": qid, "condition": name, "status": row.get("status"),
            })
    if "retrieved_k0" not in by_condition:
        raise ValueError("Run requires same-prompt closed-book retrieved_k0")
    zero = by_condition["retrieved_k0"]
    condition_stats = {}
    for name, predictions in sorted(by_condition.items()):
        sample = list(predictions.items())
        condition_stats[name] = {
            "n": len(sample),
            "accuracy": mean(_correct(row, gold[qid]) for qid, row in sample),
            "invalid_answers_in_denominator": sum(
                row.get("status") == "invalid_answer" for _, row in sample),
            "mean_prompt_chars": mean(row.get("prompt_chars", 0) for _, row in sample),
            "mean_context_chars": mean(row.get("document_characters", 0)
                                      for _, row in sample),
            "mean_documents": mean(row.get("document_count", 0)
                                   for _, row in sample),
            "total_billed_prompt_tokens": sum(
                row.get("prompt_tokens_billed") or 0 for _, row in sample),
            "total_billed_completion_tokens": sum(
                row.get("completion_tokens_billed") or 0 for _, row in sample),
            "source_type": sample[0][1]["source_type"],
        }
    comparisons = {}
    for name, predictions in sorted(by_condition.items()):
        if name != "retrieved_k0":
            comparisons["retrieved_k0 -> " + name] = _compare(
                zero, predictions, gold, seed=seed, boots=bootstrap_samples)

    depth_names = sorted(
        (x for x in by_condition if x.startswith("retrieved_k")
         and x[len("retrieved_k"):].isdigit()),
        key=lambda x: int(x[len("retrieved_k"):]))
    for prev_name, next_name in zip(depth_names, depth_names[1:]):
        comparisons[prev_name + " -> " + next_name] = _compare(
            by_condition[prev_name], by_condition[next_name], gold,
            seed=seed, boots=bootstrap_samples)
    controls = (
        ("prefix3_own_tail2", "prefix3_other_question_tail2"),
        ("prefix3_own_tail5", "prefix3_other_question_tail5"),
        ("retrieved_k5", "generated_k5"),
        ("retrieved_k5", "retrieved3_generated2"),
    )
    for left, right in controls:
        if left in by_condition and right in by_condition:
            comparisons[left + " -> " + right] = _compare(
                by_condition[left], by_condition[right], gold,
                seed=seed, boots=bootstrap_samples)

    # Oracle best-prefix is EVALUATION ONLY; never use for deployed routing.
    depth_ids = sorted(
        set.intersection(*(set(by_condition[n]) for n in depth_names))
        if depth_names else set())
    depth_sequences = {
        qid: [by_condition[name][qid]["prediction"] for name in depth_names]
        for qid in depth_ids
    }
    curving = {
        "n_fully_paired_across_depths": len(depth_sequences),
        "depths": depth_names,
        "fraction_ever_regresses": (
            mean(any(a == gold[qid] and b != gold[qid] for a, b in zip(seq, seq[1:]))
                 for qid, seq in depth_sequences.items())
            if depth_sequences else None),
        "fraction_ever_recovers": (
            mean(any(a != gold[qid] and b == gold[qid] for a, b in zip(seq, seq[1:]))
                 for qid, seq in depth_sequences.items())
            if depth_sequences else None),
        "oracle_best_depth_accuracy": (
            mean(any(p == gold[qid] for p in seq)
                 for qid, seq in depth_sequences.items())
            if depth_sequences else None),
        "oracle_for_diagnosis_only": True,
    }
    reference_stats = {}
    for name, pred in (references or {}).items():
        common = sorted(set(all_ids) & set(pred) & set(gold))
        if not common:
            reference_stats[name] = {"n": 0}
            continue
        reference_stats[name] = {
            "n": len(common),
            "accuracy": sum(pred[qid] == gold[qid] for qid in common) / len(common),
            "comparison_caveat": "These are prior method predictions, possibly from "
                                 "different prompts/retrieval budgets. Descriptive only.",
        }
    summary = {
        "dataset": dataset, "questions_with_any_record": len(all_ids),
        "conditions": {name: {
            "n": x["n"], "accuracy": x["accuracy"],
            "mean_context_chars": x["mean_context_chars"],
        } for name, x in condition_stats.items()},
        "depth_behaviour": curving,
        "unavailable_or_over_budget": len(invalid_records),
        "reference_methods": reference_stats,
        "no_emrag_modules_compared": True,
        "no_manual_evidence_sufficiency_claim": True,
        "development_repeated_testing_warning": (
            "If these cases have already guided prior experiments, use them as "
            "exploratory development data; do not claim fresh held-out significance."),
    }
    return {
        "summary": summary,
        "per_condition": condition_stats,
        "paired_comparisons": comparisons,
        "depth_behaviour": curving,
        "missing_or_budget_rejected": invalid_records,
        "reference_methods": reference_stats,
        "bootstrap_seed": seed,
    }
