"""Post-hoc human sufficiency labels × frozen Reader outcomes.

A human-labeled subset is intentionally targeted at interesting transitions.
DO NOT generalize its prevalence to the full benchmark without weighting.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .data import load_benchmark, read_jsonl


FIELDS = (
    "sufficient_k5", "sufficient_k20",
    "decisive_new_fact_5_to_20",
    "source_conflicts_prior", "observed_qualifier_loss",
)
CHOICES = {"yes", "no", "unknown"}


def summarize_annotations(annotations: list[dict], predictions: list[dict],
                          gold: dict[str, str], dataset: str) -> dict:
    label_by_id = {}
    for label in annotations:
        if label.get("dataset") != dataset:
            raise ValueError("Mixing annotation datasets")
        qid = str(label["id"])
        if qid in label_by_id:
            raise ValueError(f"Duplicate annotation for {qid}; adjudicate first")
        for field in FIELDS:
            if label.get(field, "unknown") not in CHOICES:
                raise ValueError(f"Invalid {field} for {qid}")
        label_by_id[qid] = label
    rows = defaultdict(dict)
    for item in predictions:
        if item.get("dataset") != dataset:
            raise ValueError("Prediction dataset mismatch")
        qid = str(item["id"])
        condition = str(item["condition"])
        if condition in rows[qid]:
            raise ValueError(f"Duplicate prediction {qid}/{condition}")
        rows[qid][condition] = item
    outcomes = []
    for qid, label in label_by_id.items():
        if qid not in rows or qid not in gold:
            raise ValueError(f"Missing prediction/gold case for annotated {qid}")
        for k in (5, 20):
            suff = label.get(f"sufficient_k{k}", "unknown")
            pred = rows[qid].get(f"retrieved_k{k}")
            if suff == "unknown" or pred is None:
                continue
            if pred.get("status") not in ("ok", "invalid_answer"):
                continue
            outcomes.append({
                "id": qid, "depth": k, "sufficient": suff,
                "correct": pred.get("prediction") == gold[qid],
            })
    stage_stats = {}
    for k in (5, 20):
        stage_stats[str(k)] = {}
        for suff in ("yes", "no"):
            matched = [x for x in outcomes if x["depth"] == k and
                       x["sufficient"] == suff]
            stage_stats[str(k)][suff] = {
                "n": len(matched),
                "reader_accuracy": (
                    sum(x["correct"] for x in matched) / len(matched)
                    if matched else None),
                "reader_failure_despite_human_sufficiency": (
                    sum(not x["correct"] for x in matched)
                    if suff == "yes" else None),
            }
    newly_sufficient = []
    for qid, label in label_by_id.items():
        if (label.get("sufficient_k5") == "no"
                and label.get("sufficient_k20") == "yes"
                and "retrieved_k5" in rows[qid] and "retrieved_k20" in rows[qid]):
            p5, p20 = rows[qid]["retrieved_k5"], rows[qid]["retrieved_k20"]
            if p5.get("status") in ("ok", "invalid_answer") and p20.get("status") in ("ok", "invalid_answer"):
                newly_sufficient.append({
                    "id": qid,
                    "correct_k5": p5.get("prediction") == gold[qid],
                    "correct_k20": p20.get("prediction") == gold[qid],
                })
    result = {
        "annotated_cases": len(label_by_id),
        "counts": {
            field: dict(Counter(
                label.get(field, "unknown") for label in label_by_id.values()))
            for field in FIELDS
        },
        "evidence_sufficiency_vs_reader_performance": stage_stats,
        "became_sufficient_between_5_and_20": {
            "n": len(newly_sufficient),
            "previously_wrong_now_correct": sum(
                not x["correct_k5"] and x["correct_k20"]
                for x in newly_sufficient),
            "still_wrong_despite_new_sufficiency": sum(
                not x["correct_k20"] for x in newly_sufficient),
        },
        "scope_note": (
            "Human evidence review is independent of Reader predictions, "
            "but transition-enriched audited cases are NOT a random population "
            "sample. Report per-stratum observations instead of extrapolating "
            "these counts to the entire MedRGAG benchmark."),
    }
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    _, gold = load_benchmark(args.benchmark, args.dataset)
    result = summarize_annotations(
        read_jsonl(args.annotations), read_jsonl(args.predictions),
        gold, args.dataset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
