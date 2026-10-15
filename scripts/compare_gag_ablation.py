"""Four-arm, ONE-ROUND GAG ablation: offline metrics only, no model calls.

Reads EM-RAG traces and gold labels for evaluation AFTER all runs. Enforces
identical IDs, round-0 answers, and first-round retrieved document IDs.
Any candidate disagreement between gate variants is reported as a warning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emrag.cli import load_benchmark


def read_run(folder: str) -> dict[str, dict]:
    path = Path(folder) / "traces.jsonl"
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            identifier = str(item["id"])
            if identifier in records:
                raise ValueError(f"Duplicate question ID {identifier} in {folder}")
            if len(item.get("traces", [])) != 2:
                raise ValueError("Use --max-rounds 1 and do not early-stop "
                                 f"when producing the ablation traces: {identifier}")
            records[identifier] = item
    if not records:
        raise ValueError(f"Empty run: {folder}")
    return records


def summarize_variant(rows: dict[str, dict], gold: dict[str, str]) -> dict:
    n = len(rows)
    counts = {
        "n": n, "baseline_correct": 0,
        "reader_candidate_correct": 0, "final_correct": 0,
        "reader_rescue_opportunities": 0, "reader_harm_opportunities": 0,
        "rescues_accepted": 0, "harmful_changes_accepted": 0,
        "correct_candidates_blocked": 0, "harmful_candidates_blocked": 0,
        "changed_final_answers": 0, "logical_model_calls": 0,
    }
    for qid, row in rows.items():
        step = row["traces"][1]
        a0 = row["traces"][0].get("selected_choice")
        proposed = step.get("proposed_choice")
        emitted = step.get("selected_choice")
        gold_choice = gold[qid]
        counts["baseline_correct"] += int(a0 == gold_choice)
        counts["reader_candidate_correct"] += int(proposed == gold_choice)
        counts["final_correct"] += int(emitted == gold_choice)
        counts["logical_model_calls"] += int(row["model_calls"])
        counts["changed_final_answers"] += int(emitted != a0)
        if proposed is not None and proposed != a0:
            if a0 != gold_choice and proposed == gold_choice:
                counts["reader_rescue_opportunities"] += 1
                if emitted == proposed:
                    counts["rescues_accepted"] += 1
                elif emitted == a0:
                    counts["correct_candidates_blocked"] += 1
            if a0 == gold_choice and proposed != gold_choice:
                counts["reader_harm_opportunities"] += 1
                if emitted == proposed:
                    counts["harmful_changes_accepted"] += 1
                elif emitted == a0:
                    counts["harmful_candidates_blocked"] += 1
    counts["baseline_accuracy"] = counts["baseline_correct"] / n
    counts["reader_candidate_accuracy"] = counts["reader_candidate_correct"] / n
    counts["final_accuracy"] = counts["final_correct"] / n
    counts["mean_logical_model_calls"] = counts["logical_model_calls"] / n
    return counts


def compare(variants: dict[str, dict[str, dict]],
            gold: dict[str, str]) -> dict:
    expected = ("without_gag_gate", "with_gag_gate",
                "without_gag_latest", "with_gag_latest")
    if set(variants) != set(expected):
        raise ValueError(f"Need exactly the four variants: {expected}")
    baseline = variants[expected[0]]
    ids = list(baseline)
    for name in expected[1:]:
        run = variants[name]
        if set(run) != set(ids):
            raise ValueError(f"Unpaired question IDs in {name}")
        for qid in ids:
            original = baseline[qid]["traces"]
            other = run[qid]["traces"]
            if other[0].get("selected_choice") != original[0].get("selected_choice"):
                raise ValueError(f"Baseline changed across arms: {name}/{qid}")
            if other[1].get("retrieved_doc_ids") != original[1].get("retrieved_doc_ids"):
                raise ValueError(f"Retrieved documents differ: {name}/{qid}. "
                                 "Fix retrieval before attributing gains to GAG.")
    if set(ids) - set(gold):
        raise ValueError("Gold labels missing for some run IDs")
    stats = {name: summarize_variant(variants[name], gold) for name in expected}
    disagree = {
        group: sum(
            variants[f"{prefix}_gate"][qid]["traces"][1].get("proposed_choice") !=
            variants[f"{prefix}_latest"][qid]["traces"][1].get("proposed_choice")
            for qid in ids
        )
        for group, prefix in (("no_gag", "without_gag"), ("gag", "with_gag"))
    }
    return {
        "questions": len(ids),
        "variants": stats,
        "diagnostic_deltas": {
            "gag_candidate_accuracy_vs_no_gag_under_gate": (
                stats["with_gag_gate"]["reader_candidate_accuracy"]
                - stats["without_gag_gate"]["reader_candidate_accuracy"]),
            "gag_final_accuracy_vs_no_gag_under_gate": (
                stats["with_gag_gate"]["final_accuracy"]
                - stats["without_gag_gate"]["final_accuracy"]),
            "gag_final_accuracy_vs_no_gag_without_gate": (
                stats["with_gag_latest"]["final_accuracy"]
                - stats["without_gag_latest"]["final_accuracy"]),
        },
        "candidate_disagreement_between_gate_and_latest_repeats": disagree,
        "interpretation_warning": (
            "Golden answers used for OFFLINE development diagnosis only. "
            "Even with equal documents, independent LLM calls may be non-deterministic; "
            "if proposal disagreement > 0 the arms are not perfectly paired. "
            "The same GAG text may be based on prior model knowledge, not external truth. "
            "Historical KGCC generation cost is NOT in logical_model_calls."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Four-arm one-round cached KGCC ablation")
    for name in ("without_gag_gate", "with_gag_gate",
                 "without_gag_latest", "with_gag_latest"):
        p.add_argument("--" + name.replace("_", "-"), required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", default="pubmedqa")
    p.add_argument("--output", default="outputs/gag_ablation_summary.json")
    args = p.parse_args()
    variants = {
        name: read_run(getattr(args, name))
        for name in ("without_gag_gate", "with_gag_gate",
                     "without_gag_latest", "with_gag_latest")
    }
    _, gold = load_benchmark(args.benchmark, args.dataset, limit=None)
    report = compare(variants, gold)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
