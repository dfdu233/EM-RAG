"""Cross-method paired panel for original MedRGAG paper baselines.

Consumes EXISTING per-question prediction.json outputs; no RAG code and
no inference are run. Methods may differ in prompts, Reader versions and
token budgets, so effects are descriptive rather than identified causal
effects. Compare SAME Reader + dataset + IDs wherever possible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .analysis import paired_interval
from .data import (load_benchmark, read_manifest, read_reference_predictions)

PAPER_BASELINES = (
    "DirectResponse", "VanillaRAG", "MedRAG", "iMedRAG",
    "GENREAD", "MedGENIE", "GRG", "CGAP", "MedRGAG",
)


def panel(gold: dict[str, str], selected_ids: list[str],
          methods: dict[str, dict[str, str | None]],
          *, control: str = "DirectResponse",
          seed: int = 2026, bootstrap_samples: int = 1000) -> dict:
    if control not in methods:
        raise ValueError("Specify original DirectResponse predictions as control")
    if not selected_ids:
        raise ValueError("Empty evaluation manifest")
    ref = methods[control]
    report: dict = {}
    for name, pred in methods.items():
        ids = [qid for qid in selected_ids if qid in pred and qid in ref]
        if not ids:
            report[name] = {"paired_n": 0, "missing": len(selected_ids)}
            continue
        old = [int(ref[qid] == gold[qid]) for qid in ids]
        new = [int(pred[qid] == gold[qid]) for qid in ids]
        changes = [b - a for a, b in zip(old, new)]
        n = len(ids)
        detailed = paired_interval(changes, seed=seed, n_boot=bootstrap_samples)
        detailed.update({
            "paired_n": n,
            "missing": len(selected_ids) - n,
            "control_accuracy_on_pairs": sum(old) / n,
            "method_accuracy_on_pairs": sum(new) / n,
            "rescue_count": sum(a == 0 and b == 1 for a, b in zip(old, new)),
            "harm_count": sum(a == 1 and b == 0 for a, b in zip(old, new)),
            "changed_prediction_count": sum(pred[qid] != ref[qid] for qid in ids),
            "coverage": sum(pred[qid] is not None for qid in ids) / n,
            "macro_recall": (
                mean(
                    sum(pred[qid] == c for qid in ids if gold[qid] == c)
                    / sum(gold[qid] == c for qid in ids)
                    for c in set(gold[qid] for qid in ids)
                ) if len({gold[qid] for qid in ids}) >= 2 else None
            ),
        })
        report[name] = detailed
    # Multi-method commonality, not a single-method artifact:
    affected = {}
    for qid in selected_ids:
        if qid not in gold or qid not in ref:
            continue
        observed = {
            name: pred[qid] for name, pred in methods.items()
            if name != control and qid in pred
        }
        if not observed:
            continue
        effects = {
            name: ("harm" if ref[qid] == gold[qid] and choice != gold[qid]
                   else "rescue" if ref[qid] != gold[qid] and choice == gold[qid]
                   else "unchanged_correctness")
            for name, choice in observed.items()
        }
        affected[qid] = {
            "control_correct": ref[qid] == gold[qid],
            "method_harm_count": sum(v == "harm" for v in effects.values()),
            "method_rescue_count": sum(v == "rescue" for v in effects.values()),
            "methods_with_harm": [k for k, v in effects.items() if v == "harm"],
            "methods_with_rescue": [k for k, v in effects.items() if v == "rescue"],
            "n_methods_observed": len(effects),
        }
    return {
        "n_manifest": len(selected_ids),
        "methods": report,
        "cross_method_transitions": affected,
        "n_cases_harmed_by_multiple_methods": sum(
            x["method_harm_count"] >= 2 for x in affected.values()),
        "n_cases_rescued_by_multiple_methods": sum(
            x["method_rescue_count"] >= 2 for x in affected.values()),
        "caution": "Common cross-method errors may reflect shared source, model "
                   "priors, dataset label issues or prompt differences. "
                   "Do not infer cause without controlled evidence interventions. "
                   "Some methods are not RAG retrieval methods (GAG baselines).",
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Cross-method MedRGAG baseline outcome panel")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--reader-label", required=True)
    p.add_argument("--reference", action="append", required=True,
                   help="NAME=/path/to/original/prediction.json; DirectResponse mandatory")
    p.add_argument("--control", default="DirectResponse")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    _, gold = load_benchmark(args.benchmark, args.dataset)
    ids = read_manifest(args.manifest, args.dataset)
    methods = {}
    for item in args.reference:
        if "=" not in item:
            raise ValueError("--reference NAME=FILE required")
        name, path = item.split("=", 1)
        if name in methods or not name.strip():
            raise ValueError(f"Duplicate or empty method name {name}")
        methods[name] = read_reference_predictions(path)
    result = panel(gold, ids, methods, control=args.control,
                   seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    result.update({
        "dataset": args.dataset, "reader": args.reader_label,
        "paper_baseline_names": PAPER_BASELINES,
        "note": "Original baseline prediction files imported, NOT re-run.",
    })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps({
        "reader": args.reader_label, "dataset": args.dataset,
        "by_method": {
            k: {x: v.get(x) for x in (
                "paired_n", "method_accuracy_on_pairs",
                "harm_count", "rescue_count", "mean_delta")}
            for k, v in result["methods"].items()},
        "multi_method_harm_cases": result["n_cases_harmed_by_multiple_methods"],
        "multi_method_rescue_cases": result["n_cases_rescued_by_multiple_methods"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
