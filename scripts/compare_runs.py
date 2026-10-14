#!/usr/bin/env python3
"""Compare saved EM-RAG metrics across update policies at per-round budgets."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", help="metrics.json from multiple runs")
    parser.add_argument("--csv", help="Optional CSV output path")
    args = parser.parse_args()
    datasets = []
    for name in args.metrics:
        path = Path(name)
        with path.open(encoding="utf-8") as f:
            stats = json.load(f)
        ids = []
        traces = path.with_name("traces.jsonl")
        if traces.exists():
            with traces.open(encoding="utf-8") as f:
                ids = [str(json.loads(line)["id"]) for line in f if line.strip()]
        datasets.append((path.parent.name, stats, ids))
    if len({d[1]["n"] for d in datasets}) > 1:
        raise SystemExit("Cannot compare runs with unequal numbers of questions.")
    id_lists = [d[2] for d in datasets if d[2]]
    if len(id_lists) > 1 and any(x != id_lists[0] for x in id_lists[1:]):
        raise SystemExit("Run IDs/order differ: enforce identical questions before comparing.")
    rows = []
    for label, stats, _ in datasets:
        for step in stats["rounds"]:
            rows.append({
                "run": label, "round": step["round"],
                "accuracy": step["accuracy"], "coverage": step["coverage"],
                "verified_coverage": step["verified_coverage"],
                "harm_rate": step["harm_rate"], "rescue_rate": step["rescue_rate"],
                "mean_model_calls": step["mean_model_calls"],
            })
    print("run\tround\taccuracy\tcoverage\tharm\trescue\tmean_llm_calls")
    for r in rows:
        print(f"{r['run']}\t{r['round']}\t{r['accuracy']:.4f}\t"
              f"{r['coverage']:.4f}\t{r['harm_rate']:.4f}\t"
              f"{r['rescue_rate']:.4f}\t{r['mean_model_calls']:.1f}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
