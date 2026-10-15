"""Offline paired analysis of LIVE static-vs-adaptive acquisition.

Works on rag_mechanisms/iterative.py output. Gold is read only after the
retrieval-and-reader experiment is complete; no trained gate is involved.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .data import load_benchmark, read_jsonl
from .analysis import paired_interval


def report(rows: list[dict], gold: dict[str, str],
           *, seed: int = 2026, boots: int = 1000) -> dict:
    if not rows:
        raise ValueError("No live experiment records")
    grouped = defaultdict(dict)
    ids = set()
    for item in rows:
        qid = str(item["id"])
        if qid in ids:
            raise ValueError(f"Duplicate question id {qid}")
        ids.add(qid)
        if qid not in gold:
            raise KeyError(f"Unknown gold id {qid}")
        for step in item["steps"]:
            key = (str(step["condition"]), int(step["round"]))
            if key in grouped[qid]:
                raise ValueError(f"Duplicate arm/round for {qid}/{key}")
            grouped[qid][key] = step
    pairs = {}
    all_rounds = sorted({
        step["round"] for item in rows for step in item["steps"]
    })
    for rnd in all_rounds:
        group = []
        for qid, cases in grouped.items():
            static = cases.get(("static", rnd))
            adaptive = cases.get(("adaptive", rnd))
            if static is None or adaptive is None:
                continue
            if static.get("status") not in ("ok", "invalid_answer"):
                continue
            if adaptive.get("status") not in ("ok", "invalid_answer"):
                continue
            group.append((qid, static, adaptive))
        if not group:
            continue
        changes = [
            int(adaptive.get("prediction") == gold[qid])
            - int(static.get("prediction") == gold[qid])
            for qid, static, adaptive in group
        ]
        paired = paired_interval(changes, seed=seed + rnd, n_boot=boots)
        rescue = sum(static.get("prediction") != gold[qid]
                     and adaptive.get("prediction") == gold[qid]
                     for qid, static, adaptive in group)
        harm = sum(static.get("prediction") == gold[qid]
                   and adaptive.get("prediction") != gold[qid]
                   for qid, static, adaptive in group)
        paired.update({
            "round": rnd,
            "static_accuracy": sum(
                static.get("prediction") == gold[qid] for qid, static, _ in group
            ) / len(group),
            "adaptive_accuracy": sum(
                adaptive.get("prediction") == gold[qid] for qid, _, adaptive in group
            ) / len(group),
            "adaptive_extra_rescue": rescue,
            "adaptive_extra_harm": harm,
            "mean_static_docs": sum(
                static.get("document_count", 0) for _, static, _ in group
            ) / len(group),
            "mean_adaptive_docs": sum(
                adaptive.get("document_count", 0) for _, _, adaptive in group
            ) / len(group),
            "mean_static_prompt_chars": sum(
                static.get("prompt_chars", 0) for _, static, _ in group
            ) / len(group),
            "mean_adaptive_prompt_chars": sum(
                adaptive.get("prompt_chars", 0) for _, _, adaptive in group
            ) / len(group),
        })
        pairs[str(rnd)] = paired
    return {
        "questions": len(ids),
        "rounds": pairs,
        "warning": (
            "Static and adaptive use matched MAX retrieval-document quotas, "
            "not necessarily identical unique documents or model calls. "
            "Planner uses an extra inference per adaptive round after the first. "
            "This is a mechanism feasibility test, not policy SOTA evidence."),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterative", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    _, gold = load_benchmark(args.benchmark, args.dataset)
    data = report(
        read_jsonl(args.iterative), gold,
        seed=args.seed, boots=args.bootstrap_samples)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
