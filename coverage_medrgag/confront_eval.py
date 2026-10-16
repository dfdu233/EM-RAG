"""Offline, strictly gold-isolated evaluation of literature-fusion MedRGAG.

Maintains exactly the same frozen B=0 budget convention as PSV and reports
paired budget curves. Distinguishes format recovery, verified *substring*
citations, order sensitivity, rescued/harmed original parseable predictions.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

from .data import benchmark
from .evaluation import paired, read_jsonl
from .progressive_eval import evaluate_progressive


def evaluate_confronted(predictions: str | Path, benchmark_path: str | Path,
                       dataset: str, *, bootstrap: int = 1000,
                       seed: int = 2026) -> dict:
    result = evaluate_progressive(
        predictions, benchmark_path, dataset, bootstrap=bootstrap, seed=seed)
    rows = read_jsonl(predictions)
    _, gold = benchmark(benchmark_path, dataset)
    if {row.get("mode") for row in rows.values()} != {"evidence_confronted_medrgag"}:
        raise ValueError("Only confrontation-run outputs accepted")
    all_stages = [s for item in rows.values() for s in item["novel_views"]]
    reasons = Counter(s.get("decision") for s in all_stages)
    parseable = {
        qid for qid, item in rows.items() if item.get("base_prediction") is not None
    }
    if parseable:
        baseline = {
            qid: {"prediction": rows[qid]["base_prediction"]} for qid in parseable
        }
        latest = {
            qid: {"prediction": rows[qid]["prediction"]} for qid in parseable
        }
        clean = paired(baseline, latest, gold, seed=seed + 400, bootstrap=bootstrap)
    else:
        clean = None
    clipped = sum(bool(s.get("query_clipped")) for s in all_stages)
    source_quoted = sum(bool(s.get("source_citations")) for s in all_stages)
    switched = sum(item.get("prediction") != item.get("base_prediction")
                   for item in rows.values())
    detail = {
        "total_attempted_plans": len(all_stages),
        "query_clipped_to_corpus_bound": clipped,
        "rejected_plans": sum(s.get("query") is None for s in all_stages),
        "new_source_view_count": sum(s.get("fresh_source_count", 0) > 0
                                     for s in all_stages),
        "source_quoted_verbatim_count": source_quoted,
        "order_checked_count": sum(s.get("reversed_prediction") is not None
                                   for s in all_stages),
        "order_disagreement_count": sum(
            s.get("reversed_prediction") is not None and
            s.get("source_prediction") != s.get("reversed_prediction")
            for s in all_stages),
        "confrontation_invocations": sum(
            s.get("confront_prediction") is not None for s in all_stages),
        "decisions": dict(reasons),
        "final_answer_changed": switched,
        "baseline_unparseable": len(rows) - len(parseable),
        "baseline_field_parse_recovered": sum(
            bool(x.get("baseline_parse_recovered")) for x in rows.values()),
        "source_field_parse_recovered": sum(
            bool(s.get("source_parse_recovered")) for s in all_stages),
        "final_format_recovery_from_unparseable_baseline": sum(
            rows[qid].get("base_prediction") is None and
            rows[qid].get("prediction") is not None for qid in rows),
        "comparison_on_parseable_baseline_only": clean,
        "mean_total_reader_calls": mean(
            row.get("reader_calls", 0) for row in rows.values()),
        "mean_total_planner_calls": mean(
            row.get("auxiliary_calls", 0) for row in rows.values()),
        "mean_total_live_retrieval_calls": mean(
            row.get("retrieval_calls", 0) for row in rows.values()),
    }
    result["confrontation_diagnostics"] = detail
    result["warning"] = (
        "This is a literature-fusion baseline, not a new SOTA result. "
        "Exact quotations prove text occurrence only, NOT truth, support, "
        "or calibration; reverse-order agreement is not independent evidence. "
        "No answer-gold was available at inference. Original KGCC/KADS cache "
        "preparation was reused and excluded from inference costs. "
        "Report the real GPU+API+retrieval costs on a fully frozen split.")
    return result
