"""GPU-free command line diagnosis for cached EM-RAG JSONL runs.

This does NOT run EM-RAG again or change gate thresholds. All gold labels are
used solely for retrospective analysis and deterministic development sampling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

from .cli import load_benchmark
from .diagnostics import (
    annotation_template, audit_casebook, load_annotations, read_traces,
    report_cases, stratified_sample,
)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for item in data:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_fixed_ids(path: str | Path) -> list[str]:
    """Existing sample manifest may be JSONL or JSON list of IDs/objects."""
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fp:
            entries = [json.loads(s) for s in fp if s.strip()]
    else:
        with path.open(encoding="utf-8") as fp:
            entries = json.load(fp)
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain a list")
    ids = [str(item["id"]) if isinstance(item, dict) else str(item) for item in entries]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("No IDs or duplicate IDs in fixed manifest")
    return ids


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose retrieved evidence, Reader proposals and EM-RAG gate "
                    "from cached traces, without running a single model call")
    p.add_argument("--traces", required=True, help="Existing traces.jsonl")
    p.add_argument("--benchmark", required=True, help="MedRGAG benchmark.json or QA JSONL")
    p.add_argument("--dataset", default="pubmedqa")
    p.add_argument("--output", default="outputs/diagnosis")
    p.add_argument("--per-stratum", type=int, default=6)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--annotations", help="Existing manual audit annotations JSONL")
    p.add_argument("--fixed-manifest", help="Re-use these SAME IDs for another analysis")
    p.add_argument("--original-margin", type=int,
                   help="Read-only: if run_config.json absent, specify ORIGINAL gate margin")
    p.add_argument("--purpose", choices=("development", "heldout_reporting"),
                   default="development")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_traces(args.traces)
    # Loading all gold labels is post-hoc only. Gold is NEVER given to the engine.
    _, all_gold = load_benchmark(args.benchmark, args.dataset, limit=None)
    gold = {str(row["id"]): all_gold[str(row["id"])] for row in rows}
    traces_dir = Path(args.traces).parent
    configuration = traces_dir / "run_config.json"
    run_config: dict = {}
    if configuration.exists():
        with configuration.open(encoding="utf-8") as fp:
            run_config = json.load(fp)
    recorded_margin = run_config.get("engine_config", {}).get("min_score_margin")
    if (recorded_margin is not None and args.original_margin is not None
            and recorded_margin != args.original_margin):
        raise ValueError(
            "The original margin is known from run_config.json; refusing "
            "a changed gate threshold during retrospective diagnosis")
    margin = (recorded_margin if recorded_margin is not None else
              args.original_margin if args.original_margin is not None else 2)
    if not isinstance(margin, int) or margin < 0:
        raise ValueError("Original margin must be a nonnegative integer")

    labels = load_annotations(args.annotations)
    unknown = set(labels) - {str(row["id"]) for row in rows}
    if unknown:
        raise ValueError(f"Annotations contain non-trace IDs: {sorted(unknown)[:10]}")
    stats, cases = report_cases(rows, gold, margin=margin, annotations=labels)
    stats["original_gate_margin"] = margin
    stats["recorded_gate_margin_available"] = recorded_margin is not None
    stats["purpose"] = args.purpose
    stats["no_models_called"] = True
    stats["source_traces"] = str(Path(args.traces).resolve())

    if args.fixed_manifest:
        fixed_ids = _read_fixed_ids(args.fixed_manifest)
        index = {case["id"]: case for case in cases}
        absent = set(fixed_ids) - set(index)
        if absent:
            raise ValueError(f"Fixed manifest includes IDs missing from traces: {sorted(absent)[:8]}")
        selected = [index[qid] for qid in fixed_ids]
    else:
        selected = stratified_sample(cases, args.per_stratum, seed=args.seed)
    stats["sample_seed"] = args.seed
    stats["sample_size"] = len(selected)
    stats["selected_sample_ids"] = [case["id"] for case in selected]
    stats["sampling_warning"] = (
        "Stratification uses offline gold and is for *development diagnosis* only. "
        "Never adapt thresholds on held-out test labels. Reuse fixed_manifest "
        "for all subsequent probe conditions."
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "diagnostic_summary.json", stats)
    _write_jsonl(out / "all_cases.jsonl", cases)
    _write_jsonl(out / "sample_manifest.jsonl", selected)
    _write_jsonl(out / "annotations_template.jsonl",
                 [annotation_template(case) for case in selected])
    with (out / "audit_casebook.md").open("w", encoding="utf-8") as fp:
        fp.write(audit_casebook(rows, selected))
    print(json.dumps({
        "n": stats["n"],
        "baseline_accuracy": stats["baseline_accuracy"],
        "final_accuracy": stats["final_accuracy"],
        "reader_candidate_oracle": stats["candidate_oracle_accuracy"],
        "proposal_transitions": stats["proposal_transition_counts"],
        "blocked_reasons": stats["blocking_reasons_multiple_per_event"],
        "casebook_n": stats["sample_size"],
        "raw_documents_available_cases": stats["cases_with_raw_documents"],
        "independent_retrieval_audits": stats["independently_annotated_retrieval_cases"],
        "outputs": str(out.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
