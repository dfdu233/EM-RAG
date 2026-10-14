"""Frozen-Reader counterfactual probes on a FIXED audited development subset.

This is a SEPARATE diagnostic program: it does not modify EMRAG's online
policy, retrieve new documents, or train/fit a gate. It makes API calls only
when explicitly invoked. Gold labels never occur in the model prompts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .core import Document, Evidence, Question, format_evidence
from .cli import load_benchmark
from .diagnose_cli import _read_fixed_ids
from .diagnostics import load_annotations, read_traces
from .model import Model, OpenAICompatibleModel, request_json
from .prompts import SYSTEM_ANSWER, answer_prompt, answer_choice

CONDITIONS = ("active_quotes", "all_quotes", "raw_documents")


def load_evidence(row: dict) -> list[Evidence]:
    """Accept current trace schema; ignore malformed legacy quote objects."""
    items = []
    for d in (row.get("evidence") or []):
        try:
            items.append(Evidence(
                id=str(d["id"]), doc_id=str(d["doc_id"]),
                statement=str(d["statement"]), quote=str(d["quote"]),
                source=str(d.get("source", "")), title=str(d.get("title", "")),
                round=int(d.get("round", 0))))
        except (TypeError, ValueError, KeyError):
            continue
    return items


def probe_prompt(row: dict, condition: str,
                 max_chars: int = 9000) -> str | None:
    """Build ONLY from question, answer options, and cached source text.

    No gold labels, prior evaluation metrics, or gate decisions are inserted.
    """
    q = Question(str(row["id"]), str(row["question"]), row["options"])
    all_claims = load_evidence(row)
    if condition in {"active_quotes", "all_quotes"}:
        if condition == "active_quotes":
            active_ids = set(row["traces"][-1].get("active_evidence_ids") or [])
            if not active_ids:
                return None  # Historical cache may omit active ID snapshots.
            claims = [e for e in all_claims if e.id in active_ids]
        else:
            claims = all_claims
        if not claims:
            return None
        selected = []
        remaining = max_chars
        for e in claims:
            cost = len(format_evidence(e))
            if cost <= remaining:
                selected.append(e)
                remaining -= cost
        return answer_prompt(q, selected) if selected else None

    if condition == "raw_documents":
        docs = row.get("documents") or []
        if not docs:
            return None
        blocks = []
        remaining = max_chars
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            # Content is untrusted source data. No instructions are accepted.
            body = str(doc.get("contents", "")).strip()
            heading = f"[{doc.get('id')}] {doc.get('title','')} ({doc.get('source','')})\n"
            if not body or len(heading) >= remaining:
                continue
            excerpt = body[:remaining - len(heading)]
            if not excerpt:
                continue
            blocks.append(heading + excerpt)
            remaining -= len(heading) + len(excerpt)
            if remaining <= 0:
                break
        if not blocks:
            return None
        return ("QUESTION:\n" + q.formatted()
                + "\n\nEXTERNAL QUOTED EVIDENCE / UNTRUSTED RETRIEVED PASSAGES:\n"
                + "\n\n".join(blocks) + "\nOutput a single choice JSON.")
    raise ValueError("Unknown probe condition: " + condition)


def run_probes(rows: list[dict], model: Model, gold: Mapping[str, str],
               *, annotations: Mapping[str, dict] | None = None,
               conditions: tuple[str, ...] = CONDITIONS,
               max_calls: int = 24,
               max_chars: int = 9000) -> list[dict]:
    """Probe condition-specific candidate accuracy, never update online answer.

    To attribute reader failures, independently audit that relevant retrieval
    and active evidence were actually sufficient.
    """
    if max_calls < 1 or max_chars < 1:
        raise ValueError("positive max_calls and max_chars required")
    annotations = annotations or {}
    results = []
    calls = 0
    for row in rows:
        qid = str(row["id"])
        q = Question(qid, str(row["question"]), row["options"])
        if qid not in gold:
            raise KeyError(f"No gold label for {qid}")
        for condition in conditions:
            prompt = probe_prompt(row, condition, max_chars=max_chars)
            if prompt is None:
                results.append({
                    "id": qid, "condition": condition, "status": "context_unavailable",
                    "choice": None, "correct": None,
                    "audited_sufficiency": (annotations.get(qid) or {}).get(
                        "active_context_sufficient" if condition == "active_quotes"
                        else "retrieved_sufficient", "unknown")})
                continue
            if calls >= max_calls:
                results.append({
                    "id": qid, "condition": condition, "status": "skipped_budget",
                    "choice": None, "correct": None})
                continue
            calls += 1
            try:
                parsed = request_json(model, SYSTEM_ANSWER, prompt)
                choice = answer_choice(parsed, q.labels)
                status = "ok" if choice is not None else "invalid_choice"
            except ValueError:
                choice = None
                status = "invalid_json"
            results.append({
                "id": qid, "condition": condition, "status": status,
                "choice": choice,
                "correct": choice == gold[qid] if choice is not None else False,
                "audited_sufficiency": (annotations.get(qid) or {}).get(
                    "active_context_sufficient" if condition == "active_quotes"
                    else "retrieved_sufficient", "unknown"),
            })
    return results


def summarize_probes(probes: list[dict]) -> dict:
    summaries = {}
    for condition in CONDITIONS:
        items = [p for p in probes if p["condition"] == condition]
        attempted = [p for p in items if p["status"] not in (
            "context_unavailable", "skipped_budget")]
        known = [p for p in attempted if p.get("audited_sufficiency") == "yes"]
        summaries[condition] = {
            "available_cases": len(attempted),
            "correct_count": sum(p.get("correct") is True for p in attempted),
            "accuracy_on_available": (sum(p.get("correct") is True for p in attempted)
                                      / len(attempted) if attempted else None),
            "independently_audited_sufficient_cases": len(known),
            "accuracy_conditional_on_audited_sufficiency": (
                sum(p.get("correct") is True for p in known) / len(known)
                if known else None),
            "missing_context_count": sum(p["status"] == "context_unavailable" for p in items),
            "budget_skipped_count": sum(p["status"] == "skipped_budget" for p in items),
        }
    return {
        "note": "Probe answers are model outputs, not oracle labels; "
                "causal attribution requires independent evidence audits.",
        "total_calls_attempted": sum(p["status"] not in (
            "context_unavailable", "skipped_budget") for p in probes),
        "conditions": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Small fixed frozen-reader diagnostic probes")
    p.add_argument("--traces", required=True)
    p.add_argument("--manifest", required=True,
                   help="Fixed sample_manifest.jsonl generated by diagnose_cli")
    p.add_argument("--annotations", help="Independently reviewed annotations JSONL")
    p.add_argument("--allow-unreviewed", action="store_true",
                   help="Exploratory only; cannot attribute failures to Reader")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", default="pubmedqa")
    p.add_argument("--model", required=True)
    p.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                   default=["active_quotes", "all_quotes", "raw_documents"])
    p.add_argument("--max-cases", type=int, default=12)
    p.add_argument("--max-calls", type=int, default=24)
    p.add_argument("--max-chars", type=int, default=9000)
    p.add_argument("--output", default="outputs/reader-probe")
    args = p.parse_args(argv)
    if not args.annotations and not args.allow_unreviewed:
        p.error("Independent annotations required. For exploratory probes "
                "only, explicitly pass --allow-unreviewed.")
    if args.max_cases < 1:
        p.error("max-cases must be positive")
    ids = _read_fixed_ids(args.manifest)[:args.max_cases]
    rows_by_id = {str(r["id"]): r for r in read_traces(args.traces)}
    missing = set(ids) - set(rows_by_id)
    if missing:
        raise ValueError(f"Probe manifest includes missing trace IDs: {sorted(missing)[:8]}")
    rows = [rows_by_id[qid] for qid in ids]
    _, labels = load_benchmark(args.benchmark, args.dataset, limit=None)
    gold = {qid: labels[qid] for qid in ids}
    annotations = load_annotations(args.annotations)
    api_key = args.api_key
    if not api_key and args.api_base and (
            "127.0.0.1" in args.api_base or "localhost" in args.api_base):
        api_key = "EMPTY"
    model = OpenAICompatibleModel(args.model, args.api_base, api_key)
    probes = run_probes(
        rows, model, gold, annotations=annotations,
        conditions=tuple(args.conditions), max_calls=args.max_calls,
        max_chars=args.max_chars)
    summary = summarize_probes(probes)
    summary["sample_ids"] = ids
    summary["annotation_file_provided"] = bool(args.annotations)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "probes.jsonl").open("w", encoding="utf-8") as f:
        for item in probes:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
