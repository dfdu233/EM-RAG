"""Executable independent RAG mechanism study using original MedRGAG tasks.

Example: python -m rag_mechanisms.cli plan | census | run | report | audit
No import from EM-RAG. No gold answer enters the Reader inference process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from .data import (load_benchmark, load_ranked_cache, load_generated_cache,
                   hash_file, manifest, read_manifest, read_reference_predictions,
                   read_jsonl, write_jsonl)
from .design import build_conditions, census
from .reader import FrozenReader, PromptBuilder, load_medrgag_rag_template, model_parameters


def _json_file(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", required=True, help="Original MedRGAG benchmark.json")
    parser.add_argument("--dataset", required=True,
                        choices=["medqa", "medmcqa", "mmlu", "pubmedqa", "bioasq"])


def command_plan(args) -> None:
    q, _ = load_benchmark(args.benchmark, args.dataset)
    cache = load_ranked_cache(args.retrieval_cache)
    eligible = set(cache) & set(q)
    if args.generated_cache:
        gen = load_generated_cache(args.generated_cache)
        eligible &= {qid for qid, docs in gen.items() if len(docs) >= 5}
    result = manifest(q, dataset=args.dataset, seed=args.seed,
                      count=args.count, eligible_ids=eligible)
    if not result:
        raise ValueError("No paired MedRGAG question/retrieval-cache entries")
    write_jsonl(args.output, result)
    print(json.dumps({"manifest": args.output, "selected_n": len(result),
                      "dataset": args.dataset, "seed": args.seed,
                      "gold_used_for_sampling": False}, indent=2))


def command_census(args) -> None:
    q, _ = load_benchmark(args.benchmark, args.dataset)
    cache = load_ranked_cache(args.retrieval_cache)
    ids = read_manifest(args.manifest, args.dataset)
    _validate_ids(ids, q)
    value = census(q, cache, ids, tuple(args.depths))
    value.update({"dataset": args.dataset,
                  "retrieval_cache_sha256": hash_file(args.retrieval_cache)})
    _json_file(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _validate_ids(ids, questions):
    missing = set(ids) - set(questions)
    if missing:
        raise ValueError(f"Manifest includes IDs outside benchmark: {sorted(missing)[:5]}")


def _parameter_record(args, reader: FrozenReader, path: str) -> dict:
    result = {
        "schema_version": 1, "purpose": "mechanism_analysis_not_algorithm_optimization",
        "dataset": args.dataset, "benchmark_sha256": hash_file(args.benchmark),
        "retrieval_cache_sha256": hash_file(args.retrieval_cache),
        "generated_cache_sha256": (
            hash_file(args.generated_cache) if args.generated_cache else None),
        "manifest_sha256": hash_file(args.manifest),
        "template_sha256": hash_file(args.template),
        "depths": sorted(set(args.depths)),
        "generation_conditions": args.include_generation,
        "negative_controls": args.include_negative_control,
        "temperature": args.temperature,
        "reader": model_parameters(reader),
        "max_prompt_chars": args.max_prompt_chars,
        "max_input_tokens": args.max_input_tokens,
        "tokenizer": args.tokenizer,
        "stored_predictions": path,
        "gold_labels_passed_to_inference": False,
        "study_retrieval_strategy": "original_precomputed_ranked_MedRGAG",
        "prompt_recipe": "MedRGAG src/template.py::rag_template",
        "warnings": [
            "Prompt length varies with depth unless paired same-size controls are used.",
            "Generated documents are parametric knowledge, not external facts.",
            "Precomputed ranked snippets do not represent iterative retrieval.",
            "A baseline prediction with no documents is same-prompt closed-book, not necessarily the original published Direct Response prompt.",
        ],
    }
    return result


def command_run(args) -> None:
    # Gold is loaded by loader for validation but discarded; it is never sent
    # to Reader or saved in this response-only record.
    questions, _ = load_benchmark(args.benchmark, args.dataset)
    ids = read_manifest(args.manifest, args.dataset)
    _validate_ids(ids, questions)
    if len(ids) > args.max_examples:
        raise ValueError("Manifest exceeds --max-examples safety cap; raise explicitly")
    retrieval = load_ranked_cache(args.retrieval_cache)
    missing = set(ids) - set(retrieval)
    if missing:
        raise ValueError(f"Retrieval cache missing selected IDs: {sorted(missing)[:8]}")
    if args.include_generation and not args.generated_cache:
        raise ValueError("Generation study requires MedRGAG --generated-cache")
    generated = load_generated_cache(args.generated_cache) if args.generated_cache else {}
    if args.include_generation:
        unpaired = [qid for qid in ids if len(generated.get(qid, [])) < 5]
        if unpaired:
            raise ValueError(f"Generated cache lacks 5 documents: {unpaired[:8]}; "
                             "refuse unpaired comparisons")
    template = load_medrgag_rag_template(args.template)
    builder = PromptBuilder(
        template, tokenizer_path=args.tokenizer,
        max_input_tokens=args.max_input_tokens,
        max_prompt_chars=args.max_prompt_chars)
    reader = FrozenReader(
        model=args.model, base_url=args.base_url, api_key=args.api_key,
        temperature=args.temperature, max_output_tokens=args.max_output_tokens,
        seed=args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prediction_file = out / "predictions.jsonl"
    meta_path = out / "run_config.json"
    meta = _parameter_record(args, reader, str(prediction_file))
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved != meta:
            raise ValueError("Existing output has DIFFERENT experiment configuration. "
                             "Use a new --output directory.")
    _json_file(meta_path, meta)
    existing: dict[tuple[str, str], dict] = {}
    if prediction_file.exists():
        for row in read_jsonl(prediction_file):
            key = (str(row["id"]), str(row["condition"]))
            if key in existing:
                raise ValueError(f"Duplicate resumed prediction: {key}")
            existing[key] = row
    if args.max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    new_calls = 0
    with prediction_file.open("a", encoding="utf-8") as f:
        for i, qid in enumerate(ids, start=1):
            q = questions[qid]
            conditions = build_conditions(
                qid, retrieval[qid], generated.get(qid), retrieval,
                depths=tuple(args.depths),
                include_generation=args.include_generation,
                include_negative_control=args.include_negative_control,
                seed=args.seed)
            for c in conditions:
                key = (qid, c.name)
                if key in existing:
                    continue
                if new_calls >= args.max_calls:
                    print(f"STOP: reached new-call cap {args.max_calls}. Resume with "
                          "--max-calls on the SAME output after checking resources.")
                    return
                row = {
                    "dataset": args.dataset, "reader": args.model, "id": qid,
                    "condition": c.name, "family": c.family,
                    "source_type": c.source_type, "expected_k": c.expected_k,
                    "document_count": len(c.passages),
                    "doc_ids": list(d.id for d in c.passages),
                    "document_characters": sum(len(d.text) for d in c.passages),
                    "note": c.note,
                }
                try:
                    prompt = builder.build(q, c.passages)
                except ValueError as exc:
                    row.update({"status": str(exc), "prediction": None,
                                "prompt_chars": None, "prompt_tokens": None,
                                "logical_calls": 0})
                else:
                    row.update({
                        "prompt_sha256": prompt.sha256,
                        "prompt_chars": prompt.prompt_chars,
                        "prompt_tokens": prompt.prompt_tokens,
                    })
                    # Backend errors fail loudly, so downstream analysis never
                    # mistakes an API outage for model inaccuracy.
                    answer = reader.answer(prompt, q.choice_labels)
                    new_calls += 1
                    row.update({
                        "status": ("ok" if answer["prediction"] is not None
                                   else "invalid_answer"),
                        "prediction": answer["prediction"],
                        "raw_output": answer["raw_output"],
                        "prompt_tokens_billed": answer["prompt_tokens_billed"],
                        "completion_tokens_billed": answer["completion_tokens_billed"],
                        "logical_calls": 1,
                    })
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{i}/{len(ids)}] {args.dataset}/{qid}: "
                  f"new_reader_calls={new_calls}", flush=True)
    print(f"Saved {prediction_file}. No ground-truth labels were used in inference.")


def command_report(args) -> None:
    from .analysis import report
    questions, gold = load_benchmark(args.benchmark, args.dataset)
    ref = {}
    for item in args.reference or []:
        if "=" not in item:
            raise ValueError("--reference requires NAME=prediction.json")
        name, path = item.split("=", 1)
        if not name or name in ref:
            raise ValueError("Reference names must be unique and non-empty")
        ref[name] = read_reference_predictions(path)
    value = report(args.predictions, args.dataset, gold, ref,
                   seed=args.seed, bootstrap_samples=args.bootstrap_samples)
    _json_file(args.output, value)
    print(json.dumps(value["summary"], ensure_ascii=False, indent=2))


def command_audit(args) -> None:
    from .audit import prepare_casebook, annotation_template
    questions, gold = load_benchmark(args.benchmark, args.dataset)
    cache = load_ranked_cache(args.retrieval_cache)
    value, text = prepare_casebook(
        args.predictions, args.dataset, questions, gold, cache,
        seed=args.seed, per_category=args.per_category)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "audit_manifest.jsonl", value)
    write_jsonl(out / "annotations_template.jsonl", annotation_template(value))
    (out / "casebook.md").write_text(text, encoding="utf-8")
    print(json.dumps({"audit_cases": len(value), "casebook": str(out / "casebook.md"),
                      "warning": "Manual source sufficiency labels required"},
                     ensure_ascii=False, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Domain-level MedRGAG RAG scaling experiments")
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="Freeze small sampled IDs without using gold")
    _shared(plan)
    plan.add_argument("--retrieval-cache", required=True)
    plan.add_argument("--generated-cache")
    plan.add_argument("--count", type=int, default=100)
    plan.add_argument("--seed", type=int, default=2026)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=command_plan)

    cen = sub.add_parser("census", help="Offline cache coverage, length, source diversity")
    _shared(cen)
    cen.add_argument("--retrieval-cache", required=True)
    cen.add_argument("--manifest", required=True)
    cen.add_argument("--depths", nargs="+", type=int, default=[0, 1, 3, 5, 10, 20, 32])
    cen.add_argument("--output", required=True)
    cen.set_defaults(func=command_census)

    run = sub.add_parser("run", help="Frozen same-prompt budget-depth experiments")
    _shared(run)
    run.add_argument("--retrieval-cache", required=True)
    run.add_argument("--generated-cache")
    run.add_argument("--template", required=True,
                     help="Original MedRGAG src/template.py")
    run.add_argument("--manifest", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    run.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    run.add_argument("--temperature", type=float, default=0.1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-output-tokens", type=int, default=512)
    run.add_argument("--tokenizer", help="HF/local model tokenizer for exact token cap")
    run.add_argument("--max-input-tokens", type=int, default=24576)
    run.add_argument("--max-prompt-chars", type=int, default=100000)
    run.add_argument("--depths", nargs="+", type=int, default=[0, 1, 3, 5, 10, 20, 32])
    run.add_argument("--include-generation", action="store_true")
    run.add_argument("--include-negative-control", action="store_true")
    run.add_argument("--max-examples", type=int, default=100)
    run.add_argument("--max-calls", type=int, default=750,
                     help="Strict cap of NEW model calls per invocation")
    run.set_defaults(func=command_run)

    rep = sub.add_parser("report", help="Paired offline harm/rescue and scaling report")
    _shared(rep)
    rep.add_argument("--predictions", required=True)
    rep.add_argument("--reference", action="append", help="NAME=prediction.json")
    rep.add_argument("--seed", type=int, default=2026)
    rep.add_argument("--bootstrap-samples", type=int, default=1000)
    rep.add_argument("--output", required=True)
    rep.set_defaults(func=command_report)

    audit = sub.add_parser("audit", help="Blind-ready raw-evidence casebook for humans")
    _shared(audit)
    audit.add_argument("--predictions", required=True)
    audit.add_argument("--retrieval-cache", required=True)
    audit.add_argument("--seed", type=int, default=2026)
    audit.add_argument("--per-category", type=int, default=8)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=command_audit)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
