"""Run original MedRGAG vs fixed and coverage-adaptive KADS on its 5 benchmarks.

Inference NEVER sees gold. Separate offline evaluation imports benchmark
labels only AFTER outputs have been written.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .data import (DATASETS, MedRGAGCaches, benchmark, fingerprint,
                   templates)
from .models import IdentityRanker, MedCPTReranker, OpenAIChat
from .pipeline import Config, CoverageAdaptiveMedRGAG


def _load_reader_tokenizer(path: str | None):
    if path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers for MedRGAG tokenizer parity") from exc
    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def _run(args) -> int:
    root = Path(args.medrgag_root)
    files = {
        "benchmark": Path(args.benchmark or root / "benchmark.json"),
        "retrieval": Path(args.retrieval_cache or root /
                          "outputs" / args.dataset / "docs" / "retrieval" / "snippets.json"),
        "generation": Path(args.generated_cache or root /
                           "outputs" / args.dataset / "docs" / "generate" / "snippets.json"),
        "templates": Path(args.template or root / "src" / "template.py"),
    }
    if args.mode == "reference":
        files["selection"] = Path(args.selection_cache or root /
                                  "outputs" / args.dataset / "docs" /
                                  "generate" / "snippets_final_select.json")
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"MedRGAG {name} file missing: {path}")
    questions, _ = benchmark(files["benchmark"], args.dataset)
    if args.ids_file:
        chosen = []
        with Path(args.ids_file).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("dataset", args.dataset) == args.dataset:
                        chosen.append(str(item["id"]))
        if not chosen or len(chosen) != len(set(chosen)):
            raise ValueError("Sample manifest is empty or has duplicate IDs")
        if set(chosen) - set(questions):
            raise KeyError(f"Unknown benchmark ID(s): {set(chosen)-set(questions)}")
    else:
        identifiers = list(questions)
        if args.offset < 0 or args.limit < 1:
            raise ValueError("limit>=1 and offset>=0 required")
        chosen = identifiers[args.offset: args.offset + args.limit]
    if not chosen:
        raise ValueError("No questions selected")
    cache = MedRGAGCaches(
        files["retrieval"], files["generation"], files.get("selection"))
    cfg = Config(
        mode=args.mode, depths=tuple(args.depths),
        probe_excerpt_chars=args.probe_excerpt_chars,
        selector_excerpt_chars=args.selector_excerpt_chars,
        max_reader_prompt_chars=args.max_reader_prompt_chars,
        context_tokens=args.context_tokens,
        seed=args.seed)
    rag_template, kads_template = templates(files["templates"])
    public_args = {
        k: v for k, v in vars(args).items()
        if k not in {"api_key", "aux_api_key", "func"}
    }
    signature = {
        "algorithm": "Coverage-Adaptive MedRGAG research prototype",
        "reference_git": "https://github.com/ll0ruc/MedRGAG",
        "config": asdict(cfg),
        "args": public_args,
        "source_sha256": {k: fingerprint(path) for k, path in files.items()},
        "ids": chosen,
        "gold_available_to_pipeline": False,
        "source_proxy_is_truth": False,
        "precomputed_KGCC_KADS_cost_included": False,
        "note": (
            "reference reuses original KGCC/KADS caches; static reruns original "
            "KADS template live; adaptive changes only retrieval-prefix budget. "
            "No answer Gate. Source-only sufficiency can be wrong. "
            "Use --reader-tokenizer for exact MedRGAG truncation."
        ),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "run_config.json"
    traces = out / "predictions.jsonl"
    if meta.exists():
        prev = json.loads(meta.read_text(encoding="utf-8"))
        if prev != signature:
            raise ValueError("Refusing to mix different models, data, or budgets "
                             "in an existing output directory")
    elif traces.exists():
        raise ValueError("Prediction traces exist without run_config.json")
    else:
        meta.write_text(json.dumps(signature, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    completed: set[str] = set()
    if traces.exists():
        with traces.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if str(row["id"]) in completed:
                        raise ValueError(f"Duplicate recorded example: {row['id']}")
                    completed.add(str(row["id"]))
        if not completed <= set(chosen):
            raise ValueError("Trace contains IDs outside the frozen manifest")
    tokenizer = _load_reader_tokenizer(args.reader_tokenizer)
    reader = OpenAIChat(
        args.reader_model, base_url=args.api_base, api_key=args.api_key)
    aux = None
    if args.mode != "reference":
        if not args.aux_model:
            raise ValueError("--aux-model required for static/adaptive modes")
        aux = OpenAIChat(
            args.aux_model,
            base_url=args.aux_api_base or args.api_base,
            api_key=args.aux_api_key or args.api_key)
    ranker = (IdentityRanker() if args.no_rerank else
              MedCPTReranker(args.ranker_model, device=args.ranker_device))
    engine = CoverageAdaptiveMedRGAG(
        reader=reader, auxiliary=aux, ranker=ranker,
        rag_template=rag_template, kads_template=kads_template,
        config=cfg, tokenizer=tokenizer)
    new = 0
    if args.max_new_examples < 1:
        raise ValueError("max-new-examples must be positive")
    with traces.open("a", encoding="utf-8") as f:
        for idx, qid in enumerate(chosen):
            if qid in completed:
                continue
            if new >= args.max_new_examples:
                print("Reached explicit max-new-examples cap; resume with identical "
                      "run_config after resource review.", flush=True)
                break
            sources, kgcc = cache.get(qid)
            original = cache.official_selection(qid) if args.mode == "reference" else None
            row = engine.run(
                questions[qid], sources, kgcc,
                original_selection=original)
            row.update({"dataset": args.dataset,
                        "reader": args.reader_model,
                        "source_cache_question_id": qid,
                        "candidate_retrieved_count": len(sources),
                        "candidate_generated_count": len(kgcc)})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            new += 1
            print(f"[{idx+1}/{len(chosen)}] {args.dataset}/{qid} "
                  f"mode={args.mode} K={row['depth_used']} "
                  f"source_proxy={row['source_proxy_sufficient']} "
                  f"reader={row['prediction']} aux_calls={row['auxiliary_calls']}",
                  flush=True)
    print(f"Saved: {traces}; new cases: {new}; "
          "source sufficiency is a PROXY, not a ground-truth label")
    return 0


def _evaluate(args) -> int:
    from .evaluation import evaluate
    result = evaluate(
        args.predictions, args.benchmark, args.dataset,
        compare=args.compare,
        original_prediction=args.original_prediction,
        annotation_file=args.annotations,
        bootstrap=args.bootstrap, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedRGAG KGCC + KADS with optional coverage-adaptive retrieval K")
    sub = p.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--medrgag-root", default="../MedRGAG")
    run.add_argument("--benchmark")
    run.add_argument("--retrieval-cache")
    run.add_argument("--generated-cache")
    run.add_argument("--selection-cache")
    run.add_argument("--template")
    run.add_argument("--mode", choices=("reference", "static", "adaptive"),
                     default="adaptive")
    run.add_argument("--depths", nargs="+", type=int, default=[5, 10, 20, 32])
    run.add_argument("--output", required=True)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--limit", type=int, default=20)
    run.add_argument("--ids-file", help="Frozen dataset+id JSONL manifest")
    run.add_argument("--max-new-examples", type=int, default=100,
                     help="Per invocation work cap; resume by repeating same command")
    run.add_argument("--reader-model", required=True)
    run.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    run.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    run.add_argument("--aux-model")
    run.add_argument("--aux-api-base")
    run.add_argument("--aux-api-key", default=os.environ.get("AUX_OPENAI_API_KEY"))
    run.add_argument("--ranker-model", default="ncbi/MedCPT-Cross-Encoder")
    run.add_argument("--ranker-device")
    run.add_argument("--no-rerank", action="store_true",
                     help="EXPLICIT ABLATION; cannot claim original MedRGAG parity")
    run.add_argument("--reader-tokenizer",
                     help="HF/local tokenizer for original MedRGAG context-token truncation")
    run.add_argument("--context-tokens", type=int, default=128000)
    run.add_argument("--max-reader-prompt-chars", type=int, default=150000)
    run.add_argument("--probe-excerpt-chars", type=int, default=1400)
    run.add_argument("--selector-excerpt-chars", type=int, default=2500)
    run.add_argument("--seed", type=int, default=42)
    run.set_defaults(func=_run)

    ev = sub.add_parser("evaluate", help="Paired offline evaluation ONLY")
    ev.add_argument("--dataset", choices=DATASETS, required=True)
    ev.add_argument("--benchmark", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--compare", help="Another mode's predictions.jsonl")
    ev.add_argument("--original-prediction",
                    help="MedRGAG original prediction.json from identical Reader/split")
    ev.add_argument("--annotations",
                    help="Optional independent human sufficient true/false JSONL")
    ev.add_argument("--bootstrap", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=2026)
    ev.add_argument("--output", required=True)
    ev.set_defaults(func=_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
