"""Server runner for training-free Anytime Evidence Selection on original MedRGAG.

Separate experiment, no modifications to old EM-RAG Gate or CA-MedRGAG.
No auxiliary LLM. Original KGCC/KADS caches and frozen MedCPT+Reader are
reused. Gold enters ONLY the independent post-hoc evaluator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .anytime import AnytimeMedRGAG, Config
from .data import DATASETS, MedRGAGCaches, benchmark, fingerprint, templates
from .models import IdentityRanker, MedCPTReranker, OpenAIChat
from .utility import LexicalScorer, MedCPTScorer


def _read_ids(path: str | Path, dataset: str) -> list[str]:
    ids = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset", dataset) == dataset:
                ids.append(str(row["id"]))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Frozen manifest empty or contains duplicate IDs")
    return ids


def _run(args) -> int:
    root = Path(args.medrgag_root)
    paths = {
        "benchmark": Path(args.benchmark or root / "benchmark.json"),
        "retrieval": Path(args.retrieval_cache or root /
                          "outputs" / args.dataset / "docs" / "retrieval" / "snippets.json"),
        "generation": Path(args.generated_cache or root /
                           "outputs" / args.dataset / "docs" / "generate" / "snippets.json"),
        "templates": Path(args.template or root / "src" / "template.py"),
    }
    if not args.no_kads_anchor:
        paths["kads"] = Path(args.selection_cache or root /
                             "outputs" / args.dataset / "docs" / "generate" /
                             "snippets_final_select.json")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"MedRGAG {name} cache missing: {path}")
    questions, _ = benchmark(paths["benchmark"], args.dataset)
    if args.ids_file:
        ids = _read_ids(args.ids_file, args.dataset)
        unknown = set(ids) - set(questions)
        if unknown:
            raise KeyError(f"Sample IDs not in benchmark: {sorted(unknown)[:8]}")
    else:
        if args.offset < 0 or args.limit < 1:
            raise ValueError("offset>=0 and limit>=1 required")
        ids = list(questions)[args.offset:args.offset + args.limit]
    if not ids:
        raise ValueError("No benchmark examples selected")
    if args.max_new_examples < 1:
        raise ValueError("--max-new-examples must be positive")
    if args.no_rerank and args.utility == "medcpt":
        raise ValueError("MedCPT utility requires the original MedCPT reranker; "
                         "choose --utility lexical for no-GPU fixture runs")
    cfg = Config(
        depths=tuple(args.depths),
        read_each_depth=args.read_each_depth,
        initial_kads_anchor=not args.no_kads_anchor,
        context_tokens=args.context_tokens,
        max_reader_prompt_chars=args.max_reader_prompt_chars,
        reader_seed=args.reader_seed)
    sources = MedRGAGCaches(
        paths["retrieval"], paths["generation"], paths.get("kads"))
    rag_template, _ = templates(paths["templates"])
    public_args = {k: v for k, v in vars(args).items()
                   if k not in {"api_key", "func"}}
    signature = {
        "algorithm": "training-free anytime evidence utility on MedRGAG",
        "config": asdict(cfg),
        "run_args": public_args,
        "source_hashes": {k: fingerprint(path) for k, path in paths.items()},
        "ids": ids,
        "gold_in_inference": False,
        "utility_is_true_correctness_probability": False,
        "retrieval_is_precomputed": True,
        "preprocessing_KGCC_KADS_cost_excluded": True,
    }
    # The serialized config is the only comparison representation; tuple/list
    # round trips otherwise spuriously reject an identical resumed invocation.
    signature = json.loads(json.dumps(signature, ensure_ascii=False))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config_file, trace_file = out / "run_config.json", out / "predictions.jsonl"
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != signature:
            raise ValueError("Refusing to mix changed budgets or caches into prior output")
    elif trace_file.exists():
        raise ValueError("Existing predictions without configuration metadata")
    else:
        config_file.write_text(json.dumps(signature, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    done = set()
    if trace_file.exists():
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["id"])
            if qid in done:
                raise ValueError(f"Duplicate resume record {qid}")
            done.add(qid)
        if done - set(ids):
            raise ValueError("Saved output includes unselected IDs")
    from .cli import _load_reader_tokenizer
    tokenizer = _load_reader_tokenizer(args.reader_tokenizer)
    reader = OpenAIChat(
        args.reader_model, base_url=args.api_base, api_key=args.api_key,
        send_seed=not args.reader_omit_seed)
    ranker = (IdentityRanker() if args.no_rerank else
              MedCPTReranker(args.ranker_model, device=args.ranker_device))
    def scorer_factory(q):
        return (MedCPTScorer(q, ranker, max_facets=args.max_facets)
                if args.utility == "medcpt"
                else LexicalScorer(q, max_facets=args.max_facets))
    engine = AnytimeMedRGAG(
        reader=reader, ranker=ranker, rag_template=rag_template,
        scorer_factory=scorer_factory, config=cfg, tokenizer=tokenizer)
    new = 0
    with trace_file.open("a", encoding="utf-8") as file:
        for i, qid in enumerate(ids):
            if qid in done:
                continue
            if new >= args.max_new_examples:
                print("Reached explicit per-command sample cap; resume with same args")
                break
            retrieved, generated = sources.get(qid)
            anchor = (sources.official_selection(qid)
                      if not args.no_kads_anchor else None)
            result = engine.run(
                questions[qid], retrieved, generated,
                original_selection=anchor)
            result.update({
                "dataset": args.dataset, "reader": args.reader_model,
                "source_cache_question_id": qid,
            })
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            new += 1
            print(
                f"[{i+1}/{len(ids)}] {args.dataset}/{qid} "
                f"utility={result['estimated_best_utility']:.4f} "
                f"new_source={len(result['new_source_docs_in_reader'])} "
                f"answer={result['prediction']} reader_calls={result['reader_calls']}",
                flush=True)
    print(f"Saved {trace_file}; {new} new cases. No training or answer Gate.")
    return 0


def _evaluate(args) -> int:
    from .evaluation import evaluate
    from .anytime_eval import prefix_curve
    value = evaluate(
        args.predictions, args.benchmark, args.dataset,
        compare=args.compare, original_prediction=args.original_prediction,
        bootstrap=args.bootstrap, seed=args.seed)
    if args.include_prefix_curve:
        value["prefix_curve"] = prefix_curve(
            args.predictions, args.benchmark, args.dataset,
            bootstrap=args.bootstrap, seed=args.seed)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(value["summary"], ensure_ascii=False, indent=2))
    if "prefix_curve" in value:
        print(json.dumps(value["prefix_curve"]["by_depth"],
                         ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Training-free Anytime Evidence Selection with MedRGAG")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--medrgag-root", default="../MedRGAG")
    run.add_argument("--benchmark")
    run.add_argument("--retrieval-cache")
    run.add_argument("--generated-cache")
    run.add_argument("--selection-cache")
    run.add_argument("--template")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--reader-model", required=True)
    run.add_argument("--reader-tokenizer")
    run.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    run.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    run.add_argument("--reader-omit-seed", action="store_true")
    run.add_argument("--reader-seed", type=int, default=42)
    run.add_argument("--utility", choices=("medcpt", "lexical"),
                     default="medcpt")
    run.add_argument("--max-facets", type=int, default=8)
    run.add_argument("--ranker-model", default="ncbi/MedCPT-Cross-Encoder")
    run.add_argument("--ranker-device")
    run.add_argument("--no-rerank", action="store_true",
                     help="Only for lexical/fixture ablation; not MedRGAG parity")
    run.add_argument("--depths", nargs="+", type=int,
                     default=[5, 10, 20, 32])
    run.add_argument("--context-tokens", type=int, default=128000)
    run.add_argument("--max-reader-prompt-chars", type=int, default=150000)
    run.add_argument("--no-kads-anchor", action="store_true",
                     help="Ablation: skip original historical KADS selection")
    run.add_argument("--read-each-depth", action="store_true",
                     help="Evaluation-only: Reader at every K, more GPU cost")
    run.add_argument("--ids-file", help="Frozen manifest containing dataset and id")
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--limit", type=int, default=20)
    run.add_argument("--max-new-examples", type=int, default=50)
    run.add_argument("--output", required=True)
    run.set_defaults(func=_run)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", choices=DATASETS, required=True)
    ev.add_argument("--benchmark", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--compare", help="Other run prediction JSONL")
    ev.add_argument("--original-prediction", help="Original MedRGAG prediction.json")
    ev.add_argument("--include-prefix-curve", action="store_true")
    ev.add_argument("--bootstrap", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=2026)
    ev.add_argument("--output", required=True)
    ev.set_defaults(func=_evaluate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
