"""MedRGAG four-paper A/B/C/D executable. Gold read only during evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter

from .data import DATASETS, Question, MedRGAGCaches, fingerprint, templates
from .models import OpenAIChat, MedCPTReranker, IdentityRanker
from .abcd import MedRGAGABCD, Config
from .live_retrieval import NativeMedRGAGBM25


def _questions_without_gold(path: str | Path, dataset: str) -> dict[str, Question]:
    """Read only question/options; NEVER inspect the benchmark answer field.

    Evaluation loads gold in a separate post-hoc process.
    """
    if dataset not in DATASETS:
        raise ValueError("Unknown MedRGAG dataset")
    with Path(path).open(encoding="utf-8") as f:
        section = json.load(f).get(dataset)
    if not isinstance(section, dict):
        raise ValueError("Missing MedRGAG benchmark dataset section")
    questions = {}
    for key, row in section.items():
        if not isinstance(row, dict) or not isinstance(row.get("options"), dict):
            raise ValueError("Malformed benchmark question/options")
        labels = {str(k).upper(): str(v) for k, v in row["options"].items()}
        if not labels or len(labels) != len(row["options"]):
            raise ValueError("Invalid answer option labels")
        text = row.get("question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty question")
        questions[str(key)] = Question(str(key), text, labels)
    return questions


def _frozen_ids(path: str | Path, dataset: str) -> list[str]:
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row.get("dataset", dataset) == dataset:
                    rows.append(str(row["id"]))
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Manifest must contain unique selected question IDs")
    return rows


def _run(args) -> int:
    root = Path(args.medrgag_root)
    source_paths = {
        "benchmark": Path(args.benchmark or root / "benchmark.json"),
        "retrieval": Path(args.retrieval_cache or root /
                          "outputs" / args.dataset / "docs" / "retrieval" / "snippets.json"),
        "generation": Path(args.generated_cache or root /
                           "outputs" / args.dataset / "docs" / "generate" / "snippets.json"),
        "selection": Path(args.selection_cache or root /
                          "outputs" / args.dataset / "docs" / "generate" /
                          "snippets_final_select.json"),
        "templates": Path(args.template or root / "src" / "template.py"),
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Original MedRGAG {label} missing: {path}")
    if args.limit < 1 or args.offset < 0 or args.max_new_examples < 1:
        raise ValueError("Positive limit/max-new-examples and nonnegative offset required")
    config = Config(
        max_rounds=args.max_rounds, retrieval_k=args.retrieval_k,
        fresh_per_round=args.fresh_per_round,
        max_query_chars=args.max_query_chars,
        pseudo_doc_chars=args.pseudo_doc_chars,
        excerpt_chars=args.excerpt_chars,
        extract_sentences=args.extract_sentences,
        extract_chars=args.extract_chars,
        note_chars=args.note_chars,
        use_query2doc=not args.no_query2doc,
        use_ircot=not args.no_ircot,
        use_recomp=not args.no_recomp,
        use_sure=not args.no_sure,
        context_tokens=args.context_tokens,
        max_prompt_chars=args.max_prompt_chars,
        temperature=args.temperature, seed=args.seed)
    questions = _questions_without_gold(source_paths["benchmark"], args.dataset)
    if args.ids_file:
        ids = _frozen_ids(args.ids_file, args.dataset)
        if set(ids) - set(questions):
            raise ValueError("Frozen manifest contains unknown question IDs")
    else:
        ids = list(questions)[args.offset:args.offset + args.limit]
    if not ids:
        raise ValueError("No selected examples")
    if not args.no_rerank and not args.reader_tokenizer:
        # Exact original MedRGAG prompt truncation depends on tokenizer.
        print("WARNING: no --reader-tokenizer; original MedRGAG token truncation "
              "cannot be guaranteed identical, especially on long prompts.")
    files_sha = {k: fingerprint(p) for k, p in source_paths.items()}
    args_saved = {
        k: v for k, v in vars(args).items()
        if k not in {"func", "api_key", "planner_api_key"}}
    signature = {
        "algorithm": "medrgag_four_papers_abcd",
        "mode": "ABCD" if all((config.use_query2doc, config.use_ircot, config.use_recomp, config.use_sure)) else "ablation",
        "config": asdict(config),
        "run_args": args_saved,
        "frozen_files_sha256": files_sha,
        "frozen_ids": ids,
        "original_KADS_at_B0": True,
        "KGCC_is_hypothesis_not_external_fact": True,
        "source_nonoverlap_does_not_imply_independence": True,
        "no_gold_in_inference": True,
        "historical_KGCC_and_KADS_preprocessing_cost_excluded": True,
    }
    signature = json.loads(json.dumps(signature, ensure_ascii=False))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path, predictions_path = (
        out_dir / "run_config.json", out_dir / "predictions.jsonl")
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Refusing to mix budgets, corpora, Readers, or caches")
    elif predictions_path.exists():
        raise ValueError("Existing predictions without provenance metadata")
    else:
        meta_path.write_text(
            json.dumps(signature, ensure_ascii=False, indent=2),
            encoding="utf-8")
    done: set[str] = set()
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            saved = json.loads(line)
            qid = str(saved["id"])
            if qid in done:
                raise ValueError(f"Duplicate completed question {qid}")
            done.add(qid)
        if done - set(ids):
            raise ValueError("Trace records unseen manifest ID")
    from .cli import _load_reader_tokenizer
    tokenizer = _load_reader_tokenizer(args.reader_tokenizer)
    reader = OpenAIChat(
        args.reader_model, base_url=args.api_base,
        api_key=args.api_key,
        send_seed=not args.reader_omit_seed)
    auxiliary = None
    retriever = None
    if config.max_rounds:
        if config.use_ircot or config.use_sure:
            if not args.planner_model:
                raise ValueError("--planner-model required for IRCoT/SuRe")
            auxiliary = OpenAIChat(
                args.planner_model,
                base_url=args.planner_api_base or args.api_base,
                api_key=args.planner_api_key or args.api_key,
                send_seed=not args.planner_omit_seed,
                send_temperature=not args.planner_omit_temperature)
        retriever = NativeMedRGAGBM25(
            medrgag_root=root, corpus_dir=args.corpus_dir,
            corpus_name=args.corpus)
    ranker = (IdentityRanker() if args.no_rerank else
              MedCPTReranker(args.ranker_model, device=args.ranker_device))
    rag_template, _ = templates(source_paths["templates"])
    sources = MedRGAGCaches(
        source_paths["retrieval"], source_paths["generation"],
        source_paths["selection"])
    system = MedRGAGABCD(
        reader=reader, auxiliary=auxiliary, retriever=retriever,
        ranker=ranker, rag_template=rag_template,
        config=config, tokenizer=tokenizer)
    completed = 0
    with predictions_path.open("a", encoding="utf-8") as handle:
        for ix, qid in enumerate(ids):
            if qid in done:
                continue
            if completed >= args.max_new_examples:
                print("Stopped at explicit per-run case limit; resume with same configuration")
                break
            retrieved, generated = sources.get(qid)
            beginning = perf_counter()
            result = system.run(
                questions[qid], retrieved, generated,
                original_selection=sources.official_selection(qid))
            result.update({
                "dataset": args.dataset, "reader": args.reader_model,
                "planner": args.planner_model if config.max_rounds and (config.use_ircot or config.use_sure) else None,
                "corpus": args.corpus, "qid": qid,
                "elapsed_wall_seconds": perf_counter() - beginning,
                "uses_real_retrieval_index": config.max_rounds > 0,
                "ranker_is_real": not args.no_rerank,
            })
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            print(f"[{ix+1}/{len(ids)}] {args.dataset}/{qid} "
                  f"KADS={result['base_prediction']} final={result['prediction']} "
                  f"Reader={result['reader_calls']} "
                  f"new_queries={result['retrieval_calls']}", flush=True)
    print(f"Saved {predictions_path}, {completed} NEW questions. "
          "Do not use test gold to choose this algorithm.")
    return 0


def _evaluate(args) -> int:
    from .progressive_eval import evaluate_progressive
    result = evaluate_progressive(
        args.predictions, args.benchmark, args.dataset,
        bootstrap=args.bootstrap, seed=args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": args.dataset, "by_budget": result["by_budget"],
        "transitions": result["transitions"],
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedRGAG + Query2doc/IRCoT/RECOMP/SuRe-inspired mechanisms")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--medrgag-root", default="../MedRGAG")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--benchmark")
    run.add_argument("--retrieval-cache")
    run.add_argument("--generated-cache")
    run.add_argument("--selection-cache")
    run.add_argument("--template")
    run.add_argument("--ids-file")
    run.add_argument("--limit", type=int, default=20)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--max-new-examples", type=int, default=50)
    run.add_argument("--output", required=True)
    run.add_argument("--reader-model", required=True)
    run.add_argument("--reader-tokenizer")
    run.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    run.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    run.add_argument("--reader-omit-seed", action="store_true")
    run.add_argument("--planner-model")
    run.add_argument("--planner-api-base")
    run.add_argument("--planner-api-key")
    run.add_argument("--planner-omit-temperature", action="store_true")
    run.add_argument("--planner-omit-seed", action="store_true")
    run.add_argument("--corpus-dir", help="Path to already indexed original MedRGAG corpus")
    run.add_argument("--corpus", choices=("Textbooks", "Wikipedia", "MedText"),
                     default="MedText")
    run.add_argument("--ranker-model", default="ncbi/MedCPT-Cross-Encoder")
    run.add_argument("--ranker-device")
    run.add_argument("--no-rerank", action="store_true",
                     help="Offline fixture only: NOT original MedRGAG parity")
    run.add_argument("--max-rounds", type=int, default=2)
    run.add_argument("--retrieval-k", type=int, default=8,
                     help="Candidates fetched PER CORPUS per new query")
    run.add_argument("--fresh-per-round", type=int, default=3)
    run.add_argument("--max-query-chars", type=int, default=300)
    run.add_argument("--pseudo-doc-chars", type=int, default=100)
    run.add_argument("--excerpt-chars", type=int, default=750)
    run.add_argument("--extract-sentences", type=int, default=2)
    run.add_argument("--extract-chars", type=int, default=550)
    run.add_argument("--note-chars", type=int, default=1600)
    run.add_argument("--no-query2doc", action="store_true")
    run.add_argument("--no-ircot", action="store_true")
    run.add_argument("--no-recomp", action="store_true")
    run.add_argument("--no-sure", action="store_true")
    run.add_argument("--context-tokens", type=int, default=128000)
    run.add_argument("--max-prompt-chars", type=int, default=150000)
    run.add_argument("--temperature", type=float, default=0.1)
    run.add_argument("--seed", type=int, default=42)
    run.set_defaults(func=_run)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", choices=DATASETS, required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--benchmark", required=True)
    ev.add_argument("--bootstrap", type=int, default=1000)
    ev.add_argument("--seed", type=int, default=2026)
    ev.add_argument("--output", required=True)
    ev.set_defaults(func=_evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
