"""Run EM-RAG on MedRGAG benchmark.json or simple QA JSONL.

Example:
python -m emrag.cli --benchmark ../MedRGAG/benchmark.json --dataset medqa \
 --medrgag-root ../MedRGAG --db-dir ../MedRGAG/corpus \
 --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
 --max-examples 20 --output outputs/medqa
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

from .core import Question
from .engine import EMRAG, EMRAGConfig
from .metrics import summarize, summarize_actions
from .model import OpenAICompatibleModel
from .retrieval import (JSONLBM25Retriever, MedRGAGRetriever,
                        MedRGAGSeedRetriever, load_kgcc_hints)


def load_benchmark(path: str, dataset: str, offset: int = 0,
                   limit: int | None = None) -> tuple[list[Question], dict[str, str]]:
    """Gold labels are returned separately, never exposed to EMRAG.run."""
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open(encoding="utf-8") as f:
            items = [(str(i), json.loads(line)) for i, line in enumerate(f)
                     if line.strip()]
    else:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if dataset:
            if dataset not in data:
                raise KeyError(f"Dataset {dataset!r} missing in {path}")
            data = data[dataset]
        items = (list(data.items()) if isinstance(data, dict)
                 else [(str(i), v) for i, v in enumerate(data)])
    questions: list[Question] = []
    gold: dict[str, str] = {}
    selected = items[offset:offset + limit if limit is not None else None]
    for key, value in selected:
        if not isinstance(value, dict):
            raise ValueError(f"Entry {key} is not a dictionary")
        qid = str(value.get("id", key))
        opts = value.get("options")
        if not isinstance(opts, dict):
            raise ValueError(f"Entry {qid} must contain options dictionary")
        question = Question(
            id=qid, question=str(value.get("question", "")),
            options={str(k).upper(): str(v) for k, v in opts.items()})
        questions.append(question)
        raw = str(value.get("answer", value.get("gt_answer", ""))).strip()
        if raw:
            if raw.upper() in question.labels:
                gold[qid] = raw.upper()
            else:
                matches = [key for key, text in question.options.items()
                           if text.strip().casefold() == raw.casefold()]
                if len(matches) == 1:
                    gold[qid] = matches[0]
                else:
                    raise ValueError(f"Cannot resolve gold answer for entry {qid}")
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in selection")
    return questions, gold


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Training-free EM-RAG experimental runner")
    p.add_argument("--benchmark", required=True, help="MedRGAG benchmark.json or QA JSONL")
    p.add_argument("--dataset", default="medqa", help="Key in MedRGAG benchmark JSON")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--output", default="outputs/emrag")
    p.add_argument("--model", required=True, help="OpenAI-compatible served model name")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    r = p.add_mutually_exclusive_group(required=True)
    r.add_argument("--jsonl-corpus", nargs="+", help="Local JSONL chunk files (small corpus)")
    r.add_argument("--medrgag-root", help="Path of separate official MedRGAG clone")
    r.add_argument("--seed-retrieval", help="MedRGAG precomputed retrieval JSON (round-1 only)")
    p.add_argument("--db-dir", default="./corpus", help="MedRGAG corpus directory")
    p.add_argument("--corpus-name", default="MedText",
                   choices=["MedText", "Textbooks", "Wikipedia"])
    p.add_argument("--live-medrgag-root", help="For cached first-turn + live follow-up")
    p.add_argument("--kgcc-hints", help="MedRGAG generated snippets JSON; search hints ONLY")
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--queries-per-round", type=int, default=2)
    p.add_argument("--docs-per-query", type=int, default=5)
    p.add_argument("--max-new-docs", type=int, default=10)
    p.add_argument("--max-context-chars", type=int, default=9000)
    p.add_argument("--min-score-margin", type=int, default=2)
    p.add_argument("--unverified-policy", choices=["baseline", "abstain"], default="baseline")
    p.add_argument("--update-policy", choices=["evidence", "latest", "frozen"], default="evidence",
                   help="Evidence-gated EM-RAG versus matched-compute ablations")
    p.add_argument("--stop-on-verified", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_examples < 1 or args.offset < 0:
        raise ValueError("max-examples must be >= 1 and offset must be nonnegative")
    questions, gold = load_benchmark(
        args.benchmark, args.dataset, args.offset, args.max_examples)
    if not questions:
        raise ValueError("No questions selected")
    if args.jsonl_corpus:
        retriever = JSONLBM25Retriever(args.jsonl_corpus)
    elif args.medrgag_root:
        retriever = MedRGAGRetriever(
            args.medrgag_root, corpus_name=args.corpus_name, db_dir=args.db_dir)
    else:
        live = (MedRGAGRetriever(args.live_medrgag_root,
                                corpus_name=args.corpus_name,
                                db_dir=args.db_dir)
                if args.live_medrgag_root else None)
        retriever = MedRGAGSeedRetriever(args.seed_retrieval, live)
        if live is None and args.max_rounds > 1:
            print("WARNING: cached first-turn retrieval without live follow-up; "
                  "this is NOT a valid multi-round retrieval comparison.", file=sys.stderr)
    hints = load_kgcc_hints(args.kgcc_hints) if args.kgcc_hints else {}
    api_key = args.api_key
    if not api_key and args.api_base and ("localhost" in args.api_base or
                                          "127.0.0.1" in args.api_base):
        api_key = "EMPTY"
    model = OpenAICompatibleModel(args.model, base_url=args.api_base, api_key=api_key)
    config = EMRAGConfig(
        max_rounds=args.max_rounds, queries_per_round=args.queries_per_round,
        docs_per_query=args.docs_per_query, max_new_docs=args.max_new_docs,
        max_context_chars=args.max_context_chars,
        min_score_margin=args.min_score_margin,
        unverified_policy=args.unverified_policy,
        update_policy=args.update_policy,
        stop_on_verified=args.stop_on_verified)
    engine = EMRAG(model, retriever, config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with (out / "traces.jsonl").open("w", encoding="utf-8") as fp:
        for index, q in enumerate(questions, start=1):
            result = engine.run(q, kgcc_hints=hints.get(q.id, []))
            results.append(result)
            fp.write(json.dumps(result.to_record(), ensure_ascii=False) + "\n")
            fp.flush()
            print(f"[{index}/{len(questions)}] id={q.id} baseline={result.baseline} "
                  f"final={result.selected} status={result.status} "
                  f"calls={result.model_calls}", flush=True)
    if len(gold) == len(results):
        stats = summarize(results, gold, max_rounds=args.max_rounds)
        stats["action_counts"] = summarize_actions(results)
        with (out / "metrics.json").open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, ensure_ascii=False, indent=2)
        print(f"Accuracy by round: "
              f"{[round(s['accuracy'], 4) for s in stats['rounds']]}")
        print(f"Final coverage: {stats['final_coverage']:.3f}")
    else:
        print("No complete gold labels; traces saved but evaluation skipped")
    print(f"Saved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
