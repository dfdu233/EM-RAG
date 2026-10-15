"""Reproducible, gold-free manifests from MedRGAG's exact benchmark.json.

Example:
python -m study.prepare --benchmark ../MedRGAG/benchmark.json \
    --dataset bioasq --limit 60 --seed 2026 \
    --output outputs/study/bioasq/manifest.jsonl

Existing 500/618 reported experiments have already influenced development.
No new "held-out" claim can be manufactured merely by changing a random seed.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from .io import DATASETS, write_json, write_jsonl, read_jsonl


def build_manifest(benchmark: str, dataset: str, limit: int | None,
                   seed: int, ids_from: str | None = None) -> tuple[list[dict], dict]:
    with Path(benchmark).open(encoding="utf-8") as f:
        whole = json.load(f)
    if dataset not in whole or not isinstance(whole[dataset], dict):
        raise ValueError(f"{dataset} missing or not ID-keyed in official MedRGAG benchmark")
    pool = whole[dataset]
    if ids_from:
        candidate = [str(row["id"]) for row in read_jsonl(ids_from)]
    else:
        candidate = sorted(pool, key=lambda id_: sha256(
            f"{seed}/{dataset}/{id_}".encode("utf-8")).hexdigest())
    if limit is not None:
        candidate = candidate[:limit]
    if not candidate or len(candidate) != len(set(candidate)):
        raise ValueError("Empty or duplicate selection")
    missing = set(candidate) - set(pool)
    if missing:
        raise ValueError(f"IDs absent from MedRGAG benchmark: {sorted(missing)[:5]}")
    rows = []
    for id_ in candidate:
        item = pool[id_]
        opts = item.get("options")
        if not isinstance(opts, dict) or not item.get("question"):
            raise ValueError(f"Invalid upstream question {id_}")
        rows.append({"id": str(id_), "question": str(item["question"]),
                     "options": {str(k).upper(): str(v) for k,v in opts.items()},
                     "dataset": dataset})
    meta = {
        "origin": "MedRGAG/benchmark.json", "dataset": dataset,
        "seed": seed, "selection": "prior_fixed_ids" if ids_from else "hash_uniform",
        "n": len(rows), "source_total": len(pool), "contains_gold": False,
        "ids_digest": sha256("\n".join(candidate).encode()).hexdigest(),
        "development_only": True,
        "warning": "Previously inspected benchmarks are not untouched test data. "
                   "Do not choose methods or tune thresholds using held-out answers.",
    }
    return rows, meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MedRGAG dataset-aligned mechanism manifest")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ids-from", help="Old EM-RAG traces.jsonl or audit manifest (same IDs)")
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    if a.limit < 1:
        p.error("limit must be >=1")
    rows, meta = build_manifest(a.benchmark, a.dataset, a.limit, a.seed, a.ids_from)
    write_jsonl(a.output, rows)
    write_json(str(a.output) + ".meta.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
