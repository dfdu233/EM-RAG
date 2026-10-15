"""Shared strictly validated JSONL, ID and source document conventions."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import re

DATASETS = ("medqa", "medmcqa", "mmlu", "pubmedqa", "bioasq")
READERS = ("qwen2.5", "llama3.1", "ministral-8B")


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError(f"Missing object/id at {path}:{i}")
            records.append(row)
    ids = [str(r["id"]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def manifest_items(path: str | Path) -> list[dict]:
    items = read_jsonl(path)
    for row in items:
        if "answer" in row or "gold" in row or "gt_answer" in row:
            raise ValueError("Gold labels cannot occur in inference manifest")
        if not isinstance(row.get("options"), dict) or not row.get("question"):
            raise ValueError("Manifest must have question/options/id")
        labels = [k.upper() for k in row["options"]]
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate choice labels")
    return items


def question_text(row: dict) -> str:
    return str(row["question"]) + "\n" + "\n".join(
        f"{k.upper()}. {v}" for k, v in row["options"].items()
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def doc_identity(doc: dict) -> str:
    """Use source/text stable identity; empty IDs can never merge distinct texts."""
    body = normalize(str(doc.get("contents", "")))
    if not body:
        raise ValueError("Document must include contents")
    return "doc-" + digest(body)


def unique_docs(docs: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for raw in docs:
        if not isinstance(raw, dict) or not str(raw.get("contents", "")).strip():
            continue
        doc = dict(raw)
        key = doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        doc["study_id"] = key
        out.append(doc)
    return out


def strict_gold(benchmark_path: str, dataset: str, ids: list[str]) -> dict[str, str]:
    """Offline evaluation ONLY. Never call within retriever/Reader modules."""
    from emrag.cli import load_benchmark
    _, labels = load_benchmark(benchmark_path, dataset, limit=None)
    missing = set(ids) - set(labels)
    if missing:
        raise ValueError(f"Missing gold labels for {sorted(missing)[:5]}")
    return {id_: labels[id_] for id_ in ids}


def same_ids(left: list[dict], right: list[dict]) -> None:
    a = [str(x["id"]) for x in left]
    b = [str(x["id"]) for x in right]
    if a != b:
        raise ValueError("Different IDs/order across paired experiments")
