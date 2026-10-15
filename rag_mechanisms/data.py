"""Grounded, gold-isolated MedRGAG benchmark/cache loaders and fixed sampling."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: dict[str, str]

    @property
    def choice_labels(self) -> tuple[str, ...]:
        return tuple(self.options)

    def medrgag_query(self) -> str:
        """Upstream retrieval embeds the options as A. text, B. text."""
        return self.text + "\n" + "\n".join(
            f"{label}. {value}" for label, value in sorted(self.options.items()))


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    title: str = ""
    source: str = "unknown"
    score: float | None = None


def _id_order(identifier: str):
    return (0, int(identifier)) if identifier.isdigit() else (1, identifier)


def load_benchmark(path: str | Path, dataset: str
                   ) -> tuple[dict[str, Question], dict[str, str]]:
    """Gold is kept separate from Question and NEVER given to generation."""
    with Path(path).open(encoding="utf-8") as f:
        source = json.load(f)
    if not isinstance(source, dict) or dataset not in source:
        raise KeyError(f"Dataset {dataset!r} missing in {path}")
    data = source[dataset]
    if not isinstance(data, dict):
        raise ValueError("Expected MedRGAG benchmark[dataset] as id-to-row dict")
    questions, gold = {}, {}
    for raw_id in sorted(data, key=lambda x: _id_order(str(x))):
        value = data[raw_id]
        if not isinstance(value, dict):
            raise ValueError(f"Malformed benchmark entry {raw_id!r}")
        qid = str(raw_id)
        labels = value.get("options")
        if not isinstance(labels, dict) or not labels:
            raise ValueError(f"No answer-choice map for {qid}")
        options = {str(k).upper(): str(v) for k, v in labels.items()}
        if len(options) != len(labels) or any(not s.strip() for s in options.values()):
            raise ValueError(f"Malformed options for {qid}")
        questions[qid] = Question(id=qid, text=str(value.get("question", "")),
                                  options=options)
        raw_gold = str(value.get("answer", "")).strip()
        if raw_gold.upper() in options:
            gold[qid] = raw_gold.upper()
        else:
            candidates = [label for label, val in options.items()
                          if val.strip().casefold() == raw_gold.casefold()]
            if len(candidates) == 1:
                gold[qid] = candidates[0]
            else:
                raise ValueError(f"Invalid gold answer for {dataset}/{qid}: {raw_gold!r}")
    return questions, gold


def _load_snippet_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a MedRGAG id-keyed JSON object")
    return raw


def load_ranked_cache(path: str | Path) -> dict[str, list[Passage]]:
    """Reads MedRGAG outputs/DATASET/docs/retrieval/snippets.json (ranked).

    Retain order. Do not silently transform a precomputed one-time retrieval
    cache into an index or claim to have performed fresh follow-up retrieval.
    """
    obj = _load_snippet_dict(path)
    out: dict[str, list[Passage]] = {}
    for key, value in obj.items():
        if not isinstance(value, dict) or not isinstance(value.get("snippets"), list):
            continue
        seen: set[str] = set()
        docs: list[Passage] = []
        scores = value.get("scores", [])
        scores = scores if isinstance(scores, list) else []
        for idx, item in enumerate(value["snippets"]):
            if not isinstance(item, dict):
                continue
            text = item.get("contents") or item.get("content") or item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            normalized = " ".join(text.split()).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            score = None
            if idx < len(scores) and isinstance(scores[idx], (int, float)):
                score = float(scores[idx])
            docs.append(Passage(
                id=str(item.get("id", f"rank_{idx}")),
                title=str(item.get("title", "")), text=text.strip(),
                source=str(item.get("source") or item.get("corpus") or "MedRGAG"),
                score=score))
        out[str(key)] = docs
    return out


def load_generated_cache(path: str | Path) -> dict[str, list[Passage]]:
    """MedRGAG KGCC texts are model-generated, not ground-truth evidence."""
    obj = _load_snippet_dict(path)
    out: dict[str, list[Passage]] = {}
    for key, value in obj.items():
        if not isinstance(value, dict) or not isinstance(value.get("snippets"), list):
            continue
        docs = []
        for idx, item in enumerate(value["snippets"]):
            if isinstance(item, dict) and isinstance(item.get("contents"), str) and item["contents"].strip():
                docs.append(Passage(
                    id=str(item.get("id", f"generated_{idx}")),
                    text=item["contents"].strip(), source="PARAMETRIC_GENERATED"))
        out[str(key)] = docs
    return out


def hash_file(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def manifest(questions: dict[str, Question], *, dataset: str,
             seed: int = 2026, count: int | None = None,
             eligible_ids: set[str] | None = None) -> list[dict[str, str]]:
    """Sampling never uses answers, method predictions, or experimental gains."""
    identifiers = [qid for qid in questions if eligible_ids is None or qid in eligible_ids]
    identifiers.sort(key=lambda qid: sha256(
        f"{seed}|{dataset}|{qid}".encode("utf-8")).hexdigest())
    if count is not None:
        if count < 1:
            raise ValueError("count must be >= 1")
        identifiers = identifiers[:count]
    return [{"dataset": dataset, "id": qid} for qid in identifiers]


def read_manifest(path: str | Path, dataset: str) -> list[str]:
    output: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("dataset") != dataset:
                continue
            output.append(str(item["id"]))
    if not output or len(output) != len(set(output)):
        raise ValueError("No cases or duplicate IDs for selected dataset in manifest")
    return output


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_reference_predictions(path: str | Path) -> dict[str, str | None]:
    """Read existing MedRGAG/baseline prediction.json without rerunning methods."""
    with Path(path).open(encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        iterator = obj.items()
    elif isinstance(obj, list):
        iterator = [(str(x["id"]), x) for x in obj]
    else:
        raise ValueError("Reference prediction file must be list or dict")
    predictions: dict[str, str | None] = {}
    for key, value in iterator:
        if isinstance(value, dict):
            pred = value.get("model_answer", value.get("prediction",
                   value.get("answer_choice", None)))
        else:
            pred = value
        qid = str(key)
        if qid in predictions:
            raise ValueError(f"Duplicate reference ID {qid}")
        predictions[qid] = None if pred is None else str(pred).strip().upper()
    return predictions
