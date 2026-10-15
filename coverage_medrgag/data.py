"""Original MedRGAG benchmark, KGCC and KADS cache adapters.

Reader inputs do not carry gold labels. Source IDs r0.. and g0.. are internal,
stable document positions. Original cache ordering is preserved exactly.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DATASETS = ("medqa", "medmcqa", "mmlu", "pubmedqa", "bioasq")


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: dict[str, str]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.options)

    def formatted(self) -> str:
        return self.text + "\n" + "".join(
            f"{label}: {answer}\n" for label, answer in self.options.items())

    def for_reranker(self) -> str:
        return self.text + "\n" + "\n".join(
            f"{k}. {v}" for k, v in sorted(self.options.items()))


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    origin: str
    original_id: str = ""
    title: str = ""
    score: float | None = None


def benchmark(path: str | Path, dataset: str
              ) -> tuple[dict[str, Question], dict[str, str]]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown original MedRGAG dataset {dataset}")
    with Path(path).open(encoding="utf-8") as f:
        root = json.load(f)
    if not isinstance(root, dict) or not isinstance(root.get(dataset), dict):
        raise ValueError(f"Missing MedRGAG benchmark section {dataset}")
    questions: dict[str, Question] = {}
    gold: dict[str, str] = {}
    for key, value in root[dataset].items():
        if not isinstance(value, dict) or not isinstance(value.get("options"), dict):
            raise ValueError(f"Malformed MedRGAG question {dataset}/{key}")
        qid = str(key)
        opts = {str(k).upper(): str(v) for k, v in value["options"].items()}
        if not opts or len(opts) != len(value["options"]):
            raise ValueError(f"Malformed answer-choice labels: {qid}")
        question = Question(qid, str(value.get("question", "")), opts)
        if not question.text.strip():
            raise ValueError(f"Empty question: {qid}")
        questions[qid] = question
        answer = str(value.get("answer", "")).strip()
        if answer.upper() in opts:
            gold[qid] = answer.upper()
        else:
            matched = [label for label, text in opts.items()
                       if text.strip().casefold() == answer.casefold()]
            if len(matched) != 1:
                raise ValueError(f"Gold label not resolvable for {qid}")
            gold[qid] = matched[0]
    return questions, gold


def _object(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be MedRGAG id-keyed JSON")
    return obj


class MedRGAGCaches:
    def __init__(self, retrieval: str | Path, generation: str | Path,
                 selections: str | Path | None = None):
        self.retrieved = _object(retrieval)
        self.generated = _object(generation)
        self.selected = _object(selections) if selections else None

    @staticmethod
    def _docs(data: dict, *, prefix: str, origin: str) -> list[Passage]:
        if not isinstance(data, dict) or not isinstance(data.get("snippets"), list):
            raise ValueError(f"Missing {origin} snippets")
        scores = data.get("scores", [])
        if not isinstance(scores, list):
            scores = []
        docs: list[Passage] = []
        for n, entry in enumerate(data["snippets"]):
            if not isinstance(entry, dict):
                raise ValueError(f"Malformed {origin} snippet #{n}")
            body = entry.get("contents")
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"Empty {origin} snippet #{n}")
            score = (float(scores[n]) if n < len(scores)
                     and isinstance(scores[n], (int, float)) else None)
            docs.append(Passage(
                id=f"{prefix}{n}", text=body, origin=origin,
                original_id=str(entry.get("id", n)),
                title=str(entry.get("title", "")), score=score))
        return docs

    def get(self, question_id: str
            ) -> tuple[list[Passage], list[Passage]]:
        if question_id not in self.retrieved or question_id not in self.generated:
            raise KeyError(f"MedRGAG retrieval/KGCC cache missing ID {question_id}")
        retrieval = self._docs(self.retrieved[question_id],
                               prefix="r", origin="retrieved")
        generation = self._docs(self.generated[question_id],
                                prefix="g", origin="parametric_generated")
        if len(retrieval) < 5 or len(generation) < 5:
            raise ValueError(f"MedRGAG needs >=5 retrieved and >=5 generated docs: {question_id}")
        return retrieval, generation

    def official_selection(self, question_id: str) -> tuple[int, ...]:
        if self.selected is None:
            raise ValueError("Original KADS selection cache is required for reference mode")
        if question_id not in self.selected:
            raise KeyError(f"Original KADS selection missing: {question_id}")
        row = self.selected[question_id]
        indexes = row.get("final_list") if isinstance(row, dict) else None
        if not isinstance(indexes, list) or not 1 <= len(indexes) <= 5:
            raise ValueError(f"Invalid original KADS final_list for {question_id}")
        if any(type(x) is not int or x < 0 or x >= 10 for x in indexes):
            raise ValueError(f"KADS index outside first 5 retrieved + 5 generated: {question_id}")
        if len(set(indexes)) != len(indexes):
            raise ValueError(f"Duplicate KADS indexes for {question_id}")
        return tuple(indexes)


def templates(path: str | Path) -> tuple[str, str]:
    """Read original MedRGAG templates without executing its dependencies."""
    p = Path(path)
    tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    found: dict[str, str] = {}
    wanted = {"rag_template", "llm_selection_prompt"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != wanted or not all("{documents}" in x and "{question}" in x
                                      for x in found.values()):
        raise ValueError("MedRGAG src/template.py missing original RAG/KADS prompts")
    return found["rag_template"], found["llm_selection_prompt"]


def reader_prompt(q: Question, docs: list[Passage], template: str,
                  *, tokenizer=None, context_tokens: int = 128000,
                  max_prompt_chars: int = 150000) -> str:
    # Equivalent to MedRGAG src/medrgag.py::get_prompt; tokenizer truncation
    # requires local Reader tokenizer. Never truncate unseen text silently.
    context = "\n".join(
        f"Document [{idx}]: {d.text.replace(chr(10), ' ')}"
        for idx, d in enumerate(docs))
    if tokenizer is not None:
        ids = tokenizer.encode(context, add_special_tokens=False)
        if len(ids) > context_tokens:
            context = tokenizer.decode(ids[:context_tokens])
    result = template.format(documents=context, question=q.formatted())
    if len(result) > max_prompt_chars:
        raise ValueError(f"Reader prompt too long: {len(result)} > {max_prompt_chars}")
    return result


def original_candidates(retrieved: list[Passage],
                        generated: list[Passage]) -> list[Passage]:
    return retrieved[:5] + generated[:5]


def fingerprint(path: str | Path) -> str:
    from hashlib import sha256
    digest = sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()
