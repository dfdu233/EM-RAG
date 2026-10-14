"""Retrieval: dependency-free small-corpus BM25 and live MedRGAG integration.

MedRGAG provides src/utils.py::RetrievalSystem.retrieve(question, k).
Precomputed initial retrieval is a warm start, NOT repeated live retrieval.
"""
from __future__ import annotations
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Protocol
from .core import Document


class Retriever(Protocol):
    def search(self, query: str, k: int, *, question_id: str = "") -> list[Document]: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class JSONLBM25Retriever:
    """Small-corpus reference; use MedRGAG/Pyserini for large-scale retrieval."""

    def __init__(self, paths: list[str | Path], k1: float = 1.5, b: float = 0.75):
        self.docs: list[Document] = []
        for path in paths:
            p = Path(path)
            with p.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    body = str(value.get("contents") or value.get("content") or value.get("text") or "")
                    if not body.strip():
                        continue
                    self.docs.append(Document(
                        id=str(value.get("id", f"{p.name}:{line_no}")), contents=body,
                        title=str(value.get("title", "")), source=str(value.get("source", p.stem))))
        if not self.docs:
            raise ValueError("JSONL corpus has no non-empty documents")
        self.k1 = k1
        self.b = b
        self.counts = [Counter(tokenize(d.contents)) for d in self.docs]
        self.doc_lens = [sum(c.values()) for c in self.counts]
        self.avgdl = sum(self.doc_lens) / len(self.docs)
        df: Counter[str] = Counter()
        for c in self.counts:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
                    for term, freq in df.items()}

    def search(self, query: str, k: int, *, question_id: str = "") -> list[Document]:
        if k < 1:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []
        result: list[tuple[float, int]] = []
        for index, counts in enumerate(self.counts):
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[index] / self.avgdl)
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            if score > 0:
                result.append((score, index))
        result.sort(key=lambda pair: (-pair[0], pair[1]))
        return [Document(id=self.docs[i].id, contents=self.docs[i].contents,
                         title=self.docs[i].title, source=self.docs[i].source,
                         score=score, query=query)
                for score, i in result[:k]]


class MedRGAGRetriever:
    """Calls the original MedRGAG RetrievalSystem with new queries every round.

    Requires local clone, Pyserini/FAISS dependencies and corpus/index setup.
    The upstream code may download and index corpora at initialization.
    """

    def __init__(self, medrgag_root: str, corpus_name: str = "MedText",
                 db_dir: str = "./corpus", retriever_name: str = "BM25"):
        util_file = Path(medrgag_root).resolve() / "src" / "utils.py"
        if not util_file.is_file():
            raise FileNotFoundError(f"Expected MedRGAG at {util_file}")
        spec = importlib.util.spec_from_file_location("_emrag_medrgag_utils", util_file)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import MedRGAG utils")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.system = module.RetrievalSystem(
            retriever_name=retriever_name, corpus_name=corpus_name,
            db_dir=db_dir, cache=False, HNSW=False)
        self.corpus_name = corpus_name

    def search(self, query: str, k: int, *, question_id: str = "") -> list[Document]:
        snippets, scores = self.system.retrieve(query, k=k)
        output: list[Document] = []
        for i, snippet in enumerate(snippets):
            if not isinstance(snippet, dict):
                continue
            body = str(snippet.get("contents") or snippet.get("content") or "")
            if body.strip():
                output.append(Document(
                    id=str(snippet.get("id", i)), contents=body,
                    title=str(snippet.get("title", "")),
                    source=f"MedRGAG:{self.corpus_name}",
                    score=float(scores[i]) if i < len(scores) else 0.0,
                    query=query))
        return output


class MedRGAGSeedRetriever:
    """Use cached initial retrieved snippets only once per question ID.

    Later queries fall through to a live retriever. No live adapter = no
    subsequent retrieval, not a fake iterative baseline.
    """

    def __init__(self, json_path: str, live_retriever: Retriever | None = None):
        with open(json_path, encoding="utf-8") as fp:
            self.data = json.load(fp)
        self.live = live_retriever
        self._used: set[str] = set()

    def search(self, query: str, k: int, *, question_id: str = "") -> list[Document]:
        key = str(question_id)
        if key and key not in self._used and key in self.data:
            self._used.add(key)
            data = self.data[key]
            result = []
            for idx, item in enumerate(data.get("snippets", [])[:k]):
                if not isinstance(item, dict):
                    continue
                scores = data.get("scores", [])
                result.append(Document(
                    id=str(item.get("id", idx)), contents=str(item.get("contents", "")),
                    title=str(item.get("title", "")),
                    source="MedRGAG:precomputed",
                    score=float(scores[idx]) if idx < len(scores) else 0.0,
                    query=query))
            return result
        return self.live.search(query, k, question_id=question_id) if self.live else []


def load_kgcc_hints(path: str) -> dict[str, list[str]]:
    """MedRGAG KGCC generations are unverified QUERY HINTS, not evidence."""
    with open(path, encoding="utf-8") as fp:
        obj = json.load(fp)
    return {str(qid): [str(x.get("contents", ""))[:240]
                        for x in row.get("snippets", [])[:3] if isinstance(x, dict)]
            for qid, row in obj.items() if isinstance(row, dict)}
