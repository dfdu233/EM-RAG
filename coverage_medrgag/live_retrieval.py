"""Live, on-server BM25 adapter to ORIGINAL MedRGAG's MedText corpus.

This changes the information frontier: new queries are actually sent to
Pyserini Lucene indices, unlike precomputed top-32 prefix experiments.
No retrieval index is silently cloned/built/downloaded. The original
MedRGAG utility imports heavy optional server dependencies lazily.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .data import Passage


_CORPORA = {
    "Textbooks": ("textbooks",),
    "Wikipedia": ("wikipedia",),
    "MedText": ("textbooks", "wikipedia"),
}


class NativeMedRGAGBM25:
    name = "medrgag_native_BM25"

    def __init__(self, *, medrgag_root: str | Path,
                 corpus_dir: str | Path | None = None,
                 corpus_name: str = "MedText"):
        if corpus_name not in _CORPORA:
            raise ValueError(f"Unsupported original MedRGAG corpus {corpus_name}")
        root = Path(medrgag_root)
        db = Path(corpus_dir) if corpus_dir else root / "corpus"
        for name in _CORPORA[corpus_name]:
            chunk = db / name / "chunk"
            index = db / name / "index" / "bm25"
            if not chunk.is_dir() or not index.is_dir():
                raise FileNotFoundError(
                    f"Missing original local BM25 corpus/index: {chunk} / {index}. "
                    "Refusing automatic corpus download or indexing. Provide "
                    "--corpus-dir with the pre-existing MedRGAG corpus.")
        utils_path = root / "src" / "utils.py"
        if not utils_path.is_file():
            raise FileNotFoundError(f"Original MedRGAG utilities absent: {utils_path}")
        spec = importlib.util.spec_from_file_location(
            "_emrag_medrgag_native_retrieval", utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load original retrieval: {utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.system = module.RetrievalSystem(
            "BM25", corpus_name, str(db), cache=False, HNSW=False)
        self.calls = 0
        self.corpus_name = corpus_name
        self.corpus_dir = str(db)

    def search(self, query: str, *, k: int) -> list[Passage]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if not isinstance(k, int) or k < 1:
            raise ValueError("Retrieval k must be positive")
        self.calls += 1
        # NOTE: original RetrievalSystem obtains k PER CORPUS. With MedText
        # (textbooks + Wikipedia), there may be up to 2*k raw candidates.
        raw, scores = self.system.retrieve(query, k=k, id_only=False)
        out = []
        if len(raw) != len(scores):
            raise ValueError("MedRGAG returned mismatched documents and scores")
        for i, (row, score) in enumerate(zip(raw, scores)):
            if not isinstance(row, dict) or not isinstance(row.get("contents"), str):
                continue
            if not row["contents"].strip():
                continue
            out.append(Passage(
                id=f"new{self.calls}_{i}",
                text=row["contents"],
                origin="retrieved",
                original_id=str(row.get("id", "")),
                title=str(row.get("title", "")),
                score=float(score)))
        return out
