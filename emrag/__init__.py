"""Training-free evidence-audited multi-round RAG."""
from .core import Question, Document, Evidence, EvidenceLedger, ScoreCard, RoundTrace, RunResult
from .engine import EMRAG, EMRAGConfig

__all__ = ["Question", "Document", "Evidence", "EvidenceLedger",
           "ScoreCard", "RoundTrace", "RunResult", "EMRAG", "EMRAGConfig"]
__version__ = "0.1.0"
