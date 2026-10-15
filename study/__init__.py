"""Mechanism-only experiments. Never import this package in EM-RAG inference.

The study scripts keep gold labels out of retrieval/planning/Reader prompts and
treat every claimed gain as development evidence until held-out confirmation.
"""
__version__ = "0.1.0"
