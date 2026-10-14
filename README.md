# EM-RAG

**Evidence-Monotonic RAG**: a training-free, evidence-audited multi-round retrieval-augmented generation research implementation.

EM-RAG tests whether protecting previously supported answers, grounding claims in retrieved quotations, and searching for unresolved discriminating facts improve the **harm/rescue trade-off** as retrieval budget grows. It does **not** claim a theoretical guarantee of monotonic factual accuracy.

The implementation provides adapters for [MedRGAG](https://github.com/ll0ruc/MedRGAG), a no-dependency BM25-style JSONL retrieval baseline, and OpenAI-compatible model serving (including local vLLM). Structured traces, per-round harm/rescue metrics, and offline unit tests are included. See the detailed usage instructions further below after the code is installed.

**Status:** research implementation, not a validated clinical tool; do not use for patient care.
