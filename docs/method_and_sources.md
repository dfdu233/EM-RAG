# EM-RAG: method, evidence standards, and implementation genealogy

This document maps the EM-RAG implementation to **specific public source files**, rather than presenting a novel end-to-end algorithm as a verbatim reproduction of a prior paper. We designed the Python package independently and import official MedRGAG code only through its retrieval interface.

## 1. What exactly comes from MedRGAG (WWW 2026)?

Paper: [From Retrieval to Generation: Unifying External and Parametric Knowledge for Medical Question Answering](https://arxiv.org/abs/2510.18297)

Official repo: https://github.com/ll0ruc/MedRGAG

Verified source paths:
- [src/utils.py](https://github.com/ll0ruc/MedRGAG/blob/master/src/utils.py): RetrievalSystem builds one Retriever per configured corpus and returns (texts, scores) from retrieve(question, k). Under the hood it uses Pyserini BM25.
- [src/medrgag.py](https://github.com/ll0ruc/MedRGAG/blob/master/src/medrgag.py): MedRGAG.retrieval(), rank_snippets(), generate_knowledge(), and get_prompt(). These connect initial retrieval, cross-encoder ranking, generated knowledge and answering.
- [src/llm_by_gpt.py](https://github.com/ll0ruc/MedRGAG/blob/master/src/llm_by_gpt.py): summary / explore / selection modes correspond to retrieval grounding, Knowledge-Guided Context Completion (KGCC), and Knowledge-Aware Document Selection (KADS).
- [src/evaluate_utils.py](https://github.com/ll0ruc/MedRGAG/blob/master/src/evaluate_utils.py): QADataset loads benchmark.json, with an id, question, options and answer.

EM-RAG mappings:
- emrag/retrieval.py::MedRGAGRetriever.search calls the SAME live RetrievalSystem.retrieve on **fresh round-specific query strings**.
- emrag/retrieval.py::MedRGAGSeedRetriever reads the cached first-turn snippets, but refuses to fabricate further retrievals when a live index is absent.
- emrag/retrieval.py::load_kgcc_hints reads generated snippets *as unverified search hints only*. Model-generated KGCC content cannot count toward gate support.
- emrag/cli.py::load_benchmark accepts the MedRGAG benchmark schema and strictly separates gold labels from inference.

**Upstream issues that this integration avoids:** hard-coded model path placeholders, mixed model/ranker environment, missing external config and local environment assumptions. This repo does not claim upstream artifacts are directly executable without preparing their original dependencies. The upstream retrieval internals should be audited with untrusted indexes; EM-RAG's own JSONL retriever does not use eval.

## 2. Multi-round query generation: MA-RAG (ICML 2026)

Paper: [From Conflict to Consensus](https://proceedings.mlr.press/v306/wu26ba.html)

Official code: https://github.com/NJU-RL/MA-RAG/blob/main/ma_rag_entropy.py

The official entropy variant checks whether diverse answers share a prediction, and generates new queries from answer disagreement when they do not. EM-RAG borrows the **scientific question** of targeted follow-up search, but not its implementation or entropy-as-quality assumption. In emrag/prompts.py::SYSTEM_PLAN, history consists of *hypotheses*, never automatically verified knowledge. The engine retains an original-question anchor.

The independent source-supported answer-update gate is a **proposed EM-RAG component**, not an existing MA-RAG algorithm.

## 3. Missing facts and evidence sufficiency: S2G-RAG (ACL 2026)

Paper: [Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA](https://aclanthology.org/2026.acl-long.1185/)

Official code:
- https://github.com/nianaaa/S2G-RAG/blob/main/inference/inference_bm25.py
- https://github.com/nianaaa/S2G-RAG/blob/main/training/train_s2g_judge_lora.py

Their inference includes build_suff_user_prompt, call_suff_gate_batch, and structured missing fact extraction. Their released method **trains** the sufficiency judge (see LoRA trainer); only its principle of structured missing-fact search is borrowed by the frozen prompt policy in emrag/prompts.py and emrag/engine.py::_plan.

EM-RAG does NOT claim identical performance to a trained S2G-Judge.

## 4. Contrasting hypotheses: CF-RAG (ICLR 2026)

Paper: [Counterfactual Reasoning for Retrieval-Augmented Generation](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1c078897dc08d46091d0d361d9955c6b-Abstract-Conference.html)

Official repo: https://github.com/CF-RAG/CF-RAG/blob/main/cfrag_pipeline.py

Their candidate arbitration combines consistency, causal-proxy and completeness scores. The published counterfactual retrieval mechanism means **counterfactual queries alone are not a novel contribution**. EM-RAG uses only a lightweight query-planning request for *discriminating evidence*; it does not reproduce CF-RAG's complete clustering/parallel arbitration or its learned components.

## 5. Judge-scoring: transparent policy, not ground-truth oracle

Source-related papers:
- [Sufficient Context (ICLR 2025)](https://github.com/hljoren/sufficientcontext)
- [S2G-RAG](https://github.com/nianaaa/S2G-RAG)
- [CF-RAG](https://github.com/CF-RAG/CF-RAG)

EM-RAG's score is explicitly an *untrained ordinal heuristic*:

    J(a,E) = 2 * support_count - 4 * refute_count - missing_count

The prompt-based frozen judge returns supported/refuted evidence IDs, missing requirements and a binary sufficiency flag. emrag/core.py::sanitize_card rejects IDs absent from the current quote ledger. emrag/engine.py::_judge evaluates all options on the *same evidence set*. A candidate is accepted only if its evidence is sufficient, no essential contradiction remains and its margin beats other options.

Limitations:
- A source quote may be false, irrelevant, stale, or misinterpreted.
- A valid quote does not establish that its extracted paraphrase is entailed.
- A frozen generative judge is not independent of the generator; the two can share errors.
- The ordinal score is not a probability and the margin is not a calibrated risk bound.
- If omitted retrieved passages contain contradictions, a fixed active window can hide them.

Therefore, no guarantee of step-by-step truth monotonicity is asserted. The proposal should be judged empirically by reduced *Harm* without suppressing *Rescue*, at controlled budgets.

## 6. Evaluation derivation

For an example i, let correct_i(t) be 1 if the emitted answer at round t matches gold, else 0. Define unconditional rates over N cases:

    H_t = (1/N) sum_i 1[ correct_i(t-1)=1 and correct_i(t)=0 ]
    R_t = (1/N) sum_i 1[ correct_i(t-1)=0 and correct_i(t)=1 ]

Then, counting abstention as incorrect:

    Acc_t - Acc_(t-1) = R_t - H_t

This identity has no statistical assumptions, but does not guarantee R_t >= H_t. Implemented in emrag/metrics.py::summarize with a unit test.

Report BOTH conditional harm/rescue rates and unconditional transition rates. If abstention is enabled, report answer coverage and selective accuracy. A best-prefix oracle is evaluated offline using gold and is an **upper bound only**, not a deployable policy. Always compare at controlled search, document, model-call and preferably token budgets.

## 7. Proposed next research work (not implemented / not claimed)

1. Proper source-level uncertainty calibration and **selective risk coverage** bounds (requires data-dependent calibration, although the backbone may remain frozen).
2. Stronger fact-level verifier for negation, patient context, temporal validity, provenance and conflicting clinical recommendations.
3. Paired same-prefix perturbation experiments separating the influence of retrieved evidence from parametric prior.
4. Retrieval with a larger indexed medical corpus, robust adversarial noise splits, repeated seeds and external-model comparisons.
5. Multi-document corroboration that avoids treating correlated passages from one provenance as independent evidence.

The implementation should be described as a **testable starting point**, not final evidence of a state-of-the-art medical RAG gain.
