# PSV v2 — Reliable Evidence and Matched Source Controls

This independent research branch adds **query-length repair**, **uniform
gold-free answer recovery**, **KGCC-free source anchors**, **five-document
matched source controls**, and **exact selected-source replay** to
[Progressive Source Views](coverage_medrgag/PROGRESSIVE.md).

See the complete [PSV v2 experiment protocol](coverage_medrgag/PSV_V2.md).

These changes have NOT been validated as improving biomedical accuracy.
Replayed sources do not count as NEW BM25 searches and do not demonstrate
a new inference-scaling law. The original PSV branch and main are unchanged.

---

# Progressive Source Views: Active MedRGAG Scaling Diagnostics

This independent branch implements a training-free, testable RAG-GAG
progressive-retrieval experiment. It uses genuine new BM25 searches,
frozen Reader answer disagreement feedback, the original cached KGCC
and KADS as its exact budget-zero reference, and source-view answering.

See [protocol, limitations, ablations, and server commands](coverage_medrgag/PROGRESSIVE.md).
It is NOT a validated SOTA method and has NO clinical correctness guarantees.
Previous negative-result branches and the main branch are not modified.

---

# Training-free Anytime Evidence Selection (experimental branch)

This branch adds an independent, training-free MedRGAG extension under
[coverage_medrgag/ANYTIME.md](coverage_medrgag/ANYTIME.md).
It retains the original KGCC, KADS K5 anchor, MedCPT ranking, and frozen
Reader. A fixed document-coverage surrogate is optimized incrementally
over K=5/10/20/32 using best-context memory. No trained module, auxiliary
LLM sufficiency check, or answer-update Gate is required.

**Only the estimated evidence-coverage surrogate is guaranteed to be
nondecreasing, NOT real Reader answer correctness.** The optional
per-depth answer experiment tests this difference directly. The default
main branch and all prior feature branches remain unchanged.

---

# Coverage-Adaptive MedRGAG — NEW experimental research branch

**This branch is not a replacement for the EM-RAG v0.1/v0.2 history.**
The NEW minimal research prototype lives under
[coverage_medrgag/](coverage_medrgag/README.md).

It retains MedRGAG's full precomputed KGCC generation, original KADS
selection template, original MedCPT Cross-Encoder re-ranking and original
Reader prompt. Only a source-only evidence coverage signal controls how
deeply into the ranked retrieval cache to inspect (K=5/10/20/32).
One final Reader answer, no answer-update Gate, no additional model training.

Three built-in modes: official precomputed KADS reference, live K5 KADS
static control, and coverage-adaptive retrieval depth. Gold labels are
isolated in a separate evaluator with explicit S(K)/U(K)/V(K) limitations.
See [server commands and scientific controls](coverage_medrgag/README.md).

All changes are isolated from the main branch. **No new benchmark scores
or SOTA results have been established by this code release.**

---

# EM-RAG

**Evidence-Monotonic RAG (EM-RAG)** is a **training-free research prototype** for making iterative retrieval more reliable as the number of search rounds grows. It integrates with the [MedRGAG (WWW 2026)](https://github.com/ll0ruc/MedRGAG) retrieval/data format.

> **Scope and guarantee:** EM-RAG protects a *source-quote-grounded proxy decision process*; it cannot guarantee that every extra search step increases the **true** accuracy of a frozen black-box LLM. A verbatim quote is NOT proof that a source is scientifically correct, relevant to the specific patient, or that a model's paraphrase is entailed. Neither the model judge nor its score is calibrated.

**Research-only, not for patient care.** No trained gate, value network, specialized judge, or fine-tuning is needed.

### v0.2: mechanism-first diagnostics (NO SOTA claim)

Reported preliminary results on 500 cases (provided by the experimenter): EM-RAG **50.2%**, same-prompt initial Reader **50.4%**, MedRGAG **50.8%**. The current gate reportedly blocked **60 harmful proposals** and **67 beneficial corrections**, showing no verified aggregate gain. Before modifying the gate or expanding testing, use the **original cached traces** to separately inspect evidence availability, correct candidate generation and gate discrimination.

- [Mechanism diagnosis playbook](docs/mechanism_diagnosis.md) — zero-GPU retrospective audit, fixed stratified casebook, independent evidence sufficiency annotation, optional frozen Reader probes.
- [Offline diagnostic core](emrag/diagnostics.py) and [diagnostic CLI](emrag/diagnose_cli.py) — scores and decisions reconstructed without further model calls.
- [Reader probe](emrag/reader_probe.py) — opt-in, hard-budget, paired tests on saved active quotes, all quotes and available raw passages. No gold is sent to the Reader.
- [Engine instrumentation](emrag/engine.py) — per-round rejection reason traces and optional raw source passage snapshots via --record-documents.

Example first action on **existing** cached 500-question trajectories (replace paths/dataset):

~~~bash
python -m emrag.diagnose_cli \
  --traces outputs/pubmedqa-emrag/traces.jsonl \
  --benchmark ../MedRGAG/benchmark.json --dataset pubmedqa \
  --per-stratum 6 --seed 2026 \
  --output outputs/mechanism-dev
~~~

This does **not** retrain or rerun EM-RAG and does not infer factual sufficiency from the model judge. If older traces lack source passages, use --medrgag-first-turn-cache for an explicitly marked **first-round-only** view; missing later passages remain unknown. Evaluation stays paused until a causal bottleneck is established on a development sample.
 The same frozen model can be used for planning, extraction, generation and verification, though their errors may remain correlated.

## What is implemented?

The question stays fixed while an independent evidence ledger grows:

1. Produce a **round-0 parametric answer** without retrieval.
2. Plan new retrieval queries around missing necessary facts and competing hypotheses; the **first query is always the original question**.
3. Retrieve fresh passages, deduplicate them and extract *atomic claims with exact verbatim quotes*. Reject fabricated quotes and invented document identifiers.
4. Store only retrieved sources as evidence. MedRGAG **KGCC-generated documents can provide query hints only**; generated text is never promoted to verified external evidence.
5. Generate a new answer and evaluate **all answer options on the same current evidence context**. Check supported/refuted citations, missing facts, and evidence sufficiency.
6. Accept a new choice only if it has direct support, no cited refutations/missing requirements, and a score margin against competing options. Otherwise retain the previous choice; if an already verified choice becomes refuted, abstain rather than silently preserve it.
7. Save a full per-round trace; report Harm (correct -> wrong), Rescue (wrong -> correct), coverage, verified coverage, model calls, and oracle best-prefix **as an evaluation-only upper bound**.

The score is an ordinal heuristic, **not** a calibrated correctness probability. Source-grounded quotations are still interpreted by a fallible frozen language model:

~~~text
score(a, E) = 2 * |support_ids| - 4 * |refute_ids| - |missing_facts|

accept(a) iff:
  a has appeared in the candidate answer pool
  and judge says evidence is sufficient
  and at least one external source quote directly supports a
  and no cited refutations or unresolved necessary facts remain
  and score(a) - max_score(other options) >= min_score_margin
~~~

All model-generated citations are filtered against actual quote-grounded evidence IDs. Evidence is selected into a fixed-size active context and previously cited support is prioritized. Original retrieval passages and evidence claims are retained separately; the judge can still be mistaken.

## Installation

~~~bash
git clone https://github.com/dfdu233/EM-RAG.git
cd EM-RAG
python -m pip install -e ".[all]"
python -m pytest -q
~~~

Basic retrieval and the test suite use only Python's standard library plus pytest. GPU dependencies are deliberately **not** installed by default. MedRGAG/Pyserini requires its own environment and medical corpora. The API adapter requires the optional OpenAI Python client.

## 1. Small offline corpus / smoke inference

A fictional QA sample is included in examples/. It is **not a medical accuracy benchmark**.

Start an OpenAI-compatible LLM endpoint, e.g. on a CUDA server with vLLM installed:

~~~bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen --gpu-memory-utilization 0.80
~~~

Then:

~~~bash
python -m emrag.cli \
  --benchmark examples/synthetic_benchmark.json --dataset synthetic \
  --jsonl-corpus examples/synthetic_corpus.jsonl \
  --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --max-examples 1 --max-rounds 3 --queries-per-round 2 \
  --output outputs/synthetic
~~~

The generation model must emit JSON compatible with the prompts. End-to-end performance depends on the real model and the source collection, not on the unit-test fixtures.

## 2. Full MedRGAG corpus and benchmark

Clone and prepare the **upstream** repository, corpus and Pyserini index separately, following its documentation. The upstream Python environment includes heavy components (Pyserini, transformers, FAISS etc.) and may attempt to download/index corpora automatically.

~~~bash
git clone https://github.com/ll0ruc/MedRGAG.git ../MedRGAG

python -m emrag.cli \
  --benchmark ../MedRGAG/benchmark.json --dataset medqa \
  --medrgag-root ../MedRGAG \
  --db-dir ../MedRGAG/corpus --corpus-name MedText \
  --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --max-examples 20 --max-rounds 4 --queries-per-round 2 \
  --docs-per-query 5 --max-context-chars 9000 \
  --update-policy evidence \
  --output outputs/medqa-emrag
~~~

MedRGAG integration maps the upstream interface:

~~~python
# MedRGAG: src/utils.py::RetrievalSystem.retrieve(question, k)
snippets, scores = retrieval_system.retrieve(query, k=k)

# EM-RAG: emrag/retrieval.py::MedRGAGRetriever.search(query, k)
# => list[Document] with source, content, title and score
~~~

It does **not** copy the upstream MedRGAG application script, its hard-coded local model paths, or its unused API tokens. Use only trusted local MedRGAG indexes, because the original index/document lookup implementation should be audited before indexing untrusted corpora.

### Reuse MedRGAG's precomputed snippets / KGCC output

For an existing experiment directory, you can warm-start the first search with MedRGAG's retrieved JSON and use the real corpus for later rounds:

~~~bash
python -m emrag.cli \
  --benchmark ../MedRGAG/benchmark.json --dataset medqa \
  --seed-retrieval ../MedRGAG/outputs/medqa/docs/retrieval/snippets.json \
  --live-medrgag-root ../MedRGAG --db-dir ../MedRGAG/corpus \
  --kgcc-hints ../MedRGAG/outputs/medqa/docs/generate/snippets.json \
  --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --max-rounds 4 --output outputs/medqa-warm-start
~~~

Using only --seed-retrieval without --live-medrgag-root has **no real subsequent search**; the CLI explicitly warns against treating it as a multi-round scaling experiment. KGCC text is *untrusted, optional search expansion*, not source-backed evidence.

## 3. Matched-budget ablation

Use the same configured maximum rounds, queries, documents, model and dataset for the following three policies. These are **matched-budget ceilings**, not guaranteed identical queries or actual call counts: active evidence and planning can diverge, and early stopping can differ. Report the realized call counts from traces:

| Flag | Answer update rule | Intended control |
|---|---|---|
| --update-policy evidence | evidence-supported / conflict-aware acceptance | **EM-RAG** |
| --update-policy latest | always use latest valid generated choice | iterative RAG without update guard |
| --update-policy frozen | never change round-0 answer | parametric frozen control |

Example (execute each policy on the **same** questions, retrieval index and model):

~~~bash
for policy in evidence latest frozen; do
  python -m emrag.cli \
    --benchmark ../MedRGAG/benchmark.json --dataset medqa \
    --medrgag-root ../MedRGAG --db-dir ../MedRGAG/corpus \
    --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
    --max-examples 100 --max-rounds 6 --queries-per-round 2 \
    --update-policy "$policy" --output "outputs/medqa-$policy"
done
python scripts/compare_runs.py outputs/medqa-{evidence,latest,frozen}/metrics.json
~~~

Do **not** compare an abstention-enabled policy only by its selective accuracy: report coverage and overall accuracy together. --unverified-policy abstain forces abstention if no evidence-supported verified answer survived. It is mainly intended for risk-coverage experiments.

## 4. Output and metrics

Each run writes:

- run_config.json: selected dataset and inference/retrieval configuration, excluding API credentials.
- Optional raw source passages: only with the explicit --record-documents flag; copyrighted/private passages should remain outside version control.
- Optional rejection reasons in each round, for directly diagnosing false reject/false accept mechanisms.
- traces.jsonl: one record per question (round-0 baseline, queries, passage IDs, grounded evidence, new proposals, support/refutation scorecards, accepted/preserved/invalidated actions).
- metrics.json: per-round accuracy, unconditional and conditional harm/rescue rates, delta accuracy, coverage, verified coverage, mean model calls, and oracle best-prefix accuracy.

For a common fixed answer domain, with abstention counted as incorrect:

~~~text
Accuracy(t) - Accuracy(t-1) = RescueRate(t) - HarmRate(t)
~~~

Rescue/Harm in this identity are **unconditional fractions of all examples**, not their conditional rates among previously incorrect/correct examples. If a question ends early, its latest answer is carried forward for budget-curve evaluation.

Accuracy uses gold labels from the evaluation loader **only after** engine.run returns. They never enter the retrieval planner, the generator or the judge.

## Code layout

~~~text
emrag/
  core.py       Question/Document/Evidence, exact quote checking, typed traces
  retrieval.py  JSONL BM25; MedRGAG live and cached seed; KGCC query hints
  prompts.py    frozen JSON prompts (planner/extractor/generator/judge)
  model.py      OpenAI-compatible frozen LLM client; deterministic fixture model
  engine.py     multi-round loop, candidate re-evaluation, conservative gate
  metrics.py    scaling, harm/rescue, coverage, upper bound
  cli.py        MedRGAG benchmark integration and experiment runner
tests/          unit + synthetic end-to-end checks (no GPU / API)
examples/       explicitly fictional corpus and QA
docs/           source-to-algorithm cross-references and limitations
~~~

## Scientific limitations and future validation

- **Not monotonic truth:** preserved evidence, no invented quote, and deterministic update rules are executable invariants; no guarantee of increasing real-world accuracy. A source may be outdated, biased, false, or wrong for the patient. An LLM judge can still misinterpret exact quoted sentences.
- **Correlated errors:** planning, answer generation and verification can share one backbone; agreement is not an independent correctness oracle. Consider multi-backbone verification and calibration only as separately reported ablations.
- **Context truncation:** protected citations receive priority, but the fixed budget may still omit an important new contradiction. Report evidence recall, refutation detection and context-budget sensitivity.
- **Retriever differences:** built-in JSONL BM25 is a tiny-corpus reference, not performance-equivalent to MedRGAG's BM25/Pyserini and optional cross-encoder. Large-scale claims should use the same official corpus and index across policies.
- **No empirical claims yet:** passing Python tests does not establish performance on MedQA, MedMCQA or MedXpertQA. Report scaling curves across budgets, prompt sensitivity and fixed model-call/token budgets before making paper claims.

See [docs/method_and_sources.md](docs/method_and_sources.md) for original-source algorithm correspondence, nearby papers and reproducibility caveats.

## Citation and licensing

Original EM-RAG implementation is released under MIT in this repository. It references external projects by API and paper, and does not vendor their licensed code or data. Please separately cite and comply with MedRGAG and the other research projects as appropriate.
