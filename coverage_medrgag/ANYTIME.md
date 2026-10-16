# Training-Free Anytime Evidence Selection (TF-AES)

A minimal, falsifiable, training-free experiment based on
[MedRGAG (WWW 2026)](https://github.com/ll0ruc/MedRGAG).
The new code is isolated; all historical EM-RAG Gate, GAG, and
Coverage-Adaptive branches remain unchanged.

## Mechanism

Use original cached MedRGAG top-32 (MedCPT-ranked) source documents, five
KGCC generated passages, historical original KADS selection, final MedCPT
reranking and frozen Reader with its original prompt.

At K=5, retain the original KADS-selected five documents EXACTLY.
At K=10,20,32, propose a different five-document combination using fixed
question-conditioned evidence coverage. Keep the previously selected set
unless the new proposal scores strictly higher. One Reader answer at the end.

No additional trained model, auxiliary LLM, answer Gate, or generated
background calls. The default MedCPT utility uses the already-frozen
original MedCPT weights, not newly trained weights.

Let F(q) be a set of fixed question facets extracted from the ORIGINAL
question and choices before retrieval. Let sim(f,d) be a fixed nonnegative
facet-to-document similarity, bounded by [0,1].

    J(q,C) = mean over f in F(q) of max over d in C of sim(f,d)

J is monotone and submodular as a set function of C. A greedy cardinality-5
selection proposes new sets; an incumbent memory compares every proposal
against the previous winner using the SAME fixed scoring function. Thus
the retained score J_best(K) is rigorously nondecreasing as source K increases.

CRITICAL: the score is NOT P(correct). Its guaranteed nondecreasing property
DOES NOT guarantee correctness, medical entailment, monotonic answer accuracy,
or a lower error risk. Incorrect documents may score higher; this experiment
can disprove the usefulness of J, and must not be fitted to benchmark labels.

Two score backends:
- lexical: feature coverage, no GPU or extra scorer weights, useful control
- medcpt: sigmoid of frozen MedCPT Cross-Encoder logits for each
  (question facet, candidate passage). It is an uncalibrated similarity,
  NOT a probability of truth. Reuses MedRGAG's final reranker weights.

## Installation

~~~bash
git fetch origin
git switch feat/training-free-anytime-context
python -m pip install -e '.[all]'
python -m pytest -q
~~~

Use original MedRGAG corpus artifacts and local frozen Reader server
(Qwen2.5-7B, Llama3.1-8B, Ministral-8B). Use identical model checkpoints,
datasets, KADS/KGCC caches, tokenizer and temperature for all paired arms.

~~~bash
MED=../MedRGAG
DATA=pubmedqa
READER=qwen2.5
API=http://127.0.0.1:8000/v1
MANIFEST=/path/to/frozen_development_manifest.jsonl
~~~

### A. Start with the zero-extra-model scorer

~~~bash
python -m coverage_medrgag.anytime_cli run \
  --medrgag-root "$MED" --dataset "$DATA" \
  --ids-file "$MANIFEST" \
  --reader-model "$READER" --api-base "$API" --api-key EMPTY \
  --reader-tokenizer /path/to/Qwen2.5-7B-Instruct \
  --utility lexical --depths 5 10 20 32 \
  --max-new-examples 40 \
  --output "outputs/tf-aes/$DATA/lexical"
~~~

This still uses the ORIGINAL frozen MedCPT for final reranking. To omit
MedCPT in unit-test-only mode, use --no-rerank; that breaks MedRGAG parity.

### B. Reuse MedCPT as the frozen facet scorer

~~~bash
python -m coverage_medrgag.anytime_cli run \
  --medrgag-root "$MED" --dataset "$DATA" \
  --ids-file "$MANIFEST" \
  --reader-model "$READER" --api-base "$API" --api-key EMPTY \
  --reader-tokenizer /path/to/Qwen2.5-7B-Instruct \
  --utility medcpt --ranker-model ncbi/MedCPT-Cross-Encoder \
  --depths 5 10 20 32 --max-new-examples 40 \
  --output "outputs/tf-aes/$DATA/medcpt"
~~~

Original KADS selection cache must exist in
MedRGAG/outputs/DATA/docs/generate/snippets_final_select.json.
The run also requires original retrieval snippets, generated snippets,
benchmark.json, and src/template.py.

### C. Measure whether the monotone score predicts REAL correctness

Use --read-each-depth only on a pre-frozen small DEV subset, with a NEW
output path. It calls Reader once at each prefix K instead of once at the end.

~~~bash
python -m coverage_medrgag.anytime_cli run \
  --medrgag-root "$MED" --dataset "$DATA" \
  --ids-file "$MANIFEST" \
  --reader-model "$READER" --api-base "$API" --api-key EMPTY \
  --reader-tokenizer /path/to/Qwen2.5-7B-Instruct \
  --utility lexical --read-each-depth \
  --depths 5 10 20 32 --max-new-examples 40 \
  --output "outputs/tf-aes/$DATA/curve"

python -m coverage_medrgag.anytime_cli evaluate \
  --benchmark "$MED/benchmark.json" --dataset "$DATA" \
  --predictions "outputs/tf-aes/$DATA/curve/predictions.jsonl" \
  --include-prefix-curve --bootstrap 1000 \
  --output "outputs/tf-aes/$DATA/curve/report.json"
~~~

The report tests both sides of the scientific question:
(a) score improvement (which is guaranteed by the memory) and
(b) actual error-to-correct and correct-to-error transitions (NOT guaranteed).
Any case with increasing score but declining accuracy directly falsifies a
stronger decision-quality guarantee. The tests intentionally include such
a case.

### D. Paired reference comparison

~~~bash
python -m coverage_medrgag.anytime_cli evaluate \
  --benchmark "$MED/benchmark.json" --dataset "$DATA" \
  --predictions "outputs/tf-aes/$DATA/medcpt/predictions.jsonl" \
  --compare /path/to/same-reader-reference/predictions.jsonl \
  --bootstrap 1000 \
  --output "outputs/tf-aes/$DATA/medcpt/report.json"
~~~

You can also use --original-prediction /path/to/MedRGAG/prediction.json
for DESCRIPTIVE historical comparison, subject to exact checkpoint and
prompt parity. Match question IDs, splits, Reader, and all caches. Avoid
selecting a preferred score backend by looking repeatedly at held-out tests.

## Research safeguards

- The cached top-32 results were retrieved/reranked beforehand; progressive
  exposure is NOT a live indexing-cost saving and is NOT iterative query
  generation. Evaluate live retrieval only if this simpler idea works.
- Reusing original MedRGAG KGCC/KADS cached results does not remove the
  preprocessing cost. Report it for complete end-to-end comparisons.
- MedCPT facet scoring can require many extra GPU forward passes. Compare
  elapsed time and total cost against MedRGAG, not only LLM API call count.
- The fixed score is a structural proxy. It does not verify true medical
  claims or conditional facts. Do not call it S(K), U(K) or V(K).
- Test whether newly adopted documents actually account for new correct
  answers. Compare against original KADS, fixed K5, and (when available)
  fixed K32 context selection with identical scorer/Reader budgets.
- Run three Readers and five original MedRGAG datasets only after a small
  development check indicates a meaningful signal; preserve a fresh final
  confirmatory holdout.

ABANDON this proxy, do not stack heuristics, if score improvement does not
predict answer improvement; if gains vanish across Readers/datasets; or if
extra MedCPT scoring cost overwhelms small accuracy changes.

## Research provenance / novelty limits

- MedRGAG WWW 2026 https://github.com/ll0ruc/MedRGAG
- Long-Context LLMs Meet RAG ICLR 2025
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/5df5b1f121c915d8bdd00db6aac20827-Abstract-Conference.html
- Inference Scaling for Long-Context RAG ICLR 2025
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/b57471758783450408f5c9b70257d15b-Abstract-Conference.html
- REPLUG NAACL 2024 https://aclanthology.org/2024.naacl-long.463/
- Adaptive Submodularity JAIR 2011
  https://mlanthology.org/jair/2011/golovin2011jair-adaptive/
- Best-of-N Reward Hacking ICML 2025
  https://proceedings.mlr.press/v267/huang25c.html

Classic greedy submodular coverage and frozen retriever reranking are NOT
new in isolation. This is a minimal experiment of whether stable
decision-relevant progress can arise without a trained scorer.
