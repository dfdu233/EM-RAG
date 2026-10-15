# Coverage-Adaptive MedRGAG (CA-MedRGAG) — research prototype

**Goal:** improve retrieval-budget scaling while retaining the FULL external
retrieval + parametric KGCC + KADS + frozen Reader foundation of
[MedRGAG (WWW 2026)](https://github.com/ll0ruc/MedRGAG).
This is a *hypothesis*, NOT established SOTA or a completed benchmark result.

## Scope of the proposed change

Core idea, one sentence: **expand ranked retrieved evidence only when a
source-only check cannot establish that it covers the question's necessary
facts; keep MedRGAG's KGCC+KADS and make one final Reader call.**

- Original MedRGAG: retrieve/rerank top candidates + precomputed KGCC
  generated knowledge + KADS select 5 from 5 retrieved and 5 generated
  + MedCPT rerank selected items + Reader.
- CA-MedRGAG: additionally inspect retrieval prefixes at K=5,10,20,32;
  stop when a conservative quotation-grounded sufficiency PROXY is met or
  the budget ends; choose 5 source candidates from the inspected pool;
  feed these plus the ORIGINAL five KGCC candidates into MedRGAG's KADS;
  MedCPT reranks the five selected documents; one Reader call.
- There is **NO answer-update Gate**; this is not v0.1 EM-RAG with more
  controller blocks. There is no iterative query planner or new trained
  module. It scales *retrieval depth*, not iterative query count.
- Training: none in the first phase; a trained source-sufficiency observer
  may later be studied ONLY if evidence shows its error is limiting.

The original KGCC document cache is required because generating *new*
background knowledge online would confound the source-budget experiment.
Their upstream generation cost still counts for end-to-end comparisons.

## Relation to S(K), U(K), V(K)

Define:
- S(K) = P(the retrieved sources INDEPENDENTLY contain adequate facts)
- U(K) = P(Reader correct | adequate facts)
- V(K) = P(Reader correct | inadequate facts)

Then Accuracy(K) = S(K)*U(K) + (1-S(K))*V(K).

**S, U and V require independent evidence annotations and gold labels for
OFFLINE measurement. They are NOT model inputs, calibrated per-instance
probabilities, or learned reward signals in this prototype.**

At inference, the coverage observer produces only a FALLIBLE source-only
sufficiency proxy. To label evidence 'sufficient', it must include a valid
answer label, no remaining declared missing facts, and at least one exact
quote from a seen retrieval source. Verbatim matching stops fabricated IDs
but does not establish medical entailment. Proxy error can cause premature
stopping; the design offers no monotonic-accuracy guarantee.

When source evidence appears insufficient, the original KGCC documents
remain available to MedRGAG KADS/Reader, preserving PARAMETRIC background
as a fallback path. V(K) is evaluated offline, not directly optimized
through a per-question oracle or answer update gate.

### Official code correspondence

- Retrieval cache ranked by original MedCPT Cross-Encoder:
  MedRGAG main.py: retrieved snippets 32 per question,
  rank_snippets(mode="score").
- KGCC generated cache: outputs/DATASET/docs/generate/snippets.json.
- KADS baseline cache: snippets_final_select.json. Its final_list indexes
  the first five retrieved followed by the first five generated.
- MedCPT rerank selected: original MedRGAG rank_snippets(mode="select").
- Reader: exact rag_template from original MedRGAG src/template.py, temperature
  0.1, seed 42, max new tokens 512. With --reader-tokenizer, the original
  context-token truncation is also reproduced.

All original original-GAG and KADS data come from precomputed MedRGAG files.
The new live KADS selection uses the SAME original KADS prompt, but may
differ from cached selections because its model, temperature and truncation
may differ. **Use static vs adaptive to isolate the NEW budget strategy**.
The reference mode is a parity diagnostic, not a guaranteed bit-exact
reproduction across vLLM and OpenAI-compatible backends.

## Install and prerequisites

Branch: feat/coverage-adaptive-medrgag.

~~~bash
git fetch origin
git switch feat/coverage-adaptive-medrgag
python -m pip install -e '.[all]'
python -m pytest -q
~~~

For the actual original MedCPT reranker, install PyTorch and transformers
in the SAME validated environment as MedRGAG; weights can be local or the
public ncbi/MedCPT-Cross-Encoder. Do not assume it can run in bare CI.

Serve the original Reader (Qwen2.5-7B, Llama3.1-8B or Ministral-8B) from a
local OpenAI-compatible vLLM endpoint. Use the SAME model checkpoint, seed,
prompt, tokenizer and dataset split for paired studies.

~~~bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5 --gpu-memory-utilization 0.80
~~~

Paths below are conventional MedRGAG locations; confirm your ACTUAL cache
files on server before running.

~~~bash
MED=../MedRGAG
BENCH="$MED/benchmark.json"
API=http://127.0.0.1:8000/v1
DATA=medqa
~~~

### 1. Reference: original KGCC and KADS selection cache

~~~bash
python -m coverage_medrgag.cli run \
  --medrgag-root "$MED" --dataset "$DATA" \
  --mode reference \
  --reader-model qwen2.5 --reader-tokenizer /path/to/Qwen2.5-7B-Instruct \
  --api-base "$API" --api-key EMPTY \
  --ranker-model ncbi/MedCPT-Cross-Encoder \
  --limit 20 --output "outputs/ca-medrgag/$DATA/reference"
~~~

This uses MedRGAG's own selection list and reranks its selected passages
with MedCPT. It does NOT regenerate KGCC or re-run upstream KADS.

### 2. Fixed baseline: K5 with original live KADS template

~~~bash
python -m coverage_medrgag.cli run \
  --medrgag-root "$MED" --dataset "$DATA" \
  --mode static \
  --reader-model qwen2.5 --reader-tokenizer /path/to/Qwen2.5-7B-Instruct \
  --api-base "$API" --api-key EMPTY \
  --aux-model gpt-6-luna --aux-api-base "$AUX_API_BASE" \
  --aux-api-key "$AUX_OPENAI_API_KEY" \
  --ranker-model ncbi/MedCPT-Cross-Encoder \
  --limit 20 --output "outputs/ca-medrgag/$DATA/static"
~~~

The auxiliary model here is a PLACEHOLDER based on your previous model
choice; ensure the actual provider endpoint/key is configured and supported.
The static arm makes one live KADS auxiliary call; it does NOT call
the new source-sufficiency observer.

### 3. NEW variant: source-coverage adaptive K

~~~bash
python -m coverage_medrgag.cli run \
  --medrgag-root "$MED" --dataset "$DATA" \
  --mode adaptive --depths 5 10 20 32 \
  --reader-model qwen2.5 --reader-tokenizer /path/to/Qwen2.5-7B-Instruct \
  --api-base "$API" --api-key EMPTY \
  --aux-model gpt-6-luna --aux-api-base "$AUX_API_BASE" \
  --aux-api-key "$AUX_OPENAI_API_KEY" \
  --ranker-model ncbi/MedCPT-Cross-Encoder \
  --limit 20 --output "outputs/ca-medrgag/$DATA/adaptive"
~~~

WARNING: use environment secrets, not keys committed to git. Cache hashes
and nonsecret run settings are saved to run_config.json. The runner is
append-only and resumes completed examples when the same settings/data are
reused. Initial cap is 20 examples; do NOT jump directly to 7,663 cases.

For strict paired sample IDs, pass the SAME --ids-file JSONL to all three
modes instead of --limit. This can be the frozen manifest generated by the
independent rag_mechanisms study.

Without --reader-tokenizer, the prototype enforces only its configured
CHAR length and is NOT exact token-level MedRGAG parity. For exploratory
smoke tests without the heavy MedCPT model, --no-rerank is supported but
**must not be reported as reproducing original MedRGAG**.

### 4. Offline report: accuracy, harm/rescue, Macro-F1, S/U/V

~~~bash
python -m coverage_medrgag.cli evaluate \
  --benchmark "$BENCH" --dataset "$DATA" \
  --predictions "outputs/ca-medrgag/$DATA/adaptive/predictions.jsonl" \
  --compare "outputs/ca-medrgag/$DATA/static/predictions.jsonl" \
  --bootstrap 1000 \
  --output "outputs/ca-medrgag/$DATA/adaptive_vs_static.json"
~~~

Report includes paired delta, harm/rescue, accuracy, Macro-F1, selection
composition, mean retrieval depth, auxiliary and Reader call counts.
When using MedRGAG's original prediction.json, add
--original-prediction /actual/path/to/MedRGAG/prediction.json.
This historical comparison is DESCRIPTIVE until checkpoint/prompt parity is
independently confirmed.

The source-only Judge's S_proxy and the conditional reader accuracies
U_proxy/V_proxy are explicitly flagged as NOT REAL S/U/V.
To measure *actual* S/U/V on a human-audited development subset, pass
--annotations /path/to/independent_review.jsonl with one JSON object per
line, containing id, boolean sufficient, and reviewer. Annotation MUST refer
to the exact documents actually inspected at each question's deployed K.

## Essential controlled ablations (pre-register)

1. Original KADS reference, to check adaptation has not silently lost
   MedRGAG's generator/reranker/Reader.
2. Live KADS K5 static versus live KADS K5/10/20/32 adaptive.
3. Fixed K20/32 KADS comparator (future control when deeper pools show value),
   to distinguish budget efficiency from simply using more retrieved text.
4. Evidence sufficiency independent review to assess how often source-only
   proxy stops prematurely or needlessly keeps retrieving.
5. Three Readers and all five original MedRGAG datasets. Keep results by
   dataset/Reader, not only average accuracy. Test with real costs.

The S(K) observer currently sees a bounded excerpt of each document and
its quoted evidence can still be semantically wrong or inapplicable to
specific patient/study conditions. Precomputed retrieval depth is NOT
follow-up query retrieval. Both limitations are explicit.

## Research provenance and novelty caution

- Original MedRGAG (WWW 2026): https://github.com/ll0ruc/MedRGAG
- Sufficient Context (ICLR 2025):
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html
- S2G-RAG (ACL 2026), structured source sufficiency + missing facts:
  https://aclanthology.org/2026.acl-long.1185/
  https://github.com/nianaaa/S2G-RAG
- RAG2 (NAACL 2025), medically useful evidence:
  https://aclanthology.org/2025.naacl-long.635/
  https://github.com/dmis-lab/RAG2
- Adaptive-RAG (NAACL 2024), adaptive retrieval complexity:
  https://aclanthology.org/2024.naacl-long.389/

Combining MedRGAG with source sufficiency and adaptive budgets is NOT
automatically a novel top-tier result: Adaptive-RAG and S2G-RAG already
study related decisions. Distinct contributions require empirically
demonstrating meaningful S(K), U(K), V(K) effects, improved cost–accuracy
frontier, and genuine cross-model/cross-dataset generalization.
