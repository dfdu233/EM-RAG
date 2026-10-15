# Domain-Level RAG Scaling Mechanism Study

**This is NOT EM-RAG.** This standalone directory is a controlled research
instrument built around the five datasets, baseline predictions, retrieved
snippets, generated snippets, and Reader prompt released with
[MedRGAG (WWW 2026)](https://github.com/ll0ruc/MedRGAG).

No training, answer-update Gate, evidence ledger or experimental EM-RAG module
is used. We do not propose an improved algorithm here. We investigate
cross-method mechanisms that could later motivate a SIMPLE principled method.

### Core scientific questions

1. **Acquisition scaling:** As retrieval budget K increases, do we obtain
   independent *necessary facts*, or mainly redundant/noisy text?
2. **Utilization scaling:** When the retrieved evidence is independently judged
   sufficient, how often does the frozen Reader still fail?
3. **Negative transfer:** When extra documents are given, how often does a
   correct closed-book/low-K answer become wrong? Are retrieved additions
   more helpful than unrelated documents at approximately matched length?
4. **Adaptive versus static budget (optional live study):** Does multiple
   evidence-gap searches outperform one original-question top-K retrieval
   under the same maximum document and Reader input budget?
5. **External/parametric complementarity (secondary):** What is the
   incremental effect of cached MedRGAG KGCC-generated background versus
   external retrieved documents? These generated texts are NOT truth oracles.

### Benchmark scope and baselines

Use original MedRGAG benchmark.json with EXACT five dataset keys:
- medqa (MedQA-US)
- medmcqa
- mmlu (MMLU-Med)
- pubmedqa (PubMedQA*)
- bioasq (BioASQ-Y/N)

Use original Reader models separately: Qwen2.5-7B-Instruct,
Llama3.1-8B-Instruct, Ministral-8B-Instruct. Serve one at a time via a
local OpenAI-compatible vLLM endpoint; use the same model for all conditions
within a run.

The original paper also reports: Direct Response, Vanilla RAG, MedRAG,
i-MedRAG, GENREAD, MedGENIE, GRG, CGAP, MedRGAG. Their ALREADY GENERATED
per-question predictions can be imported into the offline report with
--reference NAME=/path/to/prediction.json. **Do not** misrepresent the
mechanism probes as running/reproducing all nine original method pipelines:
the context budget probe reuses only their benchmark and the MedRGAG
ranked retrieval cache. A same-prompt zero-document condition is not
necessarily identical to the original Direct Response prompt.

### Installation

On the research branch:

~~~bash
git fetch origin
git switch research/medrgag-scaling-mechanisms
python -m pip install -e '.[all]'
python -m pytest -q
~~~

The Python package rag_mechanisms uses only standard-library modules in its
offline stages. A running Reader needs the openai client (included in [all]).
For exact prompt token counts, install transformers and specify a local
--tokenizer path. Without it the tool enforces a CHAR budget only; it records
this limitation and does NOT assert token-matched causal effects.

You need the original MedRGAG clone with its source and locally prepared caches.
Code will NOT launch MedRGAG's brittle main.py, rebuild the corpus or train
extra models.

### Phase 0 — freeze 40 unbiased question IDs per dataset (0 model calls)

~~~bash
MED=../MedRGAG
DATA=pubmedqa
CACHE="$MED/outputs/$DATA/docs/retrieval/snippets.json"
BENCH="$MED/benchmark.json"

python -m rag_mechanisms.cli plan \
  --benchmark "$BENCH" --dataset "$DATA" \
  --retrieval-cache "$CACHE" --count 40 --seed 2026 \
  --output "outputs/mech/$DATA/manifest.jsonl"

python -m rag_mechanisms.cli census \
  --benchmark "$BENCH" --dataset "$DATA" \
  --retrieval-cache "$CACHE" \
  --manifest "outputs/mech/$DATA/manifest.jsonl" \
  --depths 0 1 3 5 10 20 32 \
  --output "outputs/mech/$DATA/census.json"
~~~

The manifest is selected by stable SHA-256 of dataset/id/seed.
No gold answer or model prediction is used to choose IDs. If ranked snippets
are not available, generate those using your ORIGINAL MedRGAG setup before
this study. Do not substitute synthetic documents for real medical evidence.

### Phase 1 — STATIC ranked evidence depth (core experiment)

Serve a frozen Reader. Example Qwen2.5:

~~~bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5 --gpu-memory-utilization 0.85
~~~

Then from research branch:

~~~bash
python -m rag_mechanisms.cli run \
  --benchmark "$BENCH" --dataset "$DATA" \
  --retrieval-cache "$CACHE" \
  --template "$MED/src/template.py" \
  --manifest "outputs/mech/$DATA/manifest.jsonl" \
  --model qwen2.5 --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --temperature 0.1 --seed 42 \
  --depths 0 1 3 5 10 20 32 \
  --max-examples 40 --max-calls 400 \
  --output "outputs/mech/$DATA/qwen2.5"
~~~

By default the exact upstream MedRGAG rag_template is extracted from the
original src/template.py via Python AST, without executing upstream code.
Each k uses a **prefix of the same original precomputed ranked result list**.
The Reader, temperature, seed, answer format, original question and options
are held fixed. K=0 uses the very same RAG prompt with an empty document
section, providing a controlled closed-book anchor.

- Parsed answers are STRICT: invalid outputs are counted as incorrect.
  The official MedRGAG locate_answer silently defaults unknown answers to A;
  we avoid this major class-bias confound.
- The tool stores answer traces and prompt hashes but does NOT duplicate raw
  medical source passages in the prediction file by default.
- Complete runs can resume in the SAME output dir after reaching --max-calls,
  provided input/source/template hashes and model config are unchanged.
- Backend errors fail immediately rather than being mistaken for wrong answers.
- For a real **token-level** budget, set --tokenizer /local/Qwen2.5-7B-Instruct.
  Conditions that exceed configured input budget are marked unavailable,
  NOT silently truncated. This may reduce available paired cases at K=32;
  reports always disclose pairwise sample sizes.

### Phase 1b — controls for noise and parametric knowledge (optional)

Run separate experiments on exactly the same manifest and Reader, with
additional conditions enabled:

~~~bash
GEN="$MED/outputs/$DATA/docs/generate/snippets.json"

python -m rag_mechanisms.cli run \
  --benchmark "$BENCH" --dataset "$DATA" \
  --retrieval-cache "$CACHE" \
  --generated-cache "$GEN" --include-generation \
  --include-negative-control \
  --template "$MED/src/template.py" \
  --manifest "outputs/mech/$DATA/manifest.jsonl" \
  --model qwen2.5 --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --depths 0 1 3 5 10 20 32 \
  --max-examples 40 --max-calls 600 \
  --output "outputs/mech/$DATA/qwen2.5-controls"
~~~

Generation conditions: 5 KGCC documents, or 3 original retrieved + 2 KGCC.
This is a cross-source intervention, NOT a reproduction of complete KADS.
Generated documents are MODEL knowledge and must never be counted as
independently verified evidence.

Negative-control conditions: first 3 own retrieved documents + 2/5
additional own lower-ranked documents vs first 3 + 2/5 other-question cached
documents trimmed to match each appended passage's CHARACTER length.
This removes a gross document/length confound, but donor documents may still
accidentally be relevant; this is NOT a certified irrelevant control.
It measures additional-text interference, not medical truth.

### Phase 1 report — paired harm/rescue and nonmonotonicity (0 model calls)

~~~bash
python -m rag_mechanisms.cli report \
  --benchmark "$BENCH" --dataset "$DATA" \
  --predictions "outputs/mech/$DATA/qwen2.5/predictions.jsonl" \
  --seed 2026 --bootstrap-samples 1000 \
  --output "outputs/mech/$DATA/qwen2.5/report.json"
~~~

Optional reference predictions from the SAME MedRGAG dataset and Reader:

~~~bash
  --reference MedRGAG=/path/to/MedRGAG/prediction.json \
  --reference i-MedRAG=/path/to/iMedRAG/prediction.json
~~~

These references are DESCRIPTIVE because their prompts and budgets may differ.
Use only the same matched question IDs for comparison.

The report includes accuracy, macro recall/recall by option (important for
BioASQ-Y/N), conditional harm and rescue, correct->wrong and wrong->correct
counts, paired question-level bootstrap confidence intervals, cost and
prompt lengths, nonmonotonic depth curves and an **offline oracle best K**.
The oracle best K is strictly analysis-only and not deployable.

### Phase 2 — HUMAN sufficiency audit (key for a scientific Motivation)

~~~bash
python -m rag_mechanisms.cli audit \
  --benchmark "$BENCH" --dataset "$DATA" \
  --predictions "outputs/mech/$DATA/qwen2.5/predictions.jsonl" \
  --retrieval-cache "$CACHE" \
  --per-category 8 --seed 2026 \
  --output "outputs/mech/$DATA/qwen2.5/audit"
~~~

The saved casebook displays raw MedRGAG source passages for targeted
wrong->right, right->wrong, unstable, persistently wrong and stable cases.
Annotations default to UNKNOWN, and must be independently reviewed:
- sufficient_k5, sufficient_k20
- decisive_new_fact_5_to_20
- source_conflicts_prior
- observed_qualifier_loss
- exact source document IDs and reviewer rationale

After independent review, run:

~~~bash
python -m rag_mechanisms.annotation_report \
  --annotations "outputs/mech/$DATA/qwen2.5/audit/annotations_template.jsonl" \
  --predictions "outputs/mech/$DATA/qwen2.5/predictions.jsonl" \
  --benchmark "$BENCH" --dataset "$DATA" \
  --output "outputs/mech/$DATA/qwen2.5/sufficiency_vs_reader.json"
~~~

This yields, for audited questions, accuracy **conditional on sufficient
retrieved evidence** and counts where sufficient new evidence fails to
produce a correct Reader answer. These transition-enriched cases are NOT
an unbiased sample of the overall dataset; make no overall prevalence claims
without randomized annotation sampling.

### Phase 3 — optional TRUE MULTI-ROUND acquisition test

This requires LIVE original MedRGAG Pyserini BM25 corpus/index prepared on the
server, plus original MedRGAG dependencies. It does not reuse the single-search
cache for follow-up queries, and it does not use EM-RAG.

Run only after Phases 0-2 show which knowledge gaps matter:

~~~bash
python -m rag_mechanisms.iterative \
  --benchmark "$BENCH" --dataset "$DATA" \
  --manifest "outputs/mech/$DATA/manifest.jsonl" \
  --medrgag-root "$MED" --db-dir "$MED/corpus" \
  --template "$MED/src/template.py" \
  --model qwen2.5 --base-url http://127.0.0.1:8000/v1 --api-key EMPTY \
  --rounds 4 --docs-per-round 4 --max-examples 20 \
  --max-calls 240 \
  --output "outputs/mech/$DATA/qwen2.5-live"

python -m rag_mechanisms.iterative_report \
  --iterative "outputs/mech/$DATA/qwen2.5-live/iterative.jsonl" \
  --benchmark "$BENCH" --dataset "$DATA" \
  --output "outputs/mech/$DATA/qwen2.5-live/report.json"
~~~

Static arm: query original question once for 4x4=16 max documents;
read static prefixes K=4,8,12,16.

Iterative arm: retrieve 4 new documents per round, using an intentionally
simple frozen-model *missing-fact query probe* for subsequent queries.
Read cumulative documents after each round. Same Qwen Reader, original
MedRGAG rag_template, maximum document quota and input budget.

Planner calls are extra cost; actual UNIQUE documents and prompt tokens can
differ (recorded). The comparison MUST NOT be reported as strictly
equal-cost if planner calls or duplicate retrieval differ. This is a
feasibility experiment of iterative acquisition, not a new RAG baseline.
Original MedRGAG retriever currently uses eval on corpus doc IDs; only run on
trusted prepared corpora. Never run on untrusted index files.

### Pre-registered interpretation: motivation before method

Let C(K) = fraction of questions with independently sufficient external
evidence at budget K, U(K) = Reader accuracy conditional on sufficient
evidence, and A(K) = overall answer accuracy.

A(K) = C(K)*U(K) + (1-C(K))*A_insufficient(K).

This is a probability decomposition, not an algorithm or monotonicity
guarantee. Since sufficiency is HUMAN-audited, do not mistake a simple
keyword hit or rank score for C(K).

- C(K) nearly flat; static and adaptive retrieval do not add necessary facts:
  retrieval is the binding bottleneck. Investigate source coverage /
  query specificity before designing a new model.
- C(K) rises but U(K) falls or stays low:
  Reader utilization is the binding bottleneck. Consider reader-aware
  evidence selection before new retrieval rounds.
- Correct closed-book answers often change to wrong answers, especially when
  additional documents fail to add necessary facts:
  negative transfer / external-parametric interference becomes the focus.
- Acquisition and utilization both improve with K but cost rises faster:
  question-conditional retrieval budget or trained lightweight complexity
  router may be a more principled Scalable solution.
- If no stable cross-model/dataset mechanism emerges, the mechanism-driven
  paper hypothesis should be REJECTED rather than patched with components.

Keep original EM-RAG branches and historical results unchanged. DO NOT tune
prompts, thresholds or routing using the same benchmark labels repeatedly.

### Research precedents and original code

- MedRGAG WWW 2026:
  https://arxiv.org/abs/2510.18297
  https://github.com/ll0ruc/MedRGAG/blob/master/main.py
  https://github.com/ll0ruc/MedRGAG/blob/master/src/template.py
- Sufficient Context ICLR 2025:
  https://research.google/pubs/sufficient-context-a-new-lens-on-retrieval-augmented-generation-systems-2/
  https://github.com/hljoren/sufficientcontext
- Lost in the Middle TACL 2024:
  https://aclanthology.org/2024.tacl-1.9/
- RAG2 NAACL 2025:
  https://aclanthology.org/2025.naacl-long.635/
  https://github.com/dmis-lab/RAG2/blob/main/retriever/main.py
  https://github.com/dmis-lab/RAG2/blob/main/classifier/README.md
- Adaptive-RAG NAACL 2024:
  https://aclanthology.org/2024.naacl-long.389/
  https://github.com/starsuzi/Adaptive-RAG/tree/main/classifier
- RECOMP ICLR 2024:
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html
  https://github.com/carriex/recomp

**All experiments are research-only. Biomedical snippets may have copyright
or privacy restrictions. Outputs are ignored by git; do not upload raw
clinical records or private patient information.**
