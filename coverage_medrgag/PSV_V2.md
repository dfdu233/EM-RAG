# PSV v2 — Input repair and source-intervention controls

Branch: feat/psv-reliable-evidence-v2, forked from
research/progressive-source-views. Original MedRGAG, all original KGCC/KADS
caches, original final MedCPT ranking and B=0 Reader question/template are
unchanged. No training, online test-gold lookups or new mandatory Judge.

## Why this branch exists

PSV's paired full-run experiment on 500 PubMedQA + 605 BioASQ Y/N examples
showed no accuracy improvement against original FULL MedRGAG, despite roughly
2.7-2.9 Reader calls per question. The original planner rejected a notable
number of >300-character queries. Distinct fresh search results shared the
same first three KADS anchor passages, potentially including KGCC background.
Reader answer voting counted agreement but did not establish that new source
evidence caused the agreement. Format-related B0 failures also inflate the
apparent number of corrected answers. None of these observations licenses
test-label-based tuning.

## Exactly which changes are implemented?

A. Query-size compatibility (engineering repair):
- A new concise planner instruction asks for <=240 characters.
- A hard limit of <=300 characters is enforced before real BM25 calls.
- For longer outputs, deterministic word-budget compaction prioritizes
  the question, the planner-specified target relationship and medical
  qualifiers even if they appear late in the original query.
- Rewritten queries are NOT semantically guaranteed to be equivalent.
  Source traces include the original query, both lengths and a
  'compacted_unverified' status. No additional planner calls.
- Use --query-size-policy reject to reproduce the EXACT historical
  planner request and the previous >300-character rejection logic.
  Different query policies MUST be evaluated as different interventions.

B. Uniform no-gold answer extraction (engineering repair):
- The exact same conservative answer parser is applied to original B0
  and to all new Reader views.
- Parse only explicit answer fields (JSON or unambiguous 'Final Answer: B').
  Do not infer labels from reasoning, source passages, unmarked prose, or
  a default class. Conflicting explicit JSON fields fail closed.
- Parse method is logged per output. Historical predictions can be
  reparsed OFFLINE without new model calls or gold; see the reparse command.

C. Source-only anchors (SCIENTIFIC controlled experiment):
- B0 remains EXACT original KADS context (including KGCC, if present).
- The new B>=1 Reader contexts may use --anchor-mode retrieved_only:
  their first three anchor docs come ONLY from original retrieved
  passages, not from generated KGCC material.
- Remainder of each 5-doc view contains the two *new* real retrieval
  passages; all source IDs/hashes remain logged as in old PSV.
- Original anchor is the explicit control via --anchor-mode original.
- This isolates a testable form of shared-KGCC contamination, but
  cannot prove generated knowledge is wrong.

D. Fixed-five-document matched source substitution:
- Use --paired-source-control with a 3+2 context:
  control = [same 3 original anchor docs] + [2 OTHER preexisting docs]
  new view = [same 3 original anchor docs] + [2 new BM25 docs].
- If the old-anchor context exactly reproduces B0, reuse B0 without
  another Reader call. Otherwise run ONE new Reader call per question
  (not per round) and record the additional cost.
- Both prompts use the same Reader/seed/temperature/MedCPT.
  Sampling noise and different document order can still confound
  single-run comparisons, so no causal correctness claim is made.
- Requires full five-doc matched context; underfilled views do not
  generate artificial comparisons.

E. Decision rule (OPTIONAL, not a correctness oracle):
- --policy anchored_vote reproduces old two-new-view agreement policy.
- --policy latest is the original optimistic control.
- --policy contrastive_vote additionally requires that the repeated
  alternative NOT already appear in matched old-source control. It is
  conservative and can block true corrections! Requires
  --paired-source-control. Do not select this option based on test labels.
- No added answer Judge, NLI, or trained gate.

F. Explicit new-source ranking ablation:
- --fresh-ranking question reproduces original fresh-MedCPT ranking.
- --fresh-ranking target_relation adds the planner's unresolved relation
  to the frozen MedCPT query for NEW source filtering only. Final Reader
  question and ranker input remain original. This is relevance, NOT
  proof of factual support; treat it as a separate ablation.

## Run (on server with existing MedRGAG corpus / indexes)

    git fetch origin
    git switch feat/psv-reliable-evidence-v2
    python -m pip install -e '.[all]'
    python -m pytest -q

Preserve original models, the original answer choices and the same
frozen ID manifest across variants. A small independent development
manifest should be used before confirmatory test evaluation.

    MED=/path/to/MedRGAG
    CORPUS=/path/to/MedRGAG/corpus
    DATA=pubmedqa
    READER=qwen2.5
    API=http://127.0.0.1:8000/v1
    DEV=/path/to/frozen_dev_ids.jsonl
    PLANNER=YOUR_SUPPORTED_PLANNER_MODEL
    PLANNER_API=YOUR_PLANNER_OPENAI_COMPATIBLE_API

Common command (B=0 plus two real BM25 query rounds, full MedCPT):

    python -m coverage_medrgag.progressive_cli run \
      --medrgag-root "$MED" --corpus-dir "$CORPUS" \
      --dataset "$DATA" --ids-file "$DEV" \
      --reader-model "$READER" --reader-tokenizer /path/to/reader-checkpoint \
      --api-base "$API" --api-key EMPTY \
      --planner-model "$PLANNER" --planner-api-base "$PLANNER_API" \
      --planner-api-key YOUR_PLANNER_API_KEY \
      --planner-omit-seed --planner-omit-temperature \
      --max-rounds 2 --retrieval-k 8 \
      --query-size-policy compact \
      --anchor-mode retrieved_only --paired-source-control \
      --policy anchored_vote \
      --max-new-examples 40 --output outputs/psv-v2/dev/source_only

Expected costs: normally 1 B0 Reader + up to 2 distinct-new-source Reader
calls + up to 1 matched-control Reader call; 2 planner calls + up to 2 BM25
search calls. Source control can consume one EXTRA Reader call compared to
old PSV. Full KGCC/KADS cache construction costs are not included.

Run --policy contrastive_vote with a SEPARATE output directory as a
predeclared alternative. For an original-anchor arm, use --anchor-mode
original with the same other settings and a different output directory.
Likewise, fresh-ranking and query-size-policy changes require their OWN
output directories. Resume protection compares all args and cache hashes.

## Gold-blind comparison of existing FROZEN runs

Use this to measure how much apparent improvement comes only from answer
format recovery, without ANY Reader/planner/retriever calls:

    python -m coverage_medrgag.progressive_cli reparse \
      --dataset "$DATA" --benchmark "$MED/benchmark.json" \
      --predictions /path/to/old_psv/predictions.jsonl \
      --output /path/to/old_psv/predictions_reparsed.jsonl

The source file is NEVER overwritten. All prefix decisions are recomputed
from original raw Reader outputs. Only benchmark questions and option labels
are read, never benchmark gold. The output trace records parse method.

Then perform gold-isolated evaluations:

    python -m coverage_medrgag.progressive_cli evaluate \
      --dataset "$DATA" --benchmark "$MED/benchmark.json" \
      --predictions outputs/psv-v2/dev/source_only/predictions.jsonl \
      --bootstrap 1000 --output outputs/psv-v2/dev/source_only/report.json

For fair new-vs-old contrasts with identical frozen IDs and original
B0 prompt fingerprints, add:

    --compare /path/to/other_arm/predictions.jsonl

Diagnostics include failed/repaired planner status, uniform parse methods,
generated KGCC anchor count, matched old-source vs first new-source
Harm/Rescue, cost by budget, and cross-run paired correct/incorrect changes.

## Scientific controls / claims NOT supported

The first priority is to compare at identical budgets AND identical
prompts/parsing; the matched control costs extra computation, so the
reported accuracy of that arm alone is NOT evidence of compute efficiency.

Suggested pre-registered comparisons:
1. Old PSV outputs under uniform frozen answer recovery (format-only).
2. Original anchor vs retrieved-only anchor, same queries/models/Reader
   budget; ideally replay cached queries/results to isolate anchor effects.
   A new live run can generate DIFFERENT planner queries after feedback.
   The current CLI does NOT cache/replay cross-arm planned queries and BM25
   results, so this matching limitation must be stated.
3. Retrieved-only anchor + optional matched fixed-five-doc source control.
4. Anchored vote vs contrastive vote, preferably by GOLD-BLIND offline
   policy replay on identical Reader/source traces. The main CLI runs both
   policies separately; different live runs are NOT exact policy-isolated
   experimental comparisons.
5. Original question vs target-relation-based new-source reranking.
6. A repeat-sampling control at identical contexts to quantify inherent
   Reader stochasticity.
7. No claim of retrieval volume/V scaling: corpus size is held fixed.
   The new budget axis is extra real retrieval query/Reader compute.

No held-out real GPU accuracy results have been run for this code.
Read the output diagnostics before selecting any new algorithm. If new
evidence is not consistently responsible for additional REAL corrections
or cost-accuracy tradeoff worsens, reject this idea rather than adding
more gates.
