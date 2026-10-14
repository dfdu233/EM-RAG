# Mechanism-first diagnosis (v0.2)

EM-RAG v0.1 did not show a meaningful accuracy gain in the reported 500-question experiment (EM-RAG 50.2%, same-prompt baseline 50.4%, MedRGAG 50.8%). The user-supplied counts include 60 harmful changes blocked and 67 beneficial corrections blocked. Those **must be reproduced from the original per-question traces before any algorithmic change is declared effective**.

Do NOT blindly relax margin thresholds, rewrite the judge prompt to fit those 500 labels, or run more held-out test examples. This workflow identifies **which mechanism failed** instead of tuning accuracy.

## Stage 0: retrospective diagnosis — zero GPU calls

Use the **original** EM-RAG traces from the same experiment, not a synthetic benchmark. This example assumes the 500-case dataset was PubMedQA; replace paths/dataset with your actual run:

~~~bash
python -m emrag.diagnose_cli \
  --traces outputs/pubmedqa-emrag/traces.jsonl \
  --benchmark ../MedRGAG/benchmark.json \
  --dataset pubmedqa \
  --per-stratum 6 --seed 2026 \
  --output outputs/mechanism-dev
~~~

Optionally, when previous traces lack source passages but MedRGAG's initial-query cache exists:

~~~bash
python -m emrag.diagnose_cli \
  --traces outputs/pubmedqa-emrag/traces.jsonl \
  --benchmark ../MedRGAG/benchmark.json \
  --dataset pubmedqa \
  --medrgag-first-turn-cache ../MedRGAG/outputs/pubmedqa/docs/retrieval/snippets.json \
  --per-stratum 6 --seed 2026 \
  --output outputs/mechanism-dev
~~~

**Hydrated documents are FIRST-TURN ONLY.** No later-round search result is invented or inferred. If the first-turn cache does not answer the question, that DOES NOT prove later rounds failed to retrieve sufficient material. Existing raw documents in traces are never overwritten.

The loader reads the original gate margin from adjacent run_config.json. For legacy traces without configuration it assumes the original v0.1 margin (2), marking that assumption; --original-margin is for recording the *actual historic* setting, NOT for optimization.

Outputs:

| File | Purpose |
|---|---|
| diagnostic_summary.json | Baseline/final accuracy; final harm/rescue; candidate upper bound; proposal-change counts; rejection reasons; descriptive score-difference AUC |
| all_cases.jsonl | Per-question failure flags/events (development analysis only, includes gold) |
| sample_manifest.jsonl | Stable, seeded stratified case IDs, reusable across all probes |
| annotations_template.jsonl | Human independent sufficiency/entailment review, initially unknown |
| audit_casebook.md | Questions, proposals, gate reasons, quotes and saved passages when available |

An opportunity count is **per proposed answer change**. A final harm/rescue count is **per question**. The same question can produce multiple opportunities; these two quantities MUST NOT be conflated.

### Objective definitions

For previous output a_(t-1), proposed p_t, emitted a_t and gold y:

- Rescue opportunity: a_(t-1) != y and p_t = y
- Harm opportunity: a_(t-1) = y and p_t != y
- Adopted: a_t = p_t
- Blocked: a_t = a_(t-1) != p_t
- Rerouted: system selected a historical third choice
- Candidate oracle: at least one generation across all rounds (including baseline) matched gold; offline upper bound ONLY.

The blocked-candidate reasons explain the ORIGINAL gate: no support, insufficient, missing facts, refuted, or margin. Several reasons may co-occur. A scorecard can be absent in old logs.

AUC compares rescue and harm attempts using

    delta_t = score(proposal, E_t) - score(previous_output, E_t)

where both scores refer to the SAME saved evidence state. This is purely descriptive: repeated attempts on one question are correlated, scores are not calibrated, and no significance interval is claimed.

## Stage 1: independent raw-source sufficiency review

Review only the fixed small sample. Never call a model's own "sufficient" flag a gold sufficiency label.

Fill the annotations_template.jsonl file (one JSON object per case):

~~~json
{"id":"question_id",
 "retrieved_sufficient":"unknown",
 "active_context_sufficient":"unknown",
 "quote_entailment":"unknown",
 "source_quality":"unknown",
 "notes":""}
~~~

Allowed labels: yes / no / unknown.

- retrieved_sufficient: Were the **original raw retrieved passages** enough to justify the gold choice with appropriate clinical conditions?
- active_context_sufficient: Did the evidence actually provided to the Reader preserve enough relevant facts?
- quote_entailment: Does each extracted claim follow from the quoted source text?
- source_quality: Is the source current, appropriate and reliable?

Do not label retrieved_sufficient=no solely because extraction returned nothing. Check *all relevant rounds* or leave unknown. With only a first-turn cache, a positive sufficiency finding is possible, but failure to find a fact in the cache says nothing about missing later rounds.

For medical cases distinguish age, population, contraindications, negation, laterality, time and multiple-hop dependencies.

~~~bash
python -m emrag.diagnose_cli \
  --traces outputs/pubmedqa-emrag/traces.jsonl \
  --benchmark ../MedRGAG/benchmark.json --dataset pubmedqa \
  --annotations outputs/mechanism-dev/annotations_template.jsonl \
  --fixed-manifest outputs/mechanism-dev/sample_manifest.jsonl \
  --output outputs/mechanism-reviewed
~~~

Only an independent "active sufficient=yes" annotation allows the diagnostic tool to describe a case as "Reader did not propose a correct answer despite sufficiently annotated evidence".

## Stage 2: fixed frozen Reader probes (optional, small GPU budget)

For independently reviewed cases only, perform short offline interventions. **No new searches, no training and no online gate modifications**:

~~~bash
python -m emrag.reader_probe \
  --traces outputs/pubmedqa-emrag/traces.jsonl \
  --manifest outputs/mechanism-dev/sample_manifest.jsonl \
  --annotations outputs/mechanism-dev/annotations_template.jsonl \
  --benchmark ../MedRGAG/benchmark.json --dataset pubmedqa \
  --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --max-cases 12 --max-calls 24 \
  --output outputs/reader-probe-dev
~~~

Three source-context conditions:
- active_quotes — exactly the active evidence IDs seen by the original Reader in the final recorded round.
- all_quotes — all extracted quote-grounded evidence in the saved ledger.
- raw_documents — full retrieved passage text, **only if actually present in the trace or an explicitly scoped warm-start cache**.

The frozen model is asked to answer the same question. It may still rely on parametric knowledge; therefore these conditions do NOT establish a perfect "source-only" oracle. Missing contexts are marked unavailable and incur no model call.

Exploratory unreviewed probes require --allow-unreviewed; such results cannot attribute failures to the Reader.

## Stage 3: NEW DEV-only reruns with full document snapshots

If legacy traces do not contain later-round documents, rerun ONLY the previously fixed development sample:

~~~bash
python -m emrag.cli \
  --benchmark ../MedRGAG/benchmark.json --dataset pubmedqa \
  --ids-file outputs/mechanism-dev/sample_manifest.jsonl \
  --medrgag-root ../MedRGAG --db-dir ../MedRGAG/corpus \
  --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --max-rounds 4 --queries-per-round 2 \
  --update-policy evidence --record-documents \
  --output outputs/selected-dev-full-docs
~~~

This DOES consume inference/retrieval budget. Raw documents may be copyrighted or sensitive. Keep them outside git (outputs/ is ignored) and follow data-governance requirements.

## Decision tree

| Independently validated observation | Main bottleneck | Next algorithmic change |
|---|---|---|
| Retrieved documents genuinely insufficient | Retrieval | Improve missing-fact queries/indexing without changing the gate |
| Retrieved sufficient, active context insufficient | Extraction/selection | Preserve necessary facts, qualifiers, and contradictions in ledger |
| Active context sufficient, no correct proposed answer | Reader | Study prior/context interference, candidate generation |
| Correct candidate proposed but gated out; rescue and harm score distributions overlap | Gate | Replace evidence-update criterion, not just its threshold |
| Accepted correct choice later regresses | Sequential consistency | Investigate provenance, contradictions and context displacement |

**Evaluation remains paused.** Make a new algorithmic claim only after a specific causal bottleneck is supported by independent development evidence, then preregister a held-out comparison.

## Open-source code correspondence

- MedRGAG: https://github.com/ll0ruc/MedRGAG/blob/master/src/medrgag.py (retrieval, rank_snippets, generate_knowledge), https://github.com/ll0ruc/MedRGAG/blob/master/src/utils.py (RetrievalSystem.retrieve).
- S2G-RAG (ACL 2026): https://github.com/nianaaa/S2G-RAG/blob/main/inference/inference_bm25.py (build_suff_user_prompt, call_suff_gate_batch). Original S2G Judge has separate LoRA training; EM-RAG keeps a frozen model.
- MA-RAG (ICML 2026): https://github.com/NJU-RL/MA-RAG/blob/main/ma_rag_entropy.py (answer-disagreement follow-up search).
- CF-RAG (ICLR 2026): https://github.com/CF-RAG/CF-RAG/blob/main/cfrag_pipeline.py (counterfactual candidate arbitration).

EM-RAG own audit code: ../emrag/diagnostics.py, ../emrag/diagnose_cli.py, ../emrag/reader_probe.py, ../emrag/engine.py.
