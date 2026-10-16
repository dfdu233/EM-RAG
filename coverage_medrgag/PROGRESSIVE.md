# Progressive Source Views: Scalable MedRGAG research prototype

Training-free, original frozen MedRGAG resources. This branch does NOT
claim improved biomedical accuracy, SOTA, or a novel verified scaling law.

## The ONLY experimental feedback loop

Keep original cached MedRGAG KGCC, KADS selected five passages, final
MedCPT and Reader prompt unchanged as an exact budget-zero reference.

At each extra budget round:
1. Read KGCC snippets as unverified parametric hypotheses, and use a
   frozen auxiliary planner for ONE new task-specific corpus search query.
   Later queries may respond to actual Reader answer disagreement.
2. Execute the new query on the ORIGINAL MedRGAG BM25 corpus (not cached
   Top32 prefixes); select two unseen retrieved source snippets. Preserve
   up to three original KADS anchor passages in the final five-doc context.
3. Run the same frozen Reader on that evidence view. Record its output.
   If two distinct NEW source views agree against a parseable KADS answer,
   prefer their answer; on ties preserve KADS. This vote is an uncalibrated
   heuristic, NOT a correctness guarantee.

One Reader call for B=0, and only one additional Reader call per round
that contributes new documents. KGCC/KADS baseline costs are cached and
must be included when reporting full end-to-end cost.

Disjoint source ITEM IDs do not imply independent medical sources or
independent Reader errors. Two views can agree wrongly. This is a
falsifiable retrieval-plus-Reader experiment, not a verified theorem.

## Run on server

    git fetch origin
    git switch research/progressive-source-views
    python -m pip install -e '.[all]'
    python -m pytest -q

Original MedRGAG root must contain:
- benchmark.json
- src/template.py
- outputs/DATA/docs/retrieval/snippets.json
- outputs/DATA/docs/generate/snippets.json
- outputs/DATA/docs/generate/snippets_final_select.json

For extra rounds, the original BM25 Lucene index MUST exist on disk
under CORPUS_DIR/textbooks/index/bm25 and wikipedia/index/bm25, plus the
corresponding corpus chunk folders. Absent indexes cause a hard failure:
the tool NEVER downloads/builds an index silently.

Example environment (adjust paths and model names):

    MED=/home/dbw/MedRGAG
    DATA=pubmedqa
    READER=qwen2.5
    API=http://127.0.0.1:8000/v1
    DEV_IDS=/path/to/frozen_dev_ids.jsonl

Original baseline parity only: B=0, one frozen Reader call.

    python -m coverage_medrgag.progressive_cli run \
      --medrgag-root "$MED" --dataset "$DATA" \
      --reader-model "$READER" --reader-tokenizer /path/to/reader-checkpoint \
      --api-base "$API" --api-key EMPTY --max-rounds 0 \
      --ids-file "$DEV_IDS" --max-new-examples 40 \
      --output "outputs/psv/$DATA/B0"

B=0,1,2 in ONE cached execution, each prefix independently evaluated:

    python -m coverage_medrgag.progressive_cli run \
      --medrgag-root "$MED" --dataset "$DATA" \
      --reader-model "$READER" --reader-tokenizer /path/to/reader-checkpoint \
      --api-base "$API" --api-key EMPTY \
      --planner-model YOUR_SUPPORTED_PLANNER_MODEL \
      --planner-api-base YOUR_COMPATIBLE_PLANNER_API_URL \
      --planner-api-key YOUR_PLANNER_API_KEY \
      --planner-omit-seed --planner-omit-temperature \
      --corpus-dir /path/to/original/MedRGAG/corpus --corpus MedText \
      --max-rounds 2 --retrieval-k 8 \
      --planner-mode kgcc --policy anchored_vote \
      --ids-file "$DEV_IDS" --max-new-examples 40 \
      --output "outputs/psv/$DATA/kgcc_2q"

Official scientific evaluation, after generation only:

    python -m coverage_medrgag.progressive_cli evaluate \
      --benchmark "$MED/benchmark.json" --dataset "$DATA" \
      --predictions "outputs/psv/$DATA/kgcc_2q/predictions.jsonl" \
      --bootstrap 1000 --output "outputs/psv/$DATA/kgcc_2q/report.json"

Model identifiers and paths are examples. Omit planner-api-base if
the same API serves both Reader and Planner. If your planner supports
temperature/seed, you may keep these flags off. Never commit API keys.

## Evaluations that MUST accompany any efficacy claim

- Same IDs/seed/checkpoint and exact KADS B=0 prompt SHA.
- Answer Accuracy(B), Macro-F1(B), Rescue(B), Harm(B), full paired CI.
- Source IDs and hashes outside original Top32 vs merely different Top5.
- Correctness gain per *real* retrieval call, Reader call and elapsed time.
- Question-only planner ABLATION (planner-mode question), same budget.
- Latest-view decision ABLATION (policy latest), same budget.
- Comparison to naive 2 extra Reader self-consistency calls, standard
  multi-query retrieval and fixed-K/KGCC MedRGAG with comparable compute.
- Only after dev: five MedRGAG datasets and all three frozen Readers.
- Test corpus scale separately: Textbooks, Wikipedia, MedText. This is
  an axis distinct from number of inference rounds and retrieval k.

No test gold enters inference: question/options are loaded without
inspecting answer values; gold is accessed ONLY by evaluate mode.

## Related / novelty limits

Recent nearest neighbors include S2G-RAG (ACL 2026), Contrastive Hypothesis
Retrieval (2026), Mixture-of-Intervention (NAACL 2025), MEGA-RAG
(Frontiers in Public Health 2025), and Beyond Retrieval Failure (preprint
2026-10-15). Multi-view voting and targeted new retrieval have ALREADY
been investigated. The present code is a DISCOVERY/DIAGNOSTIC experiment
and not yet an academically novel method.

https://aclanthology.org/2026.acl-long.1185/
https://arxiv.org/abs/2604.04593
https://doi.org/10.3389/fpubh.2025.1635381
https://aclanthology.org/2025.findings-naacl.409/
https://arxiv.org/abs/2610.21184

STOP instead of adding heuristic gates if true new evidence seldom arrives,
if answer-view agreement is frequently wrong, or if improved accuracy
does not exceed paired baseline at comparable real compute.
