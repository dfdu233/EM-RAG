# MedRGAG A+B+C+D: four-paper engineering composition

Experimental and training-free **adaptation**, not an original SOTA claim.
Main, prior PSV, TF-AES and Coverage-Adaptive branches are unchanged.

## Four published inspirations

| Component / ablation | Published mechanism | Actual adaptation here |
|---|---|---|
| A: --no-query2doc | [Query2doc, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.585/) | Reuse cached MedRGAG KGCC as an UNVERIFIED pseudo-document to expand the BM25 query, with a hard 300-character search-interface limit |
| B: --no-ircot | [IRCoT, ACL 2023](https://aclanthology.org/2023.acl-long.557/) | Frozen Reader's previous reasoning and actual new external sources influence the next corpus query |
| C: --no-recomp | [RECOMP, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bda88ed2892f5e61c9a9bf215c566913-Abstract-Conference.html) | Deterministic, TRAINING-FREE extractive sentence selection; **not** RECOMP's trained compressor or its published algorithm |
| D: --no-sure | [SuRe, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/428ceef2cd8a53add7213e04d1746479-Abstract-Conference.html) | A frozen auxiliary proposes answer-conditioned quotes from EXTERNAL sources; local validation accepts only verbatim source substrings; final Reader assesses alternatives, **not** SuRe's full validity/ranking pipeline |

No answer voting. No gold in inference. Source quotations are not a factual
truth proof. Generated KGCC background is not an independent source.

## Original MedRGAG baseline B0 is preserved

B0 uses original first five retrieved passages, original five KGCC
passages, original cached KADS final_list selection, final MedCPT
Cross-Encoder reranking and same original Reader prompt/parameters.

New B1/B2: plan query -> KGCC pseudo-doc expansion -> TRUE original
MedRGAG Pyserini BM25 search -> MedCPT rerank -> optionally compress
source passages -> up to five Reader passages (one retrieved anchor, one
generated anchor when available; new sources take other slots) ->
optional answer-conditioned SOURCE-ONLY quotation notes -> ONE frozen
Reader answer. KADS/KGCC are the cached historical upstream outputs.

A new answer is never automatically assumed better than B0. Every
budget's answer is kept for post-hoc paired harm/rescue evaluation.

## Install on your MedRGAG server

    git fetch origin
    git switch feat/medrgag-abcd
    python -m pip install -e '.[all]'
    python -m pytest -q

Heavy server dependencies (PyTorch, transformers, Pyserini/Java and the
existing MedRGAG Lucene indices) must also be installed from the original
MedRGAG environment. Our lightweight Python extras do NOT install them.

Set paths and endpoints:

    MED=/home/dbw/MedRGAG
    DATA=pubmedqa
    READER=qwen2.5
    API=http://127.0.0.1:8000/v1
    AUX_MODEL=YOUR_SUPPORTED_AUXILIARY_MODEL
    AUX_API=YOUR_AUXILIARY_API_ENDPOINT
    DEV_IDS=/path/to/frozen_dev_ids.jsonl
    TOK=/path/to/Qwen2.5-7B-Instruct

B0 parity and baseline-only (no extra search index or auxiliary model):

    python -m coverage_medrgag.abcd_cli run \
      --medrgag-root "$MED" --dataset "$DATA" \
      --reader-model "$READER" --reader-tokenizer "$TOK" \
      --api-base "$API" --api-key EMPTY --max-rounds 0 \
      --ids-file "$DEV_IDS" --max-new-examples 40 \
      --output "outputs/abcd/$DATA/B0"

Full four-component B0/B1/B2, real new BM25 queries:

    python -m coverage_medrgag.abcd_cli run \
      --medrgag-root "$MED" --dataset "$DATA" \
      --reader-model "$READER" --reader-tokenizer "$TOK" \
      --api-base "$API" --api-key EMPTY \
      --planner-model "$AUX_MODEL" --planner-api-base "$AUX_API" \
      --planner-api-key YOUR_AUX_API_KEY \
      --planner-omit-seed --planner-omit-temperature \
      --corpus-dir "$MED/corpus" --corpus MedText \
      --max-rounds 2 --retrieval-k 8 --fresh-per-round 3 \
      --ids-file "$DEV_IDS" --max-new-examples 40 \
      --output "outputs/abcd/$DATA/ABCD_B2"

Do not commit API keys. Manifest and exact cache SHA-256 signatures
prevent mixing incompatible resumes in one output directory.

Ablations (use same data/model/round budget; each its OWN output folder):
- Remove A using --no-query2doc
- Remove B using --no-ircot
- Remove C using --no-recomp
- Remove D using --no-sure

When both B and D are disabled, no auxiliary LLM is needed and
--planner-model can be omitted. Baseline B0 also needs none.

Post-hoc evaluation (gold accessed ONLY in this command):

    python -m coverage_medrgag.abcd_cli evaluate \
      --benchmark "$MED/benchmark.json" --dataset "$DATA" \
      --predictions "outputs/abcd/$DATA/ABCD_B2/predictions.jsonl" \
      --bootstrap 1000 \
      --output "outputs/abcd/$DATA/ABCD_B2/report.json"

Per-B Accuracy/Macro-F1, harm/rescue, paired bootstrap CI,
mean Reader/auxiliary/real BM25 calls, fresh out-of-top32 passages,
quoted support/rejection, query truncation, Reader prompt SHA and
wall-time are available in the evaluation/trace files. The evaluator
is shared with PSV. No benchmark test-gold feedback is used for
algorithm tuning at inference.

## Major scientific limitations

- This is A+B+C+D as an engineering combination of established ideas;
  component composition by itself does NOT establish top-conference
  methodological originality or SOTA.
- KGCC and KADS preprocessing cost is omitted from per-question online
  cost, and must be included in full end-to-end comparisons.
- IRCoT here is prompt-based iterative search, not the full original
  pipeline. RECOMP originally trains compressors; this code uses no
  trained compressor. SuRe's full candidate summary ranking was adapted
  into local-verbatim-validated evidence notes.
- Quotation validation checks text provenance only, NOT medical truth or
  correct entailment of the answer option.
- BM25 query overlength is reduced deterministically under 300 chars;
  truncation and planner fallbacks are logged. No false claim of
  successful retrieval is recorded on missing results.
- All components can hurt accuracy. Do not interpret agreement, quote
  acceptance, or more source docs as independent confirmation.
- This pipeline may cost more than original MedRGAG; actual SOTA and
  shifted scaling boundaries need a held-out evaluation on the
  complete original five datasets and three frozen Readers.

Use a fixed development split; old PubMedQA/BioASQ gold-informed trials
should NOT be repeatedly reused for tuning or labeled fresh test results.
