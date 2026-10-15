# Controlled first-version EM-RAG + generated background

This branch starts from EM-RAG v0.1 commit cb66697def8fa5715e7d453e566c7aa5a59154cf. It changes ONLY the post-retrieval Reader's optional input: precomputed MedRGAG generated background is appended as **unverified parametric knowledge**. The initial answer, Retriever, atomic evidence extractor, EvidenceLedger, original Gate, scoring formula and thresholds are untouched.

The generated passages come from MedRGAG's outputs/<dataset>/docs/generate/snippets.json, created by generate_knowledge in https://github.com/ll0ruc/MedRGAG/blob/master/src/medrgag.py and the pipeline in https://github.com/ll0ruc/MedRGAG/blob/master/main.py. These passages are NEVER shown to the evidence-only Judge or inserted into the EvidenceLedger.

This is NOT full MedRGAG KGCC+KADS; it is a minimal test of whether GAG helps the Reader propose more correct answers. Generated content may be false. The GAG branch uses at most 3 cached snippets and 1500 characters by default; end-to-end cost must include upstream background generation.

## Four-arm small DEV experiment

One retrieval round with one original-question query keeps the retrieved documents identical across variants. Leave --kgcc-hints OFF so the Planner cannot be affected by GAG. Use a previously selected development subset, not a new held-out test set. Example dataset PubMedQA; adapt to your own caches.

~~~bash
git clone https://github.com/dfdu233/EM-RAG.git
cd EM-RAG
git switch exp/v1-gag-reader-only
pip install -e '.[all]'
pytest -q

RET=../MedRGAG/outputs/pubmedqa/docs/retrieval/snippets.json
GEN=../MedRGAG/outputs/pubmedqa/docs/generate/snippets.json
BENCH=../MedRGAG/benchmark.json

for policy in evidence latest; do
  python -m emrag.cli \
    --benchmark "$BENCH" --dataset pubmedqa --seed-retrieval "$RET" \
    --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
    --max-examples 30 --max-rounds 1 --queries-per-round 1 \
    --update-policy "$policy" --output "outputs/gag-dev-off-$policy"

  python -m emrag.cli \
    --benchmark "$BENCH" --dataset pubmedqa --seed-retrieval "$RET" \
    --model qwen --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
    --max-examples 30 --max-rounds 1 --queries-per-round 1 \
    --update-policy "$policy" --gag-generated-cache "$GEN" \
    --output "outputs/gag-dev-on-$policy"
done
~~~

Arms: no-GAG/evidence-Gate; GAG/evidence-Gate; no-GAG/latest-answer; GAG/latest-answer. All reuse the same first-round retrieved source cache and fixed inference settings. GAG-loaded questions must each have a generated snippet; missing entries cause an error, not silent mixture.

Offline four-arm report:

~~~bash
python scripts/compare_gag_ablation.py \
  --without-gag-gate outputs/gag-dev-off-evidence \
  --with-gag-gate outputs/gag-dev-on-evidence \
  --without-gag-latest outputs/gag-dev-off-latest \
  --with-gag-latest outputs/gag-dev-on-latest \
  --benchmark "$BENCH" --dataset pubmedqa \
  --output outputs/gag-dev-comparison.json
~~~

The report checks that all arms share the same example IDs, initial answer and retrieved document IDs. It separately measures Reader proposal accuracy, actual accepted answers, rescued/corrupted answers and correct proposals blocked by the Gate. Different runs can still exhibit inference nondeterminism; compare logged candidate discrepancies and model calls.

If GAG improves Reader proposals but not accepted answers, test whether the unchanged source-only Gate cannot validate useful parametric knowledge. **Do not** immediately loosen thresholds or declare a better algorithm. If GAG does not help Reader, return to retrieval/extraction sufficiency diagnosis.

No model is fine-tuned, no new Judge is introduced, and no existing GitHub main branch code is overwritten. The earlier 500-case results have already been inspected; consider them development evidence, not an untouched confirmatory test set.
