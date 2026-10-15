#!/usr/bin/env bash
# Five original MedRGAG benchmarks, one frozen Reader.
# Run from repo root; requires ORIGINAL ranked caches.
set -euo pipefail
MED="$(printenv MEDRGAG_ROOT || printf '../MedRGAG')"
MODEL="$(printenv MODEL || printf 'qwen2.5')"
API="$(printenv API_BASE || printf 'http://127.0.0.1:8000/v1')"
N="$(printenv N || printf '40')"
OUT="$(printenv OUT_ROOT || printf 'outputs/mechanisms')"
DATASETS="$(printenv DATASETS || printf 'medqa medmcqa mmlu pubmedqa bioasq')"
CALL_CAP="$(printenv CALL_CAP || printf '400')"

for dataset in $DATASETS; do
  cache="$MED/outputs/$dataset/docs/retrieval/snippets.json"
  bench="$MED/benchmark.json"
  template="$MED/src/template.py"
  base="$OUT/$dataset"
  if [[ ! -f "$cache" || ! -f "$bench" || ! -f "$template" ]]; then
    echo "ERROR: Missing original MedRGAG data for $dataset" >&2
    exit 2
  fi
  mkdir -p "$base"
  if [[ ! -f "$base/manifest.jsonl" ]]; then
    python -m rag_mechanisms.cli plan \
      --benchmark "$bench" --dataset "$dataset" \
      --retrieval-cache "$cache" --seed 2026 --count "$N" \
      --output "$base/manifest.jsonl"
  fi
  python -m rag_mechanisms.cli census \
    --benchmark "$bench" --dataset "$dataset" \
    --retrieval-cache "$cache" --manifest "$base/manifest.jsonl" \
    --output "$base/census.json"
  python -m rag_mechanisms.cli run \
    --benchmark "$bench" --dataset "$dataset" \
    --retrieval-cache "$cache" --manifest "$base/manifest.jsonl" \
    --template "$template" --model "$MODEL" \
    --base-url "$API" --api-key EMPTY \
    --temperature 0.1 --seed 42 \
    --depths 0 1 3 5 10 20 32 \
    --max-examples "$N" --max-calls "$CALL_CAP" \
    --output "$base/$MODEL"
  python -m rag_mechanisms.cli report \
    --benchmark "$bench" --dataset "$dataset" \
    --predictions "$base/$MODEL/predictions.jsonl" \
    --seed 2026 --bootstrap-samples 1000 \
    --output "$base/$MODEL/report.json"
  echo "Saved: $base/$MODEL/report.json"
done
