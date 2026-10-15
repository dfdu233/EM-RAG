"""Gold-isolated POST-HOC evaluation of MedRGAG variants.

S(K) = P(independently sufficient evidence at budget K)
U(K) = P(correct | independently sufficient)
V(K) = P(correct | independently insufficient)
These quantities must NOT be inferred from correctness or an LLM's own
uncalibrated sufficiency claim. Online proxy strata are labeled separately.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import random
from statistics import mean

from .data import benchmark


def read_jsonl(path: str | Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record["id"])
            if qid in output:
                raise ValueError(f"Duplicate ID in {path}: {qid}")
            output[qid] = record
    if not output:
        raise ValueError(f"Empty prediction file: {path}")
    return output


def read_original_predictions(path: str | Path) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as f:
        source = json.load(f)
    values = list(source.values()) if isinstance(source, dict) else source
    if not isinstance(values, list):
        raise ValueError("Original MedRGAG output must be list or ID mapping")
    result = {}
    for item in values:
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError("Malformed original MedRGAG prediction")
        qid = str(item["id"])
        if qid in result:
            raise ValueError(f"Duplicate original prediction {qid}")
        # Ignore answer / gt_answer fields entirely (gold is benchmark-derived).
        raw = item.get("model_answer", None)
        result[qid] = {"id": qid, "prediction": (
            raw.strip().upper() if isinstance(raw, str) else None)}
    return result


def paired(first: dict, second: dict, gold: dict[str, str],
           *, seed: int = 2026, bootstrap: int = 1000) -> dict:
    ids = sorted(set(first) & set(second) & set(gold))
    if not ids:
        raise ValueError("No paired questions shared by compared runs")
    d = []
    rescues = harms = changed = 0
    for qid in ids:
        a = first[qid].get("prediction") == gold[qid]
        b = second[qid].get("prediction") == gold[qid]
        d.append(int(b) - int(a))
        rescues += int(not a and b)
        harms += int(a and not b)
        changed += int(first[qid].get("prediction") != second[qid].get("prediction"))
    observed = mean(d)
    if bootstrap:
        rng = random.Random(seed)
        resamples = sorted(
            mean(d[rng.randrange(len(d))] for _ in d)
            for _ in range(bootstrap))
        lo = resamples[int((bootstrap - 1) * .025)]
        hi = resamples[int((bootstrap - 1) * .975)]
    else:
        lo = hi = None
    assert abs(observed - (rescues - harms) / len(ids)) < 1e-12
    return {
        "paired_n": len(ids), "mean_accuracy_delta": observed,
        "harm": harms, "rescue": rescues,
        "changed_prediction_count": changed,
        "ci95_question_bootstrap": [lo, hi] if bootstrap else None,
        "not_training_or_algorithm_selection": True,
    }


def _classification(rows: dict[str, dict], gold: dict[str, str]) -> dict:
    ids = list(rows)
    n = len(ids)
    if any(qid not in gold for qid in ids):
        raise KeyError("Unknown benchmark IDs in prediction file")
    per_label = {}
    for label in sorted(set(gold[qid] for qid in ids)):
        support = sum(gold[qid] == label for qid in ids)
        tp = sum(gold[qid] == label and
                 rows[qid].get("prediction") == label for qid in ids)
        pred_pos = sum(rows[qid].get("prediction") == label for qid in ids)
        precision = tp / pred_pos if pred_pos else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_label[label] = {
            "support": support, "precision": precision,
            "recall": recall, "f1": f1,
        }
    correct = sum(rows[qid].get("prediction") == gold[qid] for qid in ids)
    return {
        "n": n, "correct": correct,
        "accuracy": correct / n,
        "coverage": sum(rows[qid].get("prediction") is not None
                        for qid in ids) / n,
        "per_label": per_label,
        "macro_f1": mean(x["f1"] for x in per_label.values()),
    }


def _proxy_decomposition(rows: dict[str, dict], gold: dict[str, str]) -> dict | None:
    categorized = {True: [], False: []}
    for qid, row in rows.items():
        flag = row.get("source_proxy_sufficient")
        if isinstance(flag, bool):
            categorized[flag].append(qid)
    total = sum(len(ids) for ids in categorized.values())
    if total == 0:
        return None
    s_proxy = len(categorized[True]) / total
    u_proxy = (sum(rows[qid].get("prediction") == gold[qid]
                   for qid in categorized[True]) / len(categorized[True])
               if categorized[True] else None)
    v_proxy = (sum(rows[qid].get("prediction") == gold[qid]
                   for qid in categorized[False]) / len(categorized[False])
               if categorized[False] else None)
    return {
        "n": total, "S_proxy": s_proxy,
        "U_proxy_given_judge_sufficient": u_proxy,
        "V_proxy_given_judge_insufficient": v_proxy,
        "not_true_S_U_V": True,
        "warning": "These are LLM-proxy strata, not independently audited "
                   "evidence sufficiency or calibrated probabilities.",
    }


def _independent_sufficiency(path: str | Path, rows: dict[str, dict],
                             gold: dict[str, str]) -> dict:
    """Accept ONLY separately reviewed human labels for the deployed K.

    Records have {id, sufficient: true|false|null, reviewer}. Do not treat
    earlier model-produced self-assessments as independent labels.
    """
    truth: dict[bool, list[str]] = {True: [], False: []}
    seen = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item["id"])
            if qid in seen:
                raise ValueError(f"Duplicate human sufficiency annotation: {qid}")
            seen.add(qid)
            if qid not in rows:
                continue
            yes = item.get("sufficient")
            if type(yes) is not bool:
                continue
            if not item.get("reviewer"):
                raise ValueError(f"Missing independent reviewer identification: {qid}")
            truth[yes].append(qid)
    n = sum(map(len, truth.values()))
    return {
        "annotated_n": n,
        "S_reviewed": len(truth[True]) / n if n else None,
        "U_reviewed": (
            sum(rows[qid].get("prediction") == gold[qid] for qid in truth[True])
            / len(truth[True]) if truth[True] else None),
        "V_reviewed": (
            sum(rows[qid].get("prediction") == gold[qid] for qid in truth[False])
            / len(truth[False]) if truth[False] else None),
        "sampling_note": "If annotations are targeted at failure cases, "
                         "S_reviewed is NOT the population S(K).",
    }


def evaluate(predictions: str, benchmark_path: str, dataset: str,
             *, compare: str | None = None,
             original_prediction: str | None = None,
             annotation_file: str | None = None,
             bootstrap: int = 1000, seed: int = 2026) -> dict:
    _, gold = benchmark(benchmark_path, dataset)
    own = read_jsonl(predictions)
    if any(row.get("dataset") != dataset for row in own.values()):
        raise ValueError("Prediction dataset mismatch")
    cls = _classification(own, gold)
    by_k: dict[int, list] = defaultdict(list)
    for item in own.values():
        if isinstance(item.get("depth_used"), int):
            by_k[item["depth_used"]].append(item)
    summary = {
        "dataset": dataset,
        **cls,
        "mode": next(iter(own.values())).get("mode"),
        "mean_retrieval_depth": mean(row["depth_used"] for row in own.values()),
        "retrieval_depth_histogram": dict(
            Counter(str(row.get("depth_used")) for row in own.values())),
        "mean_auxiliary_calls": mean(
            row.get("auxiliary_calls", 0) for row in own.values()),
        "mean_reader_calls": mean(row.get("reader_calls", 0)
                                 for row in own.values()),
        "mean_selected_generated_docs": mean(
            sum(x == "parametric_generated" for x in row.get("selected_sources", []))
            for row in own.values()),
        "source_proxy_strata": _proxy_decomposition(own, gold),
        "gold_was_not_used_during_inference": True,
        "preprocessing_KGCC_cost_not_included": True,
    }
    if annotation_file:
        summary["independent_sufficiency_subset"] = _independent_sufficiency(
            annotation_file, own, gold)
    comparisons = {}
    if compare:
        alt = read_jsonl(compare)
        if any(row.get("dataset") != dataset for row in alt.values()):
            raise ValueError("Paired comparison dataset mismatch")
        if {row.get("reader") for row in alt.values()} != {
                row.get("reader") for row in own.values()}:
            raise ValueError("Compared runs used different Readers")
        comparisons["vs_other_run"] = paired(
            alt, own, gold, seed=seed, bootstrap=bootstrap)
    if original_prediction:
        original = read_original_predictions(original_prediction)
        comparisons["vs_original_MedRGAG_prediction"] = paired(
            original, own, gold, seed=seed, bootstrap=bootstrap)
        comparisons["original_parity_caveat"] = (
            "Original MedRGAG predictions may reflect a different vLLM "
            "endpoint or model checkpoint. Compare exact caches and prompts.")
    return {"summary": summary, "comparisons": comparisons}
