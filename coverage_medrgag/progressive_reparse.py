"""Gold-blind reparsing of FROZEN PSV outputs under a single parser.

Run this on historical PSV and original KADS predictions to separate format
recovery from knowledge corrections, WITHOUT touching original predictions or
calling any Reader/retriever/planner. Results must use a DIFFERENT output file.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .progressive import decide
from .progressive_validity import recover_answer


def reparse_record(row: dict, *, labels: tuple[str, ...]) -> dict:
    result = deepcopy(row)
    if not isinstance(row.get("base_raw_reader_output"), str):
        raise ValueError("No raw B0 output; frozen reparsing impossible")
    reference = recover_answer(row["base_raw_reader_output"], labels)
    old_baseline = row.get("base_prediction")
    result["base_prediction"] = reference.prediction
    result["base_parse_method"] = reference.method
    control = result.get("matched_source_control")
    if isinstance(control, dict) and isinstance(
            control.get("raw_reader_output"), str):
        match = recover_answer(control["raw_reader_output"], labels)
        control["prediction"] = match.prediction
        control["parse_method"] = match.method

    views = result.get("novel_views", [])
    prefix = result.get("prefixes", [])
    if len(prefix) != len(views) + 1:
        raise ValueError("Malformed PSV rounds; cannot safely replay")
    if any(entry.get("round") != i for i, entry in enumerate(prefix)):
        raise ValueError("PSV prefixes are not contiguous")
    history: list[str | None] = []
    prefix[0]["prediction"] = reference.prediction
    prefix[0]["base_prediction"] = reference.prediction
    prefix[0]["base_parse_method"] = reference.method
    for i, stage in enumerate(views, 1):
        old = stage.get("prediction")
        raw = stage.get("raw_reader_output")
        if isinstance(raw, str):
            prediction = recover_answer(raw, labels)
            stage["prediction"] = prediction.prediction
            stage["parse_method"] = prediction.method
            history.append(prediction.prediction)
            if isinstance(control, dict):
                stage["control_prediction"] = control.get("prediction")
                stage["differs_from_matched_control"] = (
                    prediction.prediction != control["prediction"]
                    if prediction.prediction is not None and
                    control["prediction"] is not None else None)
        elif old is not None:
            raise ValueError(f"Missing raw Reader output for PSV view {i}")
        voted = decide(reference.prediction, history,
                       policy=result.get("policy", "anchored_vote"),
                       matched_control=control)
        this = prefix[i]
        this["prediction"] = voted["prediction"]
        this["base_prediction"] = reference.prediction
        this["reason"] = voted["reason"]
        this["vote_counts"] = voted["counts"]
        this["valid_view_count"] = sum(x is not None for x in history)
        this["selected_doc_ids"] = (
            result.get("base_selected_doc_ids", [])
            if voted["prediction"] == reference.prediction
            else next(
                (x.get("selected_doc_ids", []) for x in reversed(views[:i])
                 if x.get("prediction") == voted["prediction"]),
                result.get("base_selected_doc_ids", [])))
    result["prediction"] = prefix[-1]["prediction"]
    result["reparsed_gold_free"] = True
    result["reparse_effects"] = {
        "baseline_old": old_baseline,
        "baseline_new": reference.prediction,
        "final_old": row.get("prediction"),
        "final_new": result["prediction"],
        "original_reader_calls_reused": result.get("reader_calls"),
    }
    return result


def reparse_file(predictions: str | Path, benchmark: str | Path,
                 dataset: str, output: str | Path) -> dict:
    from .progressive_cli import _questions_without_gold
    src = Path(predictions)
    dest = Path(output)
    if src.resolve() == dest.resolve():
        raise ValueError("Never overwrite frozen originals")
    questions = _questions_without_gold(benchmark, dataset)
    original_bytes = src.read_bytes()
    rows = []
    counter = Counter()
    for line in original_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("dataset") != dataset:
            raise ValueError("Mixed dataset in PSV predictions")
        qid = str(item["id"])
        if qid not in questions:
            raise ValueError(f"Unknown selected question: {qid}")
        adjusted = reparse_record(item, labels=questions[qid].labels)
        delta = adjusted["reparse_effects"]
        counter["records"] += 1
        counter["baseline_changed"] += int(delta["baseline_old"] != delta["baseline_new"])
        counter["final_changed"] += int(delta["final_old"] != delta["final_new"])
        rows.append(adjusted)
    if not rows:
        raise ValueError("Empty prediction file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError("Refusing to overwrite a prior comparison")
    dest.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                            for x in rows), encoding="utf-8")
    result = dict(counter)
    result["input_sha256"] = hashlib.sha256(original_bytes).hexdigest()
    result["output"] = str(dest)
    result["labels_loaded_without_gold"] = True
    result["reader_calls_added"] = 0
    return result
