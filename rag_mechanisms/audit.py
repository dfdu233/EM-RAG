"""Create human-review casebook for DOMAIN-level evidence acquisition/utilization.

No frozen model judge labels sufficient or insufficient by fiat. The human
reviews retrieved passages with original source text and can choose unknown.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

from .data import Passage, Question, read_jsonl


def _valid(row: dict) -> bool:
    return row.get("status") in {"ok", "invalid_answer"}


def prepare_casebook(predictions: str, dataset: str,
                     questions: dict[str, Question], gold: dict[str, str],
                     ranked: dict[str, list[Passage]],
                     *, seed: int = 2026, per_category: int = 8
                     ) -> tuple[list[dict], str]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in read_jsonl(predictions):
        if row.get("dataset") != dataset or not _valid(row):
            continue
        by_id[str(row["id"])][str(row["condition"])] = row
    strata = defaultdict(list)
    for qid, records in by_id.items():
        if "retrieved_k0" not in records or qid not in gold or qid not in ranked:
            continue
        base = records["retrieved_k0"].get("prediction")
        large_options = sorted(
            (name for name in records if name.startswith("retrieved_k")
             and name[len("retrieved_k"):].isdigit()
             and int(name[len("retrieved_k"):]) > 0),
            key=lambda n: int(n[len("retrieved_k"):]))
        if not large_options:
            continue
        last = records[large_options[-1]].get("prediction")
        if base == gold[qid] and last != gold[qid]:
            stratum = "retrieval_harmed_correct_prior"
        elif base != gold[qid] and last == gold[qid]:
            stratum = "retrieval_rescued_incorrect_prior"
        elif base == gold[qid] and last == gold[qid]:
            stratum = "both_correct"
        else:
            stratum = "both_incorrect"
        outcomes = [records[k].get("prediction") == gold[qid] for k in
                    ["retrieved_k0"] + large_options]
        if any(x != y for x, y in zip(outcomes, outcomes[1:])) and (
                stratum in {"both_correct", "both_incorrect"}):
            stratum = "intermediate_depth_flip"
        strata[stratum].append(qid)

    selected: list[dict] = []
    order = (
        "retrieval_harmed_correct_prior",
        "retrieval_rescued_incorrect_prior",
        "intermediate_depth_flip",
        "both_incorrect", "both_correct",
    )
    for label in order:
        ids = strata[label]
        ids.sort(key=lambda qid: sha256(
            f"{seed}|{dataset}|{label}|{qid}".encode()).hexdigest())
        for qid in ids[:per_category]:
            selected.append({"dataset": dataset, "id": qid, "stratum": label})
    casebook = [
        "# MedRGAG benchmark: RAG scaling mechanism audit",
        "",
        "This is DOMAIN research, not an EM-RAG gate analysis.",
        "Gold answers appear ONLY in this offline audit, never in inference prompts.",
        "Read the RAW retrieval sources and independently label sufficiency.",
        "Do not assume that including an answer token proves supporting evidence.",
        "Do not infer retrieval failure from a model's wrong answer.",
        "",
        "Manual annotation fields (yes/no/unknown):",
        "- sufficient_k5: top-5 retrieved passages independently suffice to justify gold",
        "- sufficient_k20: top-20 retrieved passages independently suffice",
        "- decisive_new_fact_5_to_20: is a necessary independent fact newly available",
        "- source_conflicts_prior: does retrieved evidence support something contrary",
        "  to the closed-book answer, and is the source reliable/applicable?",
        "- observed_qualifier_loss: does the answer require conditions absent in short extracts?",
        "Include quote spans, source names, and rationale. Reviewers may disagree.",
        "",
    ]
    for item in selected:
        qid = item["id"]
        q = questions[qid]
        records = by_id[qid]
        casebook.extend([
            f"## {dataset}/{qid} | {item['stratum']}",
            "",
            q.text,
            "",
            "Options:",
            *(f"- {label}: {answer}" for label, answer in q.options.items()),
            "",
            f"Gold (OFFLINE ONLY): {gold[qid]}",
            "",
            "Predicted by condition:",
            *(f"- {label}: {record.get('prediction')}"
              for label, record in sorted(records.items())),
            "",
            "### Ranked MedRGAG retrieved documents (for evidence sufficiency review)",
            "",
        ])
        for i, doc in enumerate(ranked[qid][:20], start=1):
            text = doc.text[:1800].replace("\n", " ")
            suffix = " [DISPLAY TRUNCATED]" if len(doc.text) > 1800 else ""
            casebook.append(
                f"- Rank {i}, ID {doc.id}, source {doc.source}, "
                f"title {doc.title}: {text}{suffix}"
            )
        casebook.extend(["", "---", ""])
    return selected, "\n".join(casebook)


def annotation_template(cases: list[dict]) -> list[dict]:
    return [
        {
            "dataset": case["dataset"], "id": case["id"],
            "stratum": case["stratum"],
            "sufficient_k5": "unknown", "sufficient_k20": "unknown",
            "decisive_new_fact_5_to_20": "unknown",
            "source_conflicts_prior": "unknown",
            "observed_qualifier_loss": "unknown",
            "document_ids_support": [],
            "document_ids_refute": [],
            "rationale": "",
            "reviewer": "",
        }
        for case in cases
    ]


def aggregate_annotations(rows: list[dict]) -> dict:
    """Only independent human annotation files may yield sufficiency claims."""
    labels = {"yes", "no", "unknown"}
    fields = ("sufficient_k5", "sufficient_k20",
              "decisive_new_fact_5_to_20", "source_conflicts_prior",
              "observed_qualifier_loss")
    for row in rows:
        for key in fields:
            if row.get(key, "unknown") not in labels:
                raise ValueError(f"Unexpected {key} for {row.get('id')}")
    return {
        "annotated_n": len(rows),
        "counts": {field: {
            outcome: sum(x.get(field, "unknown") == outcome for x in rows)
            for outcome in ("yes", "no", "unknown")
        } for field in fields},
        "warning": "Small, nonrandom transition strata; do not estimate population "
                   "prevalence without weighting/sampling corrections.",
    }
