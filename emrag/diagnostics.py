"""Offline mechanism diagnostics from EXISTING EM-RAG traces.

Gold labels are used ONLY in this post-hoc analysis. They are never passed to
the retrieval engine, prompt builder, generator, or on-line update gate.

This tool separates (1) evidence sufficiency (requires independent annotation),
(2) reader candidate generation, and (3) gate discrimination. It will not claim
that a model-generated sufficiency flag proves ground-truth sufficiency.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
from typing import Any, Mapping

STRATA = (
    "rescue_blocked", "harm_accepted", "rescue_accepted", "harm_blocked",
    "reader_never_proposed_gold", "initially_correct_stable", "other"
)
AUDIT_FIELDS = ("retrieved_sufficient", "active_context_sufficient",
                "quote_entailment", "source_quality")
AUDIT_CHOICES = {"yes", "no", "unknown"}


def read_traces(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError(f"Malformed trace at line {n}")
            qid = str(row["id"])
            if qid in seen:
                raise ValueError(f"Duplicate trace id: {qid}")
            seen.add(qid)
            rounds = row.get("traces")
            if not isinstance(rounds, list) or not rounds:
                raise ValueError(f"Missing round traces: {qid}")
            ids = [r.get("round") for r in rounds]
            if ids[0] != 0 or ids != sorted(set(ids)):
                raise ValueError(f"Invalid/duplicate round sequence for {qid}: {ids}")
            rows.append(row)
    if not rows:
        raise ValueError("No trace records found")
    return rows


def hydrate_medrgag_first_turn(rows: list[dict], cache_path: str | Path) -> dict:
    """Attach existing MedRGAG initial-query snippets for MANUAL evidence audit.

    This is *not* multi-round retrieval replay. These initial-query snippets
    can help prove sufficiency if they contain the required fact; their failure
    to contain it says nothing about unseen later retrieval rounds.
    """
    with Path(cache_path).open(encoding="utf-8") as fp:
        corpus = json.load(fp)
    count = 0
    for row in rows:
        if row.get("documents"):
            row.setdefault("document_snapshot_scope", "existing_trace")
            continue
        original = corpus.get(str(row["id"]))
        if not isinstance(original, dict):
            continue
        snippets = original.get("snippets") or []
        if not isinstance(snippets, list):
            continue
        documents = []
        for rank, snippet in enumerate(snippets):
            if not isinstance(snippet, dict):
                continue
            content = str(snippet.get("contents") or snippet.get("content") or "")
            if not content.strip():
                continue
            documents.append({
                "id": str(snippet.get("id", f"cached_{rank}")),
                "contents": content, "title": str(snippet.get("title", "")),
                "source": "MedRGAG:cached_first_turn",
                "round": 1, "query": "<initial-original-query-not-replayed>",
            })
        if documents:
            row["documents"] = documents
            row["document_snapshot_scope"] = "cached_first_turn_only"
            count += 1
    return {"cases_hydrated": count,
            "warning": "ONLY MedRGAG first-turn cached retrieval was loaded. "
                       "No subsequent query results were reconstructed."}


def choice(value: Any) -> str | None:
    if value is None:
        return None
    c = str(value).strip().upper()
    return c or None


def last_at_or_before(row: dict, round_id: int) -> dict:
    return next(t for t in reversed(row["traces"]) if t["round"] <= round_id)


def blocking_reasons(step: dict, proposal: str, incumbent: str | None,
                     margin: int = 2) -> list[str]:
    """Explain legacy gate predicates; NOT an independent verifier.

    Unknowns are explicit because earlier runs may lack detailed scorecards.
    """
    scores = step.get("scores") or {}
    if not isinstance(scores, dict) or proposal not in scores:
        return ["scorecard_unavailable"]
    raw = scores[proposal]
    if not isinstance(raw, dict):
        return ["scorecard_unavailable"]
    reasons = []
    if raw.get("sufficient") is not True:
        reasons.append("judge_says_insufficient")
    if not raw.get("support_ids"):
        reasons.append("no_cited_support")
    if raw.get("refute_ids"):
        reasons.append("has_cited_refutation")
    if raw.get("missing_facts"):
        reasons.append("has_missing_facts")
    proposed_score = raw.get("score")
    opponents = [
        card.get("score") for c, card in scores.items()
        if c != proposal and isinstance(card, dict)
        and isinstance(card.get("score"), (float, int))
    ]
    if isinstance(proposed_score, (float, int)) and opponents:
        if proposed_score - max(opponents) < margin:
            reasons.append("score_margin")
    else:
        reasons.append("margin_unobservable")
    if step.get("action") == "no_grounded_evidence":
        reasons.append("no_grounded_evidence")
    return reasons or ["policy_path_or_historical_candidate"]


def judge_delta(step: dict, proposal: str, incumbent: str | None) -> float | None:
    scores = step.get("scores") or {}
    if not isinstance(scores, dict) or incumbent not in scores or proposal not in scores:
        return None
    a, b = scores[proposal], scores[incumbent]
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    if not isinstance(a.get("score"), (float, int)) or not isinstance(b.get("score"), (float, int)):
        return None
    return float(a["score"] - b["score"])


def opportunities(row: dict, gold: str, margin: int = 2) -> list[dict]:
    """Only actual proposed *changes* are compared with the previous output.

    A system can select a historical candidate other than this round's proposal.
    Such rerouting is recorded separately from proposal acceptance.
    """
    events = []
    steps = row["traces"]
    for before, after in zip(steps, steps[1:]):
        prev, proposed, selected = (
            choice(before.get("selected_choice")),
            choice(after.get("proposed_choice")),
            choice(after.get("selected_choice")),
        )
        if proposed is None or proposed == prev:
            continue
        category = ("rescue" if prev != gold and proposed == gold else
                    "harm" if prev == gold and proposed != gold else
                    "other")
        outcome = ("adopted" if selected == proposed else
                   "blocked" if selected == prev else "rerouted")
        events.append({
            "round": after["round"], "incumbent": prev, "proposed": proposed,
            "selected": selected, "type": category, "outcome": outcome,
            "score_delta": judge_delta(after, proposed, prev),
            "blocking_reasons": (blocking_reasons(after, proposed, prev, margin)
                                 if outcome == "blocked" else []),
        })
    return events


def case_summary(row: dict, gold: str, margin: int = 2) -> dict:
    stages = row["traces"]
    baseline = choice(stages[0].get("selected_choice"))
    final = choice(stages[-1].get("selected_choice"))
    generated = [choice(r.get("proposed_choice")) for r in stages]
    events = opportunities(row, gold, margin)
    flags = {
        "rescue_blocked": any(e["type"] == "rescue" and e["outcome"] == "blocked"
                              for e in events),
        "harm_accepted": any(e["type"] == "harm" and e["outcome"] == "adopted"
                             for e in events),
        "rescue_accepted": any(e["type"] == "rescue" and e["outcome"] == "adopted"
                               for e in events),
        "harm_blocked": any(e["type"] == "harm" and e["outcome"] == "blocked"
                            for e in events),
        "reader_never_proposed_gold": baseline != gold and gold not in generated,
        "initially_correct_stable": baseline == gold and all(
            e["type"] != "harm" for e in events),
    }
    stratum = next((label for label in STRATA if flags.get(label)), "other")
    evidence = row.get("evidence") or []
    documents = row.get("documents") or []
    return {
        "id": str(row["id"]), "gold": gold, "baseline": baseline, "final": final,
        "baseline_correct": baseline == gold, "final_correct": final == gold,
        "ever_proposed_gold": gold in generated,
        "oracle_candidate_contains_gold": gold in generated,
        "num_rounds": len(stages) - 1,
        "num_claims_saved": len(evidence),
        "num_documents_saved": len(documents),
        "raw_documents_available": bool(documents),
        "raw_document_scope": row.get("document_snapshot_scope",
                                  "full_trace" if documents else "unavailable"),
        "extracted_quotes_available": bool(evidence),
        "stratum": stratum, "strata_flags": flags,
        "events": events,
        "judge_claimed_sufficient_for_gold": any(
            (t.get("scores") or {}).get(gold, {}).get("sufficient") is True
            for t in stages),
        "gold_evidence_citations_proxy": any(
            (t.get("scores") or {}).get(gold, {}).get("support_ids")
            for t in stages),
    }


def auc_pairwise(rescue: list[float], harm: list[float]) -> float | None:
    """Descriptive concordance probability, tie=0.5. No independent-case CI."""
    if not rescue or not harm:
        return None
    s = 0.0
    for positive in rescue:
        for negative in harm:
            s += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return s / (len(rescue) * len(harm))


def load_annotations(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    labels = {}
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("id", ""))
            if not qid or qid in labels:
                raise ValueError(f"Empty/duplicate annotation id at line {n}")
            record = {}
            for field in AUDIT_FIELDS:
                value = obj.get(field, "unknown")
                if value not in AUDIT_CHOICES:
                    raise ValueError(f"Invalid {field}={value!r} for case {qid}")
                record[field] = value
            record["notes"] = str(obj.get("notes", ""))[:2000]
            labels[qid] = record
    return labels


def root_cause_signals(case: dict, annotation: Mapping[str, str] | None) -> list[str]:
    """Conservative diagnostic signals, not exhaustive mutually-exclusive causes."""
    if not annotation:
        return ["independent_evidence_annotation_missing"]
    raw = annotation.get("retrieved_sufficient", "unknown")
    active = annotation.get("active_context_sufficient", "unknown")
    signals: list[str] = []
    if raw == "no":
        signals.append("retrieval_not_sufficient")
    if raw == "yes" and active == "no":
        signals.append("extraction_or_context_selection_loss")
    if active == "yes" and not case["ever_proposed_gold"]:
        signals.append("reader_no_correct_candidate_given_annotated_context")
    if case["strata_flags"]["rescue_blocked"]:
        signals.append("gate_blocked_gold_proposal")
    if annotation.get("quote_entailment") == "no":
        signals.append("quotation_does_not_entail_extracted_claim")
    if annotation.get("source_quality") == "no":
        signals.append("source_reliability_problem")
    return signals or ["undetermined"]


def report_cases(rows: list[dict], gold: Mapping[str, str],
                 *, margin: int = 2, annotations: Mapping[str, dict] | None = None
                 ) -> tuple[dict, list[dict]]:
    labels = {str(k): choice(v) for k, v in gold.items()}
    missing = [str(row["id"]) for row in rows if str(row["id"]) not in labels]
    if missing:
        raise KeyError(f"Missing gold label for {missing[:6]}")
    annotations = annotations or {}
    cases = [case_summary(row, labels[str(row["id"])], margin) for row in rows]
    event_counts = Counter()
    reasons = Counter()
    deltas = defaultdict(list)
    for case in cases:
        case["annotation"] = annotations.get(case["id"], None)
        case["root_cause_signals"] = root_cause_signals(case, case["annotation"])
        for event in case["events"]:
            event_counts[event["type"] + "_" + event["outcome"]] += 1
            for reason in event["blocking_reasons"]:
                reasons[event["type"] + ":" + reason] += 1
            if event["type"] in ("rescue", "harm") and event["score_delta"] is not None:
                deltas[event["type"]].append(event["score_delta"])
    n = len(cases)
    audited = sum(case["annotation"] is not None and
                  case["annotation"]["retrieved_sufficient"] != "unknown"
                  for case in cases)
    stats = {
        "n": n, "no_gold_used_during_inference": True,
        "note": "All correctness labels are OFFLINE. Sufficiency needs independent source review.",
        "baseline_accuracy": sum(c["baseline_correct"] for c in cases) / n,
        "final_accuracy": sum(c["final_correct"] for c in cases) / n,
        "final_harm_count": sum(c["baseline_correct"] and not c["final_correct"] for c in cases),
        "final_rescue_count": sum(not c["baseline_correct"] and c["final_correct"] for c in cases),
        "net_correct_delta": sum(c["final_correct"] - c["baseline_correct"] for c in cases),
        "final_changed_choice_count": sum(c["baseline"] != c["final"] for c in cases),
        "candidate_oracle_accuracy": sum(c["ever_proposed_gold"] for c in cases) / n,
        "reader_never_proposed_gold_count": sum(
            c["strata_flags"]["reader_never_proposed_gold"] for c in cases),
        "cases_with_raw_documents": sum(c["raw_documents_available"] for c in cases),
        "raw_document_scopes": dict(Counter(c["raw_document_scope"] for c in cases)),
        "cases_with_extracted_quotes": sum(c["extracted_quotes_available"] for c in cases),
        "independently_annotated_retrieval_cases": audited,
        "by_primary_stratum": dict(Counter(c["stratum"] for c in cases)),
        "by_overlapping_strata_flags": {
            k: sum(c["strata_flags"].get(k, False) for c in cases)
            for k in STRATA if k != "other"},
        "proposal_transition_counts": dict(event_counts),
        "blocking_reasons_multiple_per_event": dict(reasons),
        "judge_delta_auc_rescue_vs_harm": auc_pairwise(deltas["rescue"], deltas["harm"]),
        "judge_delta_support_counts": {
            "rescue": len(deltas["rescue"]), "harm": len(deltas["harm"])},
        "root_cause_signal_counts": dict(Counter(
            signal for case in cases for signal in case["root_cause_signals"])),
    }
    return stats, cases


def stratified_sample(cases: list[dict], per_stratum: int, seed: int = 2026) -> list[dict]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    by_stratum: dict[str, list] = defaultdict(list)
    for case in cases:
        by_stratum[case["stratum"]].append(case)
    selected: list[dict] = []
    for stratum in STRATA:
        pool = by_stratum.get(stratum, [])
        pool.sort(key=lambda c: sha256(
            f"{seed}/{stratum}/{c['id']}".encode("utf-8")).hexdigest())
        selected.extend(pool[:per_stratum])
    return selected


def annotation_template(case: dict) -> dict:
    return {
        "id": case["id"],
        "retrieved_sufficient": "unknown",
        "active_context_sufficient": "unknown",
        "quote_entailment": "unknown",
        "source_quality": "unknown",
        "notes": "",
    }


def audit_casebook(rows: list[dict], selected: list[dict]) -> str:
    """Plain Markdown; excerpt truncation is for display ONLY, not evidence audit."""
    data = {str(row["id"]): row for row in rows}
    lines = [
        "# EM-RAG fixed-sample mechanism audit",
        "",
        "Gold labels appear here ONLY for *offline analysis*; never use this "
        "casebook as retrieval/generation model input.",
        "",
        "Use three independent judgements: (i) were RAW retrieved passages "
        "sufficient? (ii) did the ACTIVE evidence subset preserve those facts? "
        "(iii) did the Reader propose the correct option, and what blocked the gate?",
        "",
        "If raw documents were not saved and no external source corpus can be "
        "re-opened, mark retrieved_sufficient=unknown; extraction absence does "
        "NOT establish retrieval failure.",
        "",
    ]
    for case in selected:
        row = data[case["id"]]
        lines.extend([
            f"## {case['id']} — {case['stratum']}",
            "",
            f"**Gold**: {case['gold']} (offline only); "
            f"**baseline**: {case['baseline']}; **final**: {case['final']}",
            "",
            row.get("question", ""),
            "",
            "Options: " + json.dumps(row.get("options", {}), ensure_ascii=False),
            "",
            f"**Document snapshot scope:** {case['raw_document_scope']}",
            "",
            "### Candidate and gate events",
            "",
        ])
        events = case["events"]
        if not events:
            lines.append("(No candidate-change opportunity recorded.)")
        for event in events:
            lines.append(
                f"- Round {event['round']}: {event['incumbent']} -> "
                f"{event['proposed']} ({event['type']}/{event['outcome']}); "
                f"score delta={event['score_delta']}; "
                f"blocked by={event['blocking_reasons']}"
            )
        lines.extend(["", "### Source quotes (model extracted)", ""])
        for e in (row.get("evidence") or [])[:20]:
            lines.append(f"- [{e.get('id')}] {e.get('source')}: "
                         f"{e.get('quote','')[:650]}")
        lines.extend(["", "### Retrieved passages (may be missing in legacy cache)", ""])
        if not row.get("documents"):
            lines.append("(Not recorded. You MUST NOT infer retrieval insufficiency.)")
        for d in (row.get("documents") or [])[:16]:
            text = str(d.get("contents", ""))
            lines.append(
                f"- {d.get('id')} / {d.get('source')} / query={d.get('query')}: "
                f"{text[:1300]}" + (" [DISPLAY TRUNCATED]" if len(text) > 1300 else "")
            )
        lines.extend(["", "---", ""])
    return "\n".join(lines)
