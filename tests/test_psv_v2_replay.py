"""Exact cached new-source replay: paired interventions without query drift."""
import json
import pytest

from coverage_medrgag import progressive_cli
from tests.test_progressive_cli import (
    fake_orig_medrgag, MockNativeBM25, MockOpenAIChat, common_args
)


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def test_exact_replay_same_new_corpus_evidence_but_different_anchor(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    monkeypatch.setattr(progressive_cli, "OpenAIChat", MockOpenAIChat)
    monkeypatch.setattr(progressive_cli, "NativeMedRGAGBM25", MockNativeBM25)
    live = tmp_path / "live"
    assert progressive_cli.main(common_args(root, live)) == 0
    originals = load(live / "predictions.jsonl")

    # A replay must NOT create any planner or live retrieval clients.
    def do_not_initialize(**kwargs):
        raise AssertionError("Replay attempted a real BM25 query")
    monkeypatch.setattr(progressive_cli, "NativeMedRGAGBM25", do_not_initialize)
    replay = tmp_path / "replay"
    args = common_args(root, replay) + [
        "--replay-source-views", str(live / "predictions.jsonl"),
        "--anchor-mode", "retrieved_only",
        "--paired-source-control",
        "--policy", "contrastive_vote",
    ]
    assert progressive_cli.main(args) == 0
    results = load(replay / "predictions.jsonl")
    assert len(results) == len(originals) == 2
    for old, new in zip(originals, results):
        assert new["source_views_replayed"] is True
        assert new["retrieval_calls"] == 0
        assert new["auxiliary_calls"] == 0
        assert new["source_replay_logical_queries"] == 2
        assert new["reader_calls"] == 4
        assert new["baseline_prompt_sha256"] == old["baseline_prompt_sha256"]
        assert new["base_selected_doc_ids"] == old["base_selected_doc_ids"]
        assert [
            v["fresh_corpus_documents"] for v in new["novel_views"]
        ] == [
            v["fresh_corpus_documents"] for v in old["novel_views"]
        ]
        assert all(v["anchor_provenance"]["anchor_generated_count"] == 0
                   for v in new["novel_views"])
        assert new["matched_source_control"]["reused_B0"] is False


def test_replay_refuses_corrupted_baseline_prompt_fingerprint(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    monkeypatch.setattr(progressive_cli, "OpenAIChat", MockOpenAIChat)
    monkeypatch.setattr(progressive_cli, "NativeMedRGAGBM25", MockNativeBM25)
    live = tmp_path / "live"
    assert progressive_cli.main(common_args(root, live)) == 0
    rows = load(live / "predictions.jsonl")
    for row in rows:
        row["baseline_prompt_sha256"] = "corrupted"
    wrong = tmp_path / "invalid.jsonl"
    wrong.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="prompt SHA mismatch"):
        progressive_cli.main(
            common_args(root, tmp_path / "refuse")
            + ["--replay-source-views", str(wrong)])
