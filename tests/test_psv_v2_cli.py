"""CLI protocol isolation, gold-free reparse and cross-variant diagnostics."""
import json
import pytest

from coverage_medrgag import progressive_cli
from tests.test_progressive_cli import (
    fake_orig_medrgag, MockNativeBM25, MockOpenAIChat, common_args
)


def _rows(out):
    return [json.loads(x) for x in
            (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]


def test_psv_v2_run_diagnostics_and_paired_historical_control(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    monkeypatch.setattr(progressive_cli, "OpenAIChat", MockOpenAIChat)
    monkeypatch.setattr(progressive_cli, "NativeMedRGAGBM25", MockNativeBM25)
    old = tmp_path / "original"
    new = tmp_path / "source_only"
    assert progressive_cli.main(
        common_args(root, old)
        + ["--query-size-policy", "reject"]) == 0
    params = common_args(root, new) + [
        "--anchor-mode", "retrieved_only",
        "--paired-source-control", "--policy", "contrastive_vote",
        "--fresh-ranking", "target_relation",
    ]
    assert progressive_cli.main(params) == 0
    records = _rows(new)
    assert len(records) == 2
    assert all(row["base_prediction"] == "A" for row in records)
    assert all(row["anchor_generated_count"] == 0 for row in records)
    assert all(row["matched_source_control"]["prediction"] == "A"
               for row in records)
    assert all(row["reader_calls"] == 4 for row in records)
    assert all(row["auxiliary_calls"] == 2 for row in records)
    assert all(row["retrieval_calls"] == 2 for row in records)
    assert all(row["prediction"] == "B" for row in records)

    analysis = tmp_path / "eval.json"
    assert progressive_cli.main([
        "evaluate", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(new / "predictions.jsonl"),
        "--compare", str(old / "predictions.jsonl"),
        "--bootstrap", "30", "--output", str(analysis)]) == 0
    report = json.loads(analysis.read_text())
    assert report["diagnostics"]["matched_control_sample_count"] == 2
    assert report["diagnostics"]["matched_control_vs_first_new_view"][
        "paired_n"] == 2
    assert report["diagnostics"]["anchor_mode_counts"]["retrieved_only"] == 2
    assert report["cross_run"]["final_budget_paired"]["paired_n"] == 2
    assert report["by_budget"]["2"]["mean_reader_calls"] == 4


def test_offline_reparse_does_not_read_gold_and_preserves_original(tmp_path, monkeypatch):
    root = fake_orig_medrgag(tmp_path)
    monkeypatch.setattr(progressive_cli, "OpenAIChat", MockOpenAIChat)
    monkeypatch.setattr(progressive_cli, "NativeMedRGAGBM25", MockNativeBM25)
    out = tmp_path / "initial"
    assert progressive_cli.main(common_args(root, out)) == 0
    old = (out / "predictions.jsonl").read_bytes()
    dest = tmp_path / "frozen_reparsed.jsonl"
    assert progressive_cli.main([
        "reparse", "--dataset", "pubmedqa",
        "--benchmark", str(root / "benchmark.json"),
        "--predictions", str(out / "predictions.jsonl"),
        "--output", str(dest)]) == 0
    assert (out / "predictions.jsonl").read_bytes() == old
    assert len(dest.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        progressive_cli.main([
            "reparse", "--dataset", "pubmedqa",
            "--benchmark", str(root / "benchmark.json"),
            "--predictions", str(out / "predictions.jsonl"),
            "--output", str(dest)])


def test_invalid_contrastive_control_flag_fails_at_cli(tmp_path):
    root = fake_orig_medrgag(tmp_path)
    with pytest.raises(ValueError, match="requires matched source control"):
        progressive_cli.main(
            common_args(root, tmp_path / "out")
            + ["--policy", "contrastive_vote"])
