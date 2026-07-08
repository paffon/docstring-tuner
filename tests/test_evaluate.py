"""Tests for the pure aggregation/reporting helpers in the eval harness."""

from __future__ import annotations

import math

from docstring_tuner.evaluate import (
    ExampleScores,
    VariantSummary,
    aggregate,
    build_report,
    format_table,
)


def test_aggregate_empty_is_zeroed() -> None:
    summary = aggregate([])
    assert summary == VariantSummary(0, 0.0, 0.0, 0.0)


def test_aggregate_means_and_format_rate() -> None:
    scores = [
        ExampleScores(judge_overall=8.0, rouge_l=0.5, google_style=True),
        ExampleScores(judge_overall=6.0, rouge_l=0.3, google_style=False),
    ]
    summary = aggregate(scores)
    assert summary.n == 2
    assert summary.mean_judge == 7.0
    assert math.isclose(summary.mean_rouge_l, 0.4)
    assert summary.format_rate == 0.5


def test_build_report_has_deltas_and_meta() -> None:
    base = VariantSummary(2, 6.0, 0.30, 0.5)
    tuned = VariantSummary(2, 7.5, 0.45, 1.0)
    report = build_report(base, tuned, {"judge_backend": "mock"})
    assert report["n_examples"] == 2
    assert report["base"]["mean_judge"] == 6.0
    assert report["tuned"]["format_rate"] == 1.0
    assert math.isclose(report["deltas"]["mean_judge"], 1.5)
    assert report["judge_backend"] == "mock"


def test_format_table_mentions_metrics_and_columns() -> None:
    table = format_table(VariantSummary(2, 6.0, 0.3, 0.5), VariantSummary(2, 7.5, 0.45, 1.0))
    assert "Judge score" in table
    assert "ROUGE-L" in table
    assert "Base" in table
    assert "Tuned" in table
