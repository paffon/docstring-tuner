"""Tests for the offline judge internals: scoring, JSON parsing, backend selection."""

from __future__ import annotations

from docstring_tuner.judge import (
    Judge,
    JudgeScore,
    MockJudge,
    _extract_json,
    _payload_to_score,
    make_judge,
    resolve_backend,
)

SECTIONED = "Return the sum.\n\nReturns:\n    int: the sum."


def test_overall_is_mean_of_criteria() -> None:
    assert JudgeScore(3.0, 6.0, 9.0, "").overall == 6.0


def test_extract_json_from_chatty_text() -> None:
    assert _extract_json('Sure! {"faithfulness": 8} done.') == {"faithfulness": 8}


def test_payload_to_score_clamps_and_defaults() -> None:
    score = _payload_to_score({"faithfulness": 15, "completeness": -2, "format": "x"})
    assert score.faithfulness == 10.0
    assert score.completeness == 0.0
    assert score.format == 0.0
    assert score.rationale == ""


def test_mock_judge_is_deterministic() -> None:
    a = MockJudge().score("code", SECTIONED, SECTIONED)
    b = MockJudge().score("code", SECTIONED, SECTIONED)
    assert a == b


def test_mock_judge_rewards_similarity_and_format() -> None:
    close = MockJudge().score("code", SECTIONED, SECTIONED)
    far = MockJudge().score("code", SECTIONED, "unrelated blah blah")
    assert close.faithfulness > far.faithfulness
    assert close.format == 10.0  # candidate has a Returns: section
    assert far.format == 3.0


def test_mock_judge_satisfies_protocol() -> None:
    assert isinstance(MockJudge(), Judge)


def test_resolve_backend_passthrough() -> None:
    assert resolve_backend("mock") == "mock"
    assert resolve_backend("openai") == "openai"


def test_make_judge_mock() -> None:
    judge = make_judge("mock")
    assert isinstance(judge, MockJudge)
    assert isinstance(judge.score("c", "r", "c"), JudgeScore)
