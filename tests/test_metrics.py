"""Spec for the lexical metrics (learning phase 7)."""

from __future__ import annotations

import math

from docstring_tuner.metrics import is_google_style, rouge_l

GOOGLE_DOC = "Add two numbers.\n\nArgs:\n    a: First.\n    b: Second.\n\nReturns:\n    The sum."


def test_rouge_l_identical_is_one() -> None:
    assert rouge_l("the quick brown fox", "the quick brown fox") == 1.0


def test_rouge_l_disjoint_is_zero() -> None:
    assert rouge_l("alpha beta", "gamma delta") == 0.0


def test_rouge_l_empty_is_zero() -> None:
    assert rouge_l("", "anything here") == 0.0
    assert rouge_l("anything here", "") == 0.0


def test_rouge_l_partial_overlap_is_between() -> None:
    score = rouge_l("the quick brown fox", "the slow brown fox")
    assert 0.0 < score < 1.0


def test_rouge_l_uses_lcs_not_just_set_overlap() -> None:
    # Same tokens, reversed order -> LCS length 1 of 3 tokens -> F1 = 1/3.
    score = rouge_l("a b c", "c b a")
    assert math.isclose(score, 1 / 3, rel_tol=1e-9)


def test_rouge_l_is_symmetric_for_equal_lengths() -> None:
    a, b = "one two three four", "one two nine four"
    assert math.isclose(rouge_l(a, b), rouge_l(b, a), rel_tol=1e-9)


def test_is_google_style_true_for_sectioned_docstring() -> None:
    assert is_google_style(GOOGLE_DOC)


def test_is_google_style_false_for_one_liner() -> None:
    assert not is_google_style("Add two numbers.")


def test_is_google_style_false_for_empty() -> None:
    assert not is_google_style("   ")


def test_is_google_style_false_when_summary_missing() -> None:
    # Starts straight into a section header with no summary line.
    assert not is_google_style("Args:\n    a: First.")


def test_is_google_style_true_with_only_returns_section() -> None:
    assert is_google_style("Compute a value.\n\nReturns:\n    The value.")
