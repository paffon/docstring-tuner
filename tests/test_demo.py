"""Test the demo's side-by-side rendering (no models required)."""

from __future__ import annotations

import pytest

from docstring_tuner.demo import _print_example
from docstring_tuner.judge import JudgeScore


def test_print_example_shows_both_variants(capsys: pytest.CaptureFixture[str]) -> None:
    row = {
        "code": "def f():\n    return 1",
        "reference": "Return one.",
        "base": "returns a number",
        "tuned": "Return one.\n\nReturns:\n    int: one.",
    }
    _print_example(1, row, JudgeScore(4.0, 4.0, 3.0, "weak"), JudgeScore(9.0, 8.0, 10.0, "good"))
    out = capsys.readouterr().out
    assert "FUNCTION" in out
    assert "BASE docstring" in out
    assert "FINE-TUNED docstring" in out
    assert "returns a number" in out
    assert "int: one." in out
    assert "weak" in out and "good" in out
