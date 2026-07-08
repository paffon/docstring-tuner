"""Tests for the dependency-free output cleaning in the generation module."""

from __future__ import annotations

from docstring_tuner.generate import clean_docstring


def test_plain_text_unchanged() -> None:
    assert clean_docstring("  Return the sum.  ") == "Return the sum."


def test_strips_language_code_fence() -> None:
    assert clean_docstring("```python\nReturn the sum.\n```") == "Return the sum."


def test_strips_bare_code_fence() -> None:
    assert clean_docstring("```\nReturn the sum.\n```") == "Return the sum."


def test_strips_surrounding_triple_double_quotes() -> None:
    assert clean_docstring('"""Return the sum."""') == "Return the sum."


def test_strips_surrounding_triple_single_quotes() -> None:
    assert clean_docstring("'''Return the sum.'''") == "Return the sum."


def test_preserves_multiline_body() -> None:
    raw = '"""Summarize.\n\nArgs:\n    x: A value.\n"""'
    assert clean_docstring(raw) == "Summarize.\n\nArgs:\n    x: A value."


def test_strips_fence_then_quotes() -> None:
    assert clean_docstring('```\n"""Do a thing."""\n```') == "Do a thing."
