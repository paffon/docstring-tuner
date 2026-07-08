"""Tests for the pure (network-free) helpers in the data pipeline."""

from __future__ import annotations

from pathlib import Path

from docstring_tuner.config import Config
from docstring_tuner.data import (
    Example,
    assert_disjoint,
    normalize_docstring,
    passes_filters,
    read_jsonl,
    write_jsonl,
)

CFG = Config().data


def test_normalize_docstring_dedents_and_strips() -> None:
    raw = "Do a thing.\n\n    Args:\n        x: The x value.\n    "
    out = normalize_docstring(raw)
    assert out.startswith("Do a thing.")
    assert "\n    Args:" not in out  # continuation indentation removed
    assert "Args:" in out
    assert out == out.strip()


def test_passes_filters_accepts_structured_docstring() -> None:
    code = "def f(x):\n    return x"
    doc = "Do a thing with x.\n\nArgs:\n    x: The x value to use."
    assert passes_filters(code, doc, CFG)


def test_passes_filters_rejects_short_docstring() -> None:
    assert not passes_filters("def f(): ...", "Too short.", CFG)


def test_passes_filters_rejects_docstring_without_google_markers() -> None:
    doc = "A long prose docstring with no section headers at all, quite verbose really."
    assert not passes_filters("def f(): ...", doc, CFG)


def test_passes_filters_rejects_oversized_source() -> None:
    big_code = "x = 1\n" * 1000
    doc = "Summary.\n\nReturns:\n    Something meaningful and long enough to pass."
    assert not passes_filters(big_code, doc, CFG)


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = [
        Example(code="def a(): return 1", docstring="Return one.\n\nReturns:\n    int: one."),
        Example(code="def b(x): return x", docstring="Echo x.\n\nArgs:\n    x: value."),
    ]
    path = tmp_path / "data.jsonl"
    write_jsonl(path, examples)
    assert read_jsonl(path) == examples


def test_assert_disjoint_detects_overlap() -> None:
    shared = Example(code="def dup(): pass", docstring="Duplicate.\n\nReturns:\n    None.")
    assert_disjoint([shared], [])  # no overlap -> no raise
    try:
        assert_disjoint([shared], [shared])
    except AssertionError:
        return
    raise AssertionError("expected assert_disjoint to raise on overlap")
