"""Cheap lexical metrics for the eval harness.

============================================================================
HANDS-ON MODULE (learning phase 7) — you implement both functions below.
Make ``pytest tests/test_metrics.py`` pass, then commit.
============================================================================

Two functions:

1. ``rouge_l(candidate, reference)`` — ROUGE-L F1, the length of the Longest Common
   Subsequence (LCS) of tokens, turned into precision/recall/F1.

   - Tokenize both strings the same way. Use the provided :func:`tokenize` (lowercased
     ``\\w+`` runs) so the tests are deterministic.
   - Compute the LCS **length** of the two token lists with the classic dynamic-programming
     table: ``dp[i][j] = dp[i-1][j-1] + 1`` if tokens match, else ``max(dp[i-1][j],
     dp[i][j-1])``.
   - ``precision = lcs / len(candidate_tokens)``, ``recall = lcs / len(reference_tokens)``,
     ``F1 = 2*precision*recall / (precision + recall)``.
   - Return ``0.0`` if either side is empty or precision+recall is 0.

2. ``is_google_style(docstring)`` — our operational format-compliance check. Return True iff:

   - the stripped docstring is non-empty, AND
   - its first non-empty line (the summary) is present and is NOT itself a section header, AND
   - at least one line equals a known section header (see :data:`SECTION_HEADERS`).

   (A bare one-line docstring is intentionally treated as *not* Google-style here, because the
   task is to produce the sectioned ``Args:``/``Returns:`` format.)
"""

from __future__ import annotations

import re
from typing import List

SECTION_HEADERS: frozenset[str] = frozenset(
    {"Args:", "Arguments:", "Returns:", "Return:", "Yields:", "Raises:"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase word-run tokenization shared by the metric and its tests."""
    return re.findall(r"\w+", text.lower())


def calc_new_value(
    tokenized_candidate: List[str], tokenized_reference: List[str],
    row: int, col: int,
    dp: List[List[int]]) -> int:
  """
  Dynamic value caluclation for LCS problem
  """
  if tokenized_candidate[row - 1] == tokenized_reference[col - 1]:
    return dp[row - 1][col - 1] + 1

  return max(dp[row - 1][col], dp[row][col - 1])


def rouge_l(candidate: str, reference: str) -> float:
  """Return the ROUGE-L F1 (LCS-based) between candidate and reference, in ``[0, 1]``."""
  tokenized_candidate: List[str] = tokenize(candidate)
  tokenized_reference: List[str] = tokenize(reference)
  
  if not tokenized_candidate or not tokenized_reference:
    return 0

  length_candidate: int = len(tokenized_candidate)
  length_reference: int = len(tokenized_reference)

  dp: List[List[int]] = [
     [0 for _ in range(length_reference + 1)] for _ in range(length_candidate + 1)
  ]

  for row in range(1, length_candidate + 1):
    for col in range(1, length_reference + 1):
      dp[row][col] = calc_new_value(tokenized_candidate, tokenized_reference, row, col, dp)

  if dp[-1][-1] == 0:
    return 0

  precision = dp[-1][-1] / length_candidate
  recall = dp[-1][-1] / length_reference

  f1 = 2 * precision * recall / (precision + recall)

  return f1


def is_google_style(docstring: str) -> bool:
  """Return True if ``docstring`` looks like a sectioned Google-style docstring."""
  lines = [line.strip() for line in docstring.strip().splitlines() if line.strip()]

  if not lines:
      return False

  if lines[0] in SECTION_HEADERS:
      return False

  return any(line in SECTION_HEADERS for line in lines[1:])