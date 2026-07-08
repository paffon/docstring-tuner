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

SECTION_HEADERS: frozenset[str] = frozenset(
    {"Args:", "Arguments:", "Returns:", "Return:", "Yields:", "Raises:"}
)


def tokenize(text: str) -> list[str]:
    """Lowercase word-run tokenization shared by the metric and its tests."""
    return re.findall(r"\w+", text.lower())


def rouge_l(candidate: str, reference: str) -> float:
    """Return the ROUGE-L F1 (LCS-based) between candidate and reference, in ``[0, 1]``."""
    raise NotImplementedError("Implement rouge_l (learning phase 7)")


def is_google_style(docstring: str) -> bool:
    """Return True if ``docstring`` looks like a sectioned Google-style docstring."""
    raise NotImplementedError("Implement is_google_style (learning phase 7)")
