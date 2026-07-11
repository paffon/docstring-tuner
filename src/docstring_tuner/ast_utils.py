"""Split a function into (source-without-docstring, docstring) using the stdlib ``ast``.

============================================================================
HANDS-ON MODULE (learning phase 2) — you implement ``split_function`` below.
Make ``pytest tests/test_ast_utils.py`` pass, then commit.
============================================================================

This is how each training example is manufactured: the model's *input* is the function
with its docstring removed, and the *target* is the docstring we took out.

Algorithm (all stdlib ``ast``):

1. ``ast.parse(source)`` — if it raises ``SyntaxError``, return ``None`` (skip junk).
2. Find the first top-level ``ast.FunctionDef`` or ``ast.AsyncFunctionDef`` in
   ``tree.body``. If there isn't one, return ``None``.
3. ``doc = ast.get_docstring(func, clean=False)`` — if ``None``, the function has no
   docstring, so return ``None``. (``clean=False`` keeps the raw text; we don't want to
   reformat the target.)
4. Remove the docstring node: if ``func.body[0]`` is an ``ast.Expr`` wrapping an
   ``ast.Constant`` whose value is a ``str``, drop it. If that empties the body, replace
   it with ``[ast.Pass()]`` so the function still parses.
5. Return ``(ast.unparse(func), doc)``.

Notes:
- ``ast.unparse`` (Python 3.9+) reformats whitespace and drops comments — fine for a
  model input, and the reason the tests check structure, not exact byte-equality.
- ``ast.get_docstring`` only recognises a docstring in the very first statement, which is
  exactly the node you remove in step 4 — keep those two in sync.
"""

from __future__ import annotations

import ast  # noqa: F401  -- you'll use this once you implement split_function


def split_function(source: str) -> tuple[str, str] | None:
    """Return ``(code_without_docstring, docstring)`` for the first function in ``source``.

    Returns ``None`` if ``source`` doesn't parse, contains no top-level function, or the
    first function has no docstring.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    first_func: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            first_func = node
            break
    if first_func is None:
        return None

    docstring = ast.get_docstring(first_func, clean=False)
    if docstring is None:
        return None

    # There is at least docstring, so we can remove it from the function body.
    first_stmt = first_func.body[0]
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        # The first statement is a single string constant (the docstring), so we remove it.
        first_func.body.pop(0)

    if not first_func.body:
        # If removing the docstring left the function body empty, inject a 'pass' statement.
        first_func.body = [ast.Pass()]

    return ast.unparse(first_func), docstring
