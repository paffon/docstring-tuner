"""Spec for the AST docstring-stripper (learning phase 2).

Implement `docstring_tuner.ast_utils.split_function` until these pass. The tests check
structure (does it parse? is the docstring gone? is the body kept?) rather than exact
formatting, because `ast.unparse` normalizes whitespace.
"""

from __future__ import annotations

import ast

from docstring_tuner.ast_utils import split_function

FUNC_WITH_DOC = '''def greet(name):
    """Return a greeting for the given name.

    Args:
        name: Who to greet.

    Returns:
        A greeting string.
    """
    return "Hello, " + name
'''

ASYNC_FUNC = '''async def fetch(url):
    """Fetch a URL."""
    return url
'''

DOCSTRING_ONLY = '''def todo():
    """Not implemented yet."""
'''

NO_DOCSTRING = "def add(a, b):\n    return a + b\n"
NO_FUNCTION = "x = 1 + 2\n"
SYNTAX_ERROR = "def broken(:\n    pass\n"


def _remaining_docstring(code: str) -> str | None:
    tree = ast.parse(code)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef)
    return ast.get_docstring(func)


def test_returns_code_and_docstring() -> None:
    result = split_function(FUNC_WITH_DOC)
    assert result is not None
    _, doc = result
    assert "Return a greeting for the given name." in doc
    assert "Args:" in doc


def test_docstring_removed_from_code() -> None:
    result = split_function(FUNC_WITH_DOC)
    assert result is not None
    code, _ = result
    assert "Return a greeting for the given name." not in code
    assert _remaining_docstring(code) is None


def test_signature_and_body_preserved() -> None:
    result = split_function(FUNC_WITH_DOC)
    assert result is not None
    code, _ = result
    assert "def greet(name):" in code
    assert "'Hello, ' + name" in code  # ast.unparse normalizes to single quotes
    ast.parse(code)  # still valid Python


def test_async_function_supported() -> None:
    result = split_function(ASYNC_FUNC)
    assert result is not None
    code, doc = result
    assert doc.strip() == "Fetch a URL."
    assert code.startswith("async def fetch(url):")
    assert _remaining_docstring(code) is None


def test_docstring_only_function_keeps_valid_body() -> None:
    result = split_function(DOCSTRING_ONLY)
    assert result is not None
    code, doc = result
    assert doc.strip() == "Not implemented yet."
    ast.parse(code)  # must still parse with an empty-but-valid body
    assert _remaining_docstring(code) is None


def test_none_when_no_docstring() -> None:
    assert split_function(NO_DOCSTRING) is None


def test_none_when_no_function() -> None:
    assert split_function(NO_FUNCTION) is None


def test_none_on_syntax_error() -> None:
    assert split_function(SYNTAX_ERROR) is None
