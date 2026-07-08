"""Spec for the prompts module (learning phase 1).

These tests describe what `docstring_tuner.prompts` must do. Implement the module until
they all pass.
"""

from __future__ import annotations

from docstring_tuner import prompts

SAMPLE_CODE = "def add(a: int, b: int) -> int:\n    return a + b"


def test_build_messages_has_system_then_user() -> None:
    messages = prompts.build_messages(SAMPLE_CODE)
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_user_turn_includes_the_code_verbatim() -> None:
    messages = prompts.build_messages(SAMPLE_CODE)
    assert SAMPLE_CODE in messages[1]["content"]


def test_user_prompt_includes_the_code_verbatim() -> None:
    assert SAMPLE_CODE in prompts.user_prompt(SAMPLE_CODE)


def test_system_prompt_defines_google_style() -> None:
    system = prompts.SYSTEM_PROMPT.lower()
    assert system.strip(), "SYSTEM_PROMPT must not be empty"
    assert "google" in system
    assert "args" in system
    assert "returns" in system


def test_system_message_uses_the_system_prompt() -> None:
    messages = prompts.build_messages(SAMPLE_CODE)
    assert messages[0]["content"] == prompts.SYSTEM_PROMPT
