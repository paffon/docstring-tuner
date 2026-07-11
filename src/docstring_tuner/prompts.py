"""Prompt construction for the docstring task.

============================================================================
HANDS-ON MODULE (learning phase 1) — you implement the three items below.
Make ``pytest tests/test_prompts.py`` pass, then commit.
============================================================================

The model's job: given a Python function whose docstring has been removed, write a
**Google-style** docstring for it. A Google-style docstring looks like::

    \"\"\"One-line summary in the imperative mood.

    Optional longer description.

    Args:
        x: What x is.
        y: What y is.

    Returns:
        What the function returns.

    Raises:
        ValueError: When something is wrong.
    \"\"\"

What you need to build:

1. ``SYSTEM_PROMPT`` — a string that tells the model exactly this: it is a Python
   documentation assistant; it receives a function and must output ONLY a Google-style
   docstring (no code fences, no surrounding function, no commentary); and it should
   spell out the sections (summary line, ``Args:``, ``Returns:``, ``Raises:`` when
   relevant). The tests check that it mentions "google", "args", and "returns".

2. ``user_prompt(code)`` — wrap the function source in a short request and include the
   source verbatim (the tests check the code appears in the output).

3. ``build_messages(code)`` — return the chat message list ``[system, user]`` where each
   message is ``{"role": ..., "content": ...}``. This is what the tokenizer's chat
   template consumes at generation and training time.

Tip: keep ``SYSTEM_PROMPT`` short and imperative. The fine-tune's job is to make the model
reliably follow it; an overlong prompt just burns context.
"""

from __future__ import annotations

Message = dict[str, str]

SYSTEM_PROMPT: str = """
You're a Python documentation assistant. Given a function, output ONLY a
Google-style docstring with summary, Args, Returns, and Raises sections as
appropriate.

Google-style docstring format:
\"\"\"One-line summary in the imperative mood.

Optional longer description.

Args:
    x: What x is.
    y: What y is.

Returns:
    What the function returns.

Raises:
    ValueError: When something is wrong.
\"\"\"
"""


def user_prompt(code: str) -> str:
    """Build the user turn: present ``code`` and ask for a Google-style docstring.

    The returned string MUST contain ``code`` verbatim.
    """
    return f"""
The following is the code for which the Google-style docstring is requested.
Please provide the docstring only, without any code fences or surrounding function.
Code:
```
{code}
```

"""


def build_messages(code: str) -> list[Message]:
    """Assemble the chat messages for one function.

    Return ``[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": user_prompt(code)}]``.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(code)},
    ]
