"""LLM-as-judge behind a small, swappable, typed interface.

The :class:`Judge` protocol has a single method, ``score(code, reference, candidate)``.
Three implementations ship:

- :class:`ClaudeCliJudge` — shells out to the local ``claude`` CLI (``claude -p --model haiku``).
  No API key, no vendor lock-in. This is the default when ``claude`` is on PATH.
- :class:`OpenAICompatibleJudge` — POSTs to any OpenAI-compatible ``/chat/completions`` endpoint
  (base URL + key from ``DT_OPENAI_BASE_URL`` / ``DT_OPENAI_API_KEY``). Works against OpenAI,
  LiteLLM, Azure, or a local server.
- :class:`MockJudge` — a deterministic, offline judge (string similarity + a format heuristic)
  for tests and no-network smoke runs.

Every backend returns a :class:`JudgeScore` with three 0–10 rubric criteria and a short rationale.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

SECTION_MARKERS: tuple[str, ...] = ("Args:", "Returns:", "Yields:", "Raises:")


class JudgeError(RuntimeError):
    """Raised when a judge backend fails to produce a parseable score."""


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """Rubric scores (each 0–10) plus a short rationale."""

    faithfulness: float  # does the docstring correctly describe what the code does?
    completeness: float  # are the args / returns / raises covered?
    format: float  # is it valid Google style?
    rationale: str

    @property
    def overall(self) -> float:
        """Mean of the three rubric criteria."""
        return (self.faithfulness + self.completeness + self.format) / 3.0


@runtime_checkable
class Judge(Protocol):
    """Anything that can score a candidate docstring against a reference."""

    def score(self, code: str, reference: str, candidate: str) -> JudgeScore: ...


RUBRIC = (
    "You are grading a Python docstring. Compare the CANDIDATE docstring to the reference and "
    "the function code, and rate it on three criteria, each an integer from 0 to 10:\n"
    "- faithfulness: does it correctly describe what the code does (no hallucinated behavior)?\n"
    "- completeness: are the parameters, return value, and raised exceptions covered?\n"
    "- format: is it valid Google style (summary line, then Args:/Returns:/Raises: sections)?\n"
    'Respond with ONLY a JSON object: {"faithfulness": int, "completeness": int, '
    '"format": int, "rationale": "one short sentence"}. No other text.'
)


def build_judge_prompt(code: str, reference: str, candidate: str) -> str:
    """Assemble the full judging prompt."""
    return (
        f"{RUBRIC}\n\n"
        f"### FUNCTION CODE\n{code}\n\n"
        f"### REFERENCE DOCSTRING\n{reference}\n\n"
        f"### CANDIDATE DOCSTRING\n{candidate}\n"
    )


def _clamp10(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, number))


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a possibly chatty model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeError(f"no JSON object in judge output: {text[:120]!r}")
    parsed: dict[str, Any] = json.loads(text[start : end + 1])
    return parsed


def _payload_to_score(payload: dict[str, Any]) -> JudgeScore:
    return JudgeScore(
        faithfulness=_clamp10(payload.get("faithfulness")),
        completeness=_clamp10(payload.get("completeness")),
        format=_clamp10(payload.get("format")),
        rationale=str(payload.get("rationale", "")).strip(),
    )


class ClaudeCliJudge:
    """Judge backed by the local ``claude`` CLI."""

    def __init__(self, model: str = "haiku", timeout: float = 60.0) -> None:
        self.model = model
        self.timeout = timeout

    def _run(self, prompt: str) -> str:
        executable = shutil.which("claude")
        if executable is None:
            raise JudgeError("`claude` CLI not found on PATH")
        proc = subprocess.run(
            [executable, "-p", "--model", self.model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise JudgeError(f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:200]}")
        return proc.stdout

    def score(self, code: str, reference: str, candidate: str) -> JudgeScore:
        prompt = build_judge_prompt(code, reference, candidate)
        last_error: Exception | None = None
        for attempt in range(2):
            text = self._run(prompt if attempt == 0 else prompt + "\n\nReturn ONLY the JSON.")
            try:
                return _payload_to_score(_extract_json(text))
            except (JudgeError, json.JSONDecodeError) as error:
                last_error = error
        raise JudgeError(f"could not parse claude output: {last_error}")


class OpenAICompatibleJudge:
    """Judge backed by any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 60.0) -> None:
        self.model = model
        self.timeout = timeout
        self.base_url = os.getenv("DT_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("DT_OPENAI_API_KEY", "")

    def _run(self, prompt: str) -> str:
        if not self.api_key:
            raise JudgeError("DT_OPENAI_API_KEY is not set")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        content: str = payload["choices"][0]["message"]["content"]
        return content

    def score(self, code: str, reference: str, candidate: str) -> JudgeScore:
        text = self._run(build_judge_prompt(code, reference, candidate))
        return _payload_to_score(_extract_json(text))


class MockJudge:
    """Deterministic offline judge: string similarity + a Google-format heuristic."""

    def score(self, code: str, reference: str, candidate: str) -> JudgeScore:
        similarity = difflib.SequenceMatcher(None, candidate, reference).ratio()
        content_score = round(10.0 * similarity, 2)
        sectioned = any(marker in candidate for marker in SECTION_MARKERS)
        return JudgeScore(
            faithfulness=content_score,
            completeness=content_score,
            format=10.0 if sectioned else 3.0,
            rationale=f"offline mock: similarity={similarity:.2f}, sectioned={sectioned}",
        )


def resolve_backend(backend: str) -> str:
    """Resolve ``auto`` to a concrete backend based on what's available."""
    backend = backend.lower()
    if backend != "auto":
        return backend
    if shutil.which("claude") is not None:
        return "claude_cli"
    if os.getenv("DT_OPENAI_API_KEY"):
        return "openai"
    return "mock"


def make_judge(backend: str = "auto", model: str = "haiku") -> Judge:
    """Construct a judge for the given backend (``auto`` picks the best available)."""
    resolved = resolve_backend(backend)
    if resolved == "claude_cli":
        return ClaudeCliJudge(model=model)
    if resolved == "openai":
        return OpenAICompatibleJudge(model=model if model != "haiku" else "gpt-4o-mini")
    if resolved == "mock":
        return MockJudge()
    raise ValueError(f"Unknown judge backend: {backend!r}")
