"""Base-vs-fine-tuned evaluation: the centerpiece.

Scores each generated docstring three ways — the LLM judge (mean of its rubric), ROUGE-L
against the reference, and the boolean Google-style format check — then prints a base-vs-tuned
comparison table and writes ``artifacts/eval_report.json``.

Two modes:

- **score existing generations** (default): reads ``artifacts/generations.jsonl`` (produced on
  Colab), so this step is CPU-only and needs no GPU.
- **generate then score** (``--generate``): runs both models over the held-out test set first.
  Works on GPU (fast) or CPU (slow fallback).

Run: ``python -m docstring_tuner.evaluate``  (add ``--judge mock`` for an offline smoke test).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from subprocess import TimeoutExpired
from typing import Any

from tqdm import tqdm

from .config import Config
from .judge import Judge, JudgeError, make_judge, resolve_backend
from .metrics import is_google_style, rouge_l


@dataclass(frozen=True, slots=True)
class ExampleScores:
    """The three scores for one generated docstring."""

    judge_overall: float
    rouge_l: float
    google_style: bool


@dataclass(frozen=True, slots=True)
class VariantSummary:
    """Aggregated scores for one model variant (base or tuned)."""

    n: int
    mean_judge: float
    mean_rouge_l: float
    format_rate: float


def score_candidate(judge: Judge, code: str, reference: str, candidate: str) -> ExampleScores:
    """Score one candidate docstring with the judge, ROUGE-L, and the format check."""
    verdict = judge.score(code, reference, candidate)
    return ExampleScores(
        judge_overall=verdict.overall,
        rouge_l=rouge_l(candidate, reference),
        google_style=is_google_style(candidate),
    )


def score_variant(
    judge: Judge, rows: list[dict[str, Any]], key: str, workers: int
) -> list[ExampleScores]:
    """Score one variant (``key`` = ``base``/``tuned``) across all rows, with a progress bar.

    Each judge call is a blocking subprocess/HTTP request (I/O-bound), so a thread pool gives
    near-linear speedup despite the GIL. We advance the ``tqdm`` bar as each call *completes*
    (via ``as_completed``) for an honest ETA, then write results back by their original index so
    the returned scores stay aligned with ``rows``. ``workers=1`` runs strictly serially.
    """

    def _score(row: dict[str, Any]) -> ExampleScores:
        return score_candidate(judge, row["code"], row["reference"], row[key])

    results: list[ExampleScores | None] = [None] * len(rows)
    failures = 0

    def _store(index: int, produce: Callable[[], ExampleScores]) -> None:
        """Run one scoring call; a single judge failure is skipped, not fatal."""
        nonlocal failures
        try:
            results[index] = produce()
        except (JudgeError, TimeoutExpired) as error:
            failures += 1
            tqdm.write(f"  judge {key}[{index}] failed, skipping: {error}")

    bar = tqdm(total=len(rows), desc=f"judge {key}", unit="ex")
    if workers <= 1:
        for index, row in enumerate(rows):
            _store(index, lambda row=row: _score(row))
            bar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            index_of = {pool.submit(_score, row): i for i, row in enumerate(rows)}
            for future in as_completed(index_of):
                _store(index_of[future], future.result)
                bar.update(1)
    bar.close()

    if failures:
        tqdm.write(f"  {failures}/{len(rows)} {key} judge calls failed and were skipped")
    return [score for score in results if score is not None]


def aggregate(scores: list[ExampleScores]) -> VariantSummary:
    """Reduce per-example scores to means and a format-compliance rate."""
    if not scores:
        return VariantSummary(n=0, mean_judge=0.0, mean_rouge_l=0.0, format_rate=0.0)
    return VariantSummary(
        n=len(scores),
        mean_judge=mean(s.judge_overall for s in scores),
        mean_rouge_l=mean(s.rouge_l for s in scores),
        format_rate=sum(s.google_style for s in scores) / len(scores),
    )


def build_report(
    base: VariantSummary, tuned: VariantSummary, meta: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the machine-readable JSON report."""
    return {
        **meta,
        "n_examples": tuned.n,
        "base": asdict(base),
        "tuned": asdict(tuned),
        "deltas": {
            "mean_judge": tuned.mean_judge - base.mean_judge,
            "mean_rouge_l": tuned.mean_rouge_l - base.mean_rouge_l,
            "format_rate": tuned.format_rate - base.format_rate,
        },
    }


def format_table(base: VariantSummary, tuned: VariantSummary) -> str:
    """Render the base-vs-tuned comparison as an aligned text table."""
    rows = [
        ("Judge score (0-10)", base.mean_judge, tuned.mean_judge),
        ("ROUGE-L (F1)", base.mean_rouge_l, tuned.mean_rouge_l),
        ("Format-compliance", base.format_rate, tuned.format_rate),
    ]
    lines = [f"{'Metric':<20} {'Base':>9} {'Tuned':>9} {'Delta':>10}", "-" * 51]
    for name, base_value, tuned_value in rows:
        lines.append(
            f"{name:<20} {base_value:>9.3f} {tuned_value:>9.3f} {tuned_value - base_value:>+10.3f}"
        )
    return "\n".join(lines)


def load_generations(path: Path) -> list[dict[str, Any]]:
    """Read the generations JSONL (one object per line with code/reference/base/tuned)."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_rows(cfg: Config, limit: int | None) -> list[dict[str, str]]:
    """Generate base and tuned docstrings over the held-out test set (needs the model stack)."""
    from .data import read_jsonl
    from .generate import generate_docstrings
    from .model import load_base, load_base_with_adapter, load_tokenizer

    test = read_jsonl(Path(cfg.data.cache_dir) / "test.jsonl")
    if limit is not None:
        test = test[:limit]
    codes = [example.code for example in test]
    references = [example.docstring for example in test]
    model_id = cfg.model.base_model_id
    max_new = cfg.model.max_new_tokens

    base_model = load_base(model_id)
    base_out = generate_docstrings(
        base_model, load_tokenizer(model_id), codes, max_new_tokens=max_new
    )

    tuned_model = load_base_with_adapter(model_id, cfg.train.adapter_dir)
    tuned_tok = load_tokenizer(cfg.train.adapter_dir)
    tuned_out = generate_docstrings(tuned_model, tuned_tok, codes, max_new_tokens=max_new)

    return [
        {"code": code, "reference": reference, "base": base, "tuned": tuned}
        for code, reference, base, tuned in zip(codes, references, base_out, tuned_out, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned docstrings.")
    parser.add_argument("--config", default=None, help="Path to a TOML config override.")
    parser.add_argument(
        "--judge", default=None, help="Judge backend override (auto/claude_cli/openai/mock)."
    )
    parser.add_argument("--generations", default=None, help="Path to a generations JSONL.")
    parser.add_argument("--generate", action="store_true", help="Generate from the models first.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N examples.")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel judge calls (1 = serial)."
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    backend = args.judge or cfg.eval.judge_backend
    judge = make_judge(backend, cfg.eval.judge_model)

    gen_path = Path(args.generations or cfg.eval.generations_path)
    if args.generate or not gen_path.exists():
        rows = generate_rows(cfg, args.limit)
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with gen_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        rows = load_generations(gen_path)
        if args.limit is not None:
            rows = rows[: args.limit]

    base_scores = score_variant(judge, rows, "base", args.workers)
    tuned_scores = score_variant(judge, rows, "tuned", args.workers)
    base_summary = aggregate(base_scores)
    tuned_summary = aggregate(tuned_scores)

    meta = {"judge_backend": resolve_backend(backend), "base_model": cfg.model.base_model_id}
    report = build_report(base_summary, tuned_summary, meta)

    print(format_table(base_summary, tuned_summary))
    report_path = Path(cfg.eval.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJudge: {meta['judge_backend']}  |  wrote {report_path}")


if __name__ == "__main__":
    main()
