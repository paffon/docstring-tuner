"""End-to-end demo: base vs fine-tuned docstrings on a few held-out functions.

For each example it prints the function, the base-model docstring, the fine-tuned-model
docstring, and the judge's score + rationale for each — side by side.

If ``artifacts/generations.jsonl`` exists it is reused (fast, CPU-only); otherwise both models
are run to generate fresh outputs (``--generate`` forces this).

Run: ``python -m docstring_tuner.demo --limit 3``  (add ``--judge mock`` for an offline run).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import Config
from .evaluate import generate_rows, load_generations
from .judge import JudgeScore, make_judge

RULE = "=" * 80
THIN = "-" * 80


def _print_example(index: int, row: dict[str, Any], base: JudgeScore, tuned: JudgeScore) -> None:
    print(RULE)
    print(f"EXAMPLE {index}")
    print(THIN)
    print("FUNCTION (docstring removed):")
    print(row["code"])
    print(THIN)
    print(f"BASE docstring        [judge {base.overall:.1f}/10]:")
    print(row["base"])
    print(f"  rationale: {base.rationale}")
    print(THIN)
    print(f"FINE-TUNED docstring  [judge {tuned.overall:.1f}/10]:")
    print(row["tuned"])
    print(f"  rationale: {tuned.rationale}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo base vs fine-tuned docstrings.")
    parser.add_argument("--config", default=None, help="Path to a TOML config override.")
    parser.add_argument("--limit", type=int, default=3, help="How many examples to show.")
    parser.add_argument("--judge", default=None, help="Judge backend override.")
    parser.add_argument("--generations", default=None, help="Path to a generations JSONL.")
    parser.add_argument("--generate", action="store_true", help="Generate fresh from the models.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    judge = make_judge(args.judge or cfg.eval.judge_backend, cfg.eval.judge_model)

    gen_path = Path(args.generations or cfg.eval.generations_path)
    if args.generate or not gen_path.exists():
        rows = generate_rows(cfg, args.limit)
    else:
        rows = load_generations(gen_path)[: args.limit]

    for index, row in enumerate(rows, start=1):
        base_score = judge.score(row["code"], row["reference"], row["base"])
        tuned_score = judge.score(row["code"], row["reference"], row["tuned"])
        _print_example(index, row, base_score, tuned_score)
    print(RULE)


if __name__ == "__main__":
    main()
