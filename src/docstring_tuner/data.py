"""Build reproducible train/test splits of (function-without-docstring -> docstring).

Source: the CodeSearchNet Python subset, via the Parquet mirror
``Nan-Do/code-search-net-python`` (the original ``code_search_net`` script loader no longer
works on ``datasets`` >= 4). We stream a slice, use :func:`ast_utils.split_function` to pull
the docstring out of each function, keep only reasonably structured Google-style docstrings,
de-duplicate, and cache disjoint ``train.jsonl`` / ``test.jsonl`` files under ``data/``.

Run: ``python -m docstring_tuner.data``  (or ``dt-data``).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

from .ast_utils import split_function
from .config import Config, DataCfg
from .prompts import Message, build_messages

# Docstrings we keep must contain at least one of these section headers, biasing the
# training target toward the Google style we want the model to learn.
GOOGLE_MARKERS: tuple[str, ...] = ("Args:", "Returns:", "Yields:", "Raises:")

# Candidate columns that hold the full function source (with the docstring), in priority
# order. The mirror uses "code"; the fallbacks cover other CodeSearchNet exports.
SOURCE_COLUMNS: tuple[str, ...] = (
    "code",
    "whole_func_string",
    "func_code_string",
    "original_string",
)


@dataclass(frozen=True, slots=True)
class Example:
    """One training/eval example."""

    code: str  # function source with the docstring removed (the model input)
    docstring: str  # the reference Google-style docstring (the target)


def normalize_docstring(raw: str) -> str:
    """Clean a raw extracted docstring: dedent continuation lines and trim whitespace."""
    return inspect.cleandoc(raw).strip()


def passes_filters(code: str, doc: str, cfg: DataCfg) -> bool:
    """Return True if this (code, docstring) pair should be kept."""
    if len(doc) < cfg.min_docstring_chars:
        return False
    if len(code) > cfg.max_source_chars:
        return False
    return any(marker in doc for marker in GOOGLE_MARKERS)


def to_sft_record(example: Example) -> dict[str, list[Message]]:
    """Convert an example to the trl prompt/completion (conversational) SFT format."""
    return {
        "prompt": build_messages(example.code),
        "completion": [{"role": "assistant", "content": example.docstring}],
    }


def write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    """Write examples as one JSON object per line."""
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    """Read examples written by :func:`write_jsonl`."""
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(Example(code=record["code"], docstring=record["docstring"]))
    return examples


def _pick_source_column(row: dict[str, Any]) -> str:
    for column in SOURCE_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return column
    raise KeyError(f"None of {SOURCE_COLUMNS} found in dataset row; columns present: {sorted(row)}")


def _load_stream(cfg: DataCfg) -> Iterable[Any]:
    """Return a shuffled streaming view of the dataset (heavy import deferred)."""
    from datasets import load_dataset

    stream = load_dataset(cfg.dataset_id, split="train", streaming=True)
    return stream.shuffle(seed=cfg.seed, buffer_size=10_000)


def iter_examples(cfg: DataCfg) -> Iterator[Example]:
    """Yield filtered, normalized examples from the streaming dataset."""
    source_column: str | None = None
    stream = _load_stream(cfg)
    for row in stream:
        if source_column is None:
            source_column = _pick_source_column(row)
        source = row.get(source_column)
        if not isinstance(source, str):
            continue
        split = split_function(source)
        if split is None:
            continue
        code, raw_doc = split
        doc = normalize_docstring(raw_doc)
        if passes_filters(code, doc, cfg):
            yield Example(code=code, docstring=doc)


def build_dataset(cfg: Config) -> tuple[list[Example], list[Example]]:
    """Collect unique examples and split them into disjoint train/test lists."""
    target = cfg.data.n_train + cfg.data.n_test
    seen: set[str] = set()
    examples: list[Example] = []
    for example in iter_examples(cfg.data):
        key = hashlib.sha1(example.code.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
        if len(examples) >= target:
            break

    Random(cfg.data.seed).shuffle(examples)
    test = examples[: cfg.data.n_test]
    train = examples[cfg.data.n_test :]
    return train, test


def assert_disjoint(train: list[Example], test: list[Example]) -> None:
    """Fail loudly if any test code leaked into the training set."""
    train_codes = {example.code for example in train}
    overlap = train_codes & {example.code for example in test}
    if overlap:
        raise AssertionError(f"{len(overlap)} example(s) appear in both train and test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/test docstring datasets.")
    parser.add_argument("--config", default=None, help="Path to a TOML config override.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    train, test = build_dataset(cfg)
    assert_disjoint(train, test)

    cache = Path(cfg.data.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    write_jsonl(cache / "train.jsonl", train)
    write_jsonl(cache / "test.jsonl", test)

    print(f"Wrote {len(train)} train / {len(test)} test examples to {cache}/")
    if len(train) + len(test) < cfg.data.n_train + cfg.data.n_test:
        print("  (fewer than requested — the dataset stream was exhausted by the filters)")


if __name__ == "__main__":
    main()
