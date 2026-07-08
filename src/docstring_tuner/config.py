"""Typed, frozen configuration for docstring-tuner.

The defaults defined here are the single source of truth; ``configs/default.toml``
mirrors them for documentation. Load overrides with :meth:`Config.load`, which overlays
a (possibly partial) TOML file on top of the defaults and rejects unknown keys so typos
in a config file fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCfg:
    """Base model and generation settings."""

    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 256


@dataclass(frozen=True, slots=True)
class DataCfg:
    """Dataset selection, filtering thresholds, and split sizes."""

    dataset_id: str = "Nan-Do/code-search-net-python"
    n_train: int = 1500
    n_test: int = 200
    max_source_chars: int = 1200
    min_docstring_chars: int = 40
    seed: int = 13
    cache_dir: str = "data"


@dataclass(frozen=True, slots=True)
class TrainCfg:
    """QLoRA / SFT hyper-parameters."""

    output_dir: str = "artifacts/run"
    adapter_dir: str = "artifacts/adapter"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 768


@dataclass(frozen=True, slots=True)
class EvalCfg:
    """Evaluation and judge settings."""

    judge_backend: str = "auto"  # auto | claude_cli | openai | mock
    judge_model: str = "haiku"
    generations_path: str = "artifacts/generations.jsonl"
    report_path: str = "artifacts/eval_report.json"


@dataclass(frozen=True, slots=True)
class Config:
    """Top-level configuration aggregating every section."""

    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Return the defaults, optionally overlaid with values from a TOML file."""
        cfg = cls()
        if path is None:
            return cfg
        raw: dict[str, Any] = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        overlaid = {
            f.name: _overlay_section(getattr(cfg, f.name), raw.get(f.name), f.name)
            for f in fields(cfg)
        }
        return replace(cfg, **overlaid)


def _overlay_section(section: Any, overrides: Any, name: str) -> Any:
    """Apply a dict of overrides to one config dataclass, rejecting unknown keys."""
    if not overrides:
        return section
    if not isinstance(overrides, dict):
        raise TypeError(f"Config section [{name}] must be a table, got {type(overrides).__name__}")
    valid = {f.name for f in fields(section)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown keys in config section [{name}]: {sorted(unknown)}")
    return replace(section, **overrides)
