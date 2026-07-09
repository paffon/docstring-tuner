"""QLoRA supervised fine-tuning of the base model on the docstring task.

This is the one **GPU-only** step: 4-bit QLoRA needs bitsandbytes + CUDA. Run it on a Colab
T4 via ``notebooks/train_colab.ipynb`` (or any CUDA box) with ``python -m docstring_tuner.train``.

We attach LoRA adapters to the attention and MLP projections, train with trl's ``SFTTrainer``
using the conversational prompt/completion format (loss is masked to the docstring only via
``completion_only_loss``), and save just the adapter to ``artifacts/adapter/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from .config import Config, TrainCfg
from .data import read_jsonl, to_sft_record
from .model import has_cuda, load_base, load_tokenizer, supports_bf16

# LoRA on attention (q/k/v/o) + MLP (gate/up/down) projections — correct for Qwen2.5.
LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_sft_dataset(cfg: Config) -> Dataset:
    """Load cached training examples and convert them to the SFT prompt/completion format."""
    examples = read_jsonl(Path(cfg.data.cache_dir) / "train.jsonl")
    if not examples:
        raise FileNotFoundError(
            "No training data found. Run `python -m docstring_tuner.data` first."
        )
    return Dataset.from_list([to_sft_record(example) for example in examples])


def make_lora_config(tc: TrainCfg) -> LoraConfig:
    """Build the LoRA adapter configuration."""
    return LoraConfig(
        r=tc.lora_r,
        lora_alpha=tc.lora_alpha,
        lora_dropout=tc.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def make_sft_config(tc: TrainCfg) -> SFTConfig:
    """Build the trl SFT training configuration (fp16 on T4, bf16 only on Ampere+)."""
    bf16 = supports_bf16()
    return SFTConfig(
        output_dir=tc.output_dir,
        max_length=tc.max_length,
        packing=False,
        completion_only_loss=True,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        num_train_epochs=tc.num_train_epochs,
        learning_rate=tc.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        optim="paged_adamw_8bit",
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        save_strategy="no",
    )


def train(cfg: Config) -> str:
    """Run QLoRA SFT and save the adapter. Returns the adapter directory."""
    if not has_cuda():
        raise RuntimeError(
            "QLoRA 4-bit training requires a CUDA GPU (bitsandbytes is GPU-only). "
            "Run this on a Colab/Kaggle T4 — see notebooks/train_colab.ipynb."
        )

    model = load_base(cfg.model.base_model_id, four_bit=True)
    model.config.use_cache = False  # required alongside gradient checkpointing

    tokenizer = load_tokenizer(cfg.model.base_model_id)
    tokenizer.padding_side = "right"  # right-pad for training (left-pad is only for generation)

    trainer = SFTTrainer(
        model=model,
        args=make_sft_config(cfg.train),
        train_dataset=build_sft_dataset(cfg),
        peft_config=make_lora_config(cfg.train),
        processing_class=tokenizer,
    )
    trainer.train()

    adapter_dir = cfg.train.adapter_dir
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    # trl types ``trainer.model`` as ``Module | None``; after ``train()`` it is the PEFT model.
    trainer.model.save_pretrained(adapter_dir)  # pyright: ignore[reportCallIssue, reportOptionalMemberAccess]  # adapter weights + config only
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune the docstring model.")
    parser.add_argument("--config", default=None, help="Path to a TOML config override.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    adapter_dir = train(cfg)
    print(f"Saved adapter to {adapter_dir}/")


if __name__ == "__main__":
    main()
