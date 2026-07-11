"""Load the base model and tokenizer with explicit device/dtype handling.

Three regimes, chosen automatically:

- **CUDA + 4-bit** (default on GPU): nf4 double-quant via bitsandbytes, compute dtype fp16
  on Turing (T4) and bf16 on Ampere+ (auto-detected). This is what QLoRA training needs.
- **CUDA + fp16** (``four_bit=False``): unquantized half precision, for when you have VRAM
  to spare and want a merge-able base.
- **CPU + fp32**: the fallback for local demo/eval on a machine without a GPU. bitsandbytes
  4-bit is GPU-only, so we simply don't quantize here.

Key rule encoded below: a ``device_map`` / 4-bit model is already placed on its device — we
never call ``model.to(device)`` on it.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def has_cuda() -> bool:
    """True if a CUDA device is available."""
    return torch.cuda.is_available()


def supports_bf16() -> bool:
    """True only on Ampere (sm_80) or newer — Turing/T4 has no bf16 hardware."""
    return has_cuda() and torch.cuda.get_device_capability()[0] >= 8


def compute_dtype() -> torch.dtype:
    """The compute dtype for the current device: bf16 on Ampere+, else fp16 on GPU, else fp32."""
    if not has_cuda():
        return torch.float32
    return torch.bfloat16 if supports_bf16() else torch.float16


def make_bnb_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    """4-bit nf4 + double-quant config with the given compute dtype."""
    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]  # transformers ctor is untyped
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer configured for batched generation (left padding, pad token set)."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(model_id: str, *, four_bit: bool = True) -> PreTrainedModel:
    """Load the base causal-LM, quantized on GPU and fp32 on CPU."""
    if not has_cuda():
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    dtype = compute_dtype()
    kwargs: dict[str, Any] = {"device_map": {"": 0}, "dtype": dtype}
    if four_bit:
        kwargs["quantization_config"] = make_bnb_config(dtype)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def load_base_with_adapter(
    model_id: str, adapter_dir: str, *, four_bit: bool = True
) -> PreTrainedModel:
    """Load the base model and attach a trained LoRA adapter (kept attached, not merged).

    A 4-bit base cannot be ``merge_and_unload``-ed, so we keep the adapter attached at
    inference time.
    """
    from peft import PeftModel

    base = load_base(model_id, four_bit=four_bit)
    # PeftModel wraps but doesn't subclass PreTrainedModel; it's drop-in for our inference use.
    return cast(PreTrainedModel, PeftModel.from_pretrained(base, adapter_dir))


def model_device(model: PreTrainedModel) -> torch.device:
    """The device the model's parameters live on (works for 4-bit and CPU models)."""
    return next(model.parameters()).device
