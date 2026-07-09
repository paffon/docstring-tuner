"""Generate docstrings with an explicit ``torch.no_grad()`` batched decoding loop.

This module holds the project's deliberate, hand-written PyTorch: a manual generation loop
with explicit device handling that runs identically whether the model is 4-bit on CUDA or
fp32 on CPU. The important details:

- **left padding** (set on the tokenizer in :mod:`docstring_tuner.model`) so that batched,
  variable-length prompts stay right-aligned and the generated continuation is easy to slice off;
- move **only the input tensors** to the model's device (never ``model.to(...)`` a 4-bit /
  ``device_map`` model);
- slice ``output[:, prompt_len:]`` to drop the prompt and keep only newly generated tokens.

``clean_docstring`` post-processes raw model text (stripping code fences / surrounding triple
quotes) and is deliberately dependency-free so it can be unit-tested without a model.

Run: ``python -m docstring_tuner.generate --file some_function.py`` (add ``--base`` for the
base model without the adapter).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .config import Config
from .prompts import build_messages

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def clean_docstring(text: str) -> str:
    """Strip markdown code fences and surrounding triple quotes from raw model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening ``` (possibly ```python)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    for quote in ('"""', "'''"):
        if len(text) >= 6 and text.startswith(quote) and text.endswith(quote):
            text = text[3:-3].strip()
            break
    return text


def _render_prompts(tokenizer: PreTrainedTokenizerBase, codes: list[str]) -> list[str]:
    """Apply the model's chat template to each function's messages."""
    # ``apply_chat_template`` is typed as a broad union; ``tokenize=False`` always yields str.
    return [
        cast(
            str,
            tokenizer.apply_chat_template(
                build_messages(code), tokenize=False, add_generation_prompt=True
            ),
        )
        for code in codes
    ]


def generate_docstrings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    codes: list[str],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[str]:
    """Greedily generate one docstring per function via an explicit no-grad loop."""
    import torch

    from .model import model_device

    device = model_device(model)
    model.eval()
    outputs: list[str] = []
    with torch.no_grad():
        for start in range(0, len(codes), batch_size):
            batch = codes[start : start + batch_size]
            prompts = _render_prompts(tokenizer, batch)
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            # Move only the integer input tensors to the model's device.
            encoded = {key: value.to(device) for key, value in encoded.items()}
            # torch's ``nn.Module.__getattr__`` stub types dynamic attrs as ``Tensor | Module``,
            # so pyright misreads ``.generate`` as a non-callable Tensor.
            generated = model.generate(  # pyright: ignore[reportCallIssue]
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = encoded["input_ids"].shape[1]
            new_tokens = generated[:, prompt_len:]  # drop the prompt (safe with left padding)
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(clean_docstring(text) for text in decoded)
    return outputs


def generate_one(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    code: str,
    *,
    max_new_tokens: int = 256,
) -> str:
    """Convenience wrapper: generate a docstring for a single function."""
    return generate_docstrings(
        model, tokenizer, [code], max_new_tokens=max_new_tokens, batch_size=1
    )[0]


def _prepare_code(raw: str) -> str:
    """Strip an existing docstring so inference sees the same input shape as training."""
    from .ast_utils import split_function

    try:
        split = split_function(raw)
    except NotImplementedError:
        return raw.strip()
    return split[0] if split else raw.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a docstring for a Python function.")
    parser.add_argument("--config", default=None, help="Path to a TOML config override.")
    parser.add_argument("--file", default=None, help="Function source file (defaults to stdin).")
    parser.add_argument("--base", action="store_true", help="Use the base model (no adapter).")
    parser.add_argument("--adapter", default=None, help="Adapter dir (default: config value).")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    code = _prepare_code(raw)

    from .model import load_base, load_base_with_adapter, load_tokenizer

    if args.base:
        model = load_base(cfg.model.base_model_id)
        tokenizer = load_tokenizer(cfg.model.base_model_id)
    else:
        adapter_dir = args.adapter or cfg.train.adapter_dir
        model = load_base_with_adapter(cfg.model.base_model_id, adapter_dir)
        tokenizer = load_tokenizer(adapter_dir)

    max_new_tokens = args.max_new_tokens or cfg.model.max_new_tokens
    print(generate_one(model, tokenizer, code, max_new_tokens=max_new_tokens))


if __name__ == "__main__":
    main()
