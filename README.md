# docstring-tuner

Fine-tune a small open LLM with **4-bit QLoRA** to turn a Python function into a
**Google-style docstring**, then measure whether the fine-tune actually helped with a
**base-vs-fine-tuned evaluation harness** (LLM-as-judge + ROUGE-L + a format check).
The whole thing is small and fast on purpose: a 1.5B base model, ~1.5k training examples,
and a few minutes of training on a free Colab T4. It is a portfolio artifact, not a SOTA
attempt.

## What's inside

- **Data** — mines function/docstring pairs from the CodeSearchNet Python subset, strips the
  docstring out of the function with the stdlib `ast` module, and keeps it as the target.
- **Training** — 4-bit QLoRA (nf4 + double quant) with LoRA adapters on the attention and MLP
  projections, trained with `trl`'s `SFTTrainer`; only the adapter is saved.
- **Inference** — an explicit `torch.no_grad()` generation loop with manual device/dtype
  handling that runs 4-bit on CUDA and falls back to fp32 on CPU.
- **Eval** — runs the base and the fine-tuned model over a held-out test set and scores each
  docstring with a swappable LLM-as-judge (rubric: faithfulness, completeness, format) plus
  ROUGE-L and a boolean Google-style format check.

## Results

_Filled in after a real training run (see `docs/LEARNINGS.md`, phase 8)._

| Metric | Base | Fine-tuned | Δ |
| --- | --- | --- | --- |
| Judge score (0–10) | _TODO_ | _TODO_ | _TODO_ |
| ROUGE-L (F1) | _TODO_ | _TODO_ | _TODO_ |
| Format-compliance rate | _TODO_ | _TODO_ | _TODO_ |

Full machine-readable results: `artifacts/eval_report.json`. Sample side-by-side outputs live
in `artifacts/samples/`.

## Quickstart

This project separates the **GPU-only** step (training) from everything else. Data building,
evaluation scoring, and the demo run on CPU.

### Local (Windows / macOS / CPU)

```powershell
# 1. Install the CPU stack + this package
pip install -r requirements-cpu.txt
pip install -e . --no-deps

# 2. Build the dataset (downloads a CodeSearchNet slice, caches to data/)
python -m docstring_tuner.data          # or: dt-data

# 3. Once you have a trained adapter (from Colab, below), see base vs tuned side-by-side
python -m docstring_tuner.demo --limit 3 # or: dt-demo --limit 3
```

> On Windows there is no `make`; use the `python -m docstring_tuner.<module>` commands (or the
> installed `dt-*` console scripts). The `Makefile` is a convenience for Unix/Colab users.

### Training on Colab (free T4)

Open `notebooks/train_colab.ipynb` in Colab, set the runtime to a T4 GPU, and run all cells.
It installs `requirements-gpu.txt`, builds the data, trains the QLoRA adapter, generates base
and tuned docstrings over the test set, and lets you download `artifacts/adapter/` and
`artifacts/generations.jsonl`. Drop those into your local `artifacts/` and run the eval.

## Evaluation & the judge

The judge sits behind a small typed `Judge` protocol, so the backend is swappable:

- **`claude_cli`** (default when the `claude` CLI is on PATH) — shells out to
  `claude -p <rubric> --model haiku`. No API key, no vendor lock-in.
- **`openai`** — any OpenAI-compatible `/chat/completions` endpoint via base URL + key from env
  (`DT_OPENAI_BASE_URL`, `DT_OPENAI_API_KEY`), so it also works against LiteLLM, Azure, or a
  local server.
- **`mock`** — a deterministic offline judge for tests and CI (no network).

```powershell
python -m docstring_tuner.evaluate       # base vs tuned over the test set -> eval_report.json
python -m docstring_tuner.evaluate --judge mock   # offline smoke test
```

## Compute

- **4-bit QLoRA needs a CUDA GPU.** A free Colab/Kaggle **T4 is enough** (training the default
  1.5B model on ~1.5k examples takes a few minutes).
- A **T4 is Turing (sm_75) and has no bf16 hardware**, so training uses fp16 there; bf16 is
  auto-enabled only on Ampere or newer.
- **Apple Silicon / MPS cannot run bitsandbytes 4-bit.**
- **CPU fallback:** inference (demo/eval generation) runs on CPU in fp32 — slow, but fine for a
  handful of examples. Training on CPU is not supported.

## Scope & limitations

- Small model + short run: this demonstrates that a targeted QLoRA fine-tune shifts a base
  model toward a consistent Google-style docstring format; it is not a production docstring
  generator.
- CodeSearchNet docstrings are freeform (not natively Google-style). We **filter** toward
  examples that already contain `Args:`/`Returns:`/`Raises:` and normalize whitespace rather
  than performing full cross-style conversion. See `src/docstring_tuner/data.py`.
- ROUGE-L is a shallow lexical signal; the LLM-judge is the primary quality metric.
- The default judge is a single small model (Haiku). Judge scores are relative signal, not
  ground truth; the swappable protocol lets you drop in a stronger judge.

## Development

```powershell
pip install -e ".[dev]"
pytest          # unit tests
mypy            # type-check (strict, pragmatic overrides for the ML libs)
ruff check .    # lint
black --check . # format check
```

The project was built in small, self-contained steps; that journal lives in
[docs/LEARNINGS.md](docs/LEARNINGS.md).

## License

MIT — see [LICENSE](LICENSE).
