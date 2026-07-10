# Build journal & learning guide

This is my working log for building `docstring-tuner`. Each phase is one commit. I check off
boxes as I go. A few phases marked **(hands-on)** are ones I implement myself from a stub +
failing tests — that's where most of the learning is; the rest is plumbing I lean on but read
through so I understand it.

**How to run a hands-on phase:** open the stub module, read the tests that describe it, and make
them pass:

```powershell
pytest tests/test_<module>.py -q
```

When the tests go green, tick the box and commit.

---

## Where the build stands

The scaffolding and the fiddly/GPU modules are all in place and committed. **Three small
modules are left for me to write** — the highest-learning bits — each with failing tests already
waiting:

| Module | What to implement | Test file | Phase |
| --- | --- | --- | --- |
| `src/docstring_tuner/prompts.py` | `SYSTEM_PROMPT`, `user_prompt`, `build_messages` | `tests/test_prompts.py` | 1 |
| `src/docstring_tuner/ast_utils.py` | `split_function` | `tests/test_ast_utils.py` | 2 |
| `src/docstring_tuner/metrics.py` | `rouge_l`, `is_google_style` | `tests/test_metrics.py` | 7 |

Right now `pytest` reports **26 passed / 24 failed** — the 24 are exactly these three modules.
Turn them green one file at a time (each stub's docstring walks through the algorithm).

**Then, in order:**

1. `python -m docstring_tuner.data` — build the dataset locally (CPU, a few minutes).
2. Open `notebooks/train_colab.ipynb` on a Colab **T4** — trains the adapter and generates base
   vs tuned docstrings; download `artifacts.zip` and unzip it into the repo.
3. `python -m docstring_tuner.demo --limit 3` — see base vs tuned side by side.
4. `python -m docstring_tuner.evaluate` — score them with the local `claude` judge; writes
   `artifacts/eval_report.json`.
5. Fill the README results table and commit a couple of small `artifacts/samples/`.

---

## Phase 0 — Scaffold & tooling ✅

Set up the skeleton so everything after it has a home.

- [x] `pyproject.toml` — `src/` layout, `[project.scripts]` (`dt-*`), and config for mypy / ruff / black
- [x] `requirements-cpu.txt` (local) and `requirements-gpu.txt` (Colab) with pinned versions
- [x] `.gitignore`, `LICENSE`, `Makefile`, `configs/default.toml`, README skeleton

**What I learned / why it's shaped this way**

- **`src/` layout** stops Python from importing the package from the working tree by accident, so
  tests exercise the *installed* package — closer to how a user would consume it.
- **Console scripts beat a Makefile on Windows** (there's no `make` here). `pip install -e .`
  exposes `dt-data`, `dt-train`, etc. identically in PowerShell and on Colab.
- **Two requirements files** because the GPU and CPU stacks genuinely differ: Colab keeps its
  preinstalled CUDA `torch` (reinstalling it breaks the build) and adds `bitsandbytes`; locally I
  use a CPU `torch` and no `bitsandbytes` (4-bit is GPU-only).

---

## Phase 1 — Config + prompts **(prompts = hands-on)** ✅

- [x] Read `src/docstring_tuner/config.py` (frozen dataclasses + a tiny `tomllib` loader)
- [x] Implement `src/docstring_tuner/prompts.py` so `pytest tests/test_prompts.py` passes
- [x] Commit

**Concepts:** how the instruction we give the model (the *prompt*) defines the task, and why a
frozen dataclass config keeps runs reproducible. The prompt is half the battle — a fine-tune only
has to cover the gap the prompt can't.

---

## Phase 2 — AST docstring-stripper **(hands-on)** ✅

- [x] Implement `split_function()` in `src/docstring_tuner/ast_utils.py` so
      `pytest tests/test_ast_utils.py` passes
- [x] Commit

**Concepts:** parse code into a tree with the stdlib `ast` module, pull the docstring out with
`ast.get_docstring`, drop that node, and turn the tree back into source with `ast.unparse`. This is
how the *input* (function without its docstring) and *target* (the docstring) are manufactured.

---

## Phase 3 — Data pipeline ✅

- [x] Build train/test splits from CodeSearchNet with `dt-data`; confirm they're disjoint
- [x] Commit

**Concepts:** streaming a slice of a big HF dataset, filtering to useful examples, and caching to
disk so a run is reproducible. Golden rule: **never let a test example leak into training.**

---

## Phase 4 — Model loading ✅

- [x] Read `src/docstring_tuner/model.py`; understand the 4-bit config and the CUDA/CPU branches
- [x] Commit

**Concepts:** what nf4 + double-quant actually configures, why a T4 needs fp16 (no bf16 hardware),
and why you never `.to(device)` a `device_map`-placed 4-bit model.

---

## Phase 5 — QLoRA training ✅

- [x] Read `src/docstring_tuner/train.py`; understand LoRA target modules and completion-only loss
- [x] Commit

**Concepts:** LoRA freezes the base and trains tiny low-rank adapters on the attention + MLP
projections; `completion_only_loss` masks the prompt so the model is graded only on the docstring
it writes.

---

## Phase 6 — Generation loop ✅

- [x] Read `src/docstring_tuner/generate.py`; understand the explicit `torch.no_grad()` loop
- [x] Commit

**Concepts:** left-padding for batched generation, moving only the input tensors to the model's
device, and slicing the prompt off the output. This is the "real PyTorch" in the project.

---

## Phase 7 — Metrics + eval harness **(metrics = hands-on)**

- [x] Implement `rouge_l()` and `is_google_style()` in `src/docstring_tuner/metrics.py` so
      `pytest tests/test_metrics.py` passes
- [x] Read `src/docstring_tuner/judge.py` and `evaluate.py`
- [x] Commit

**Concepts:** ROUGE-L is a longest-common-subsequence overlap — implementing it by hand demystifies
"lexical overlap" metrics. The judge is an LLM behind a one-method `Protocol`, which is how you keep
an evaluator swappable and vendor-neutral.

---

## Phase 8 — Demo + Colab run

- [x] Run `notebooks/train_colab.ipynb` on a T4 → download `adapter/` + `generations.jsonl`
- [x] `dt-demo` and `dt-eval` locally; fill the README results table; commit a few `artifacts/samples/`
- [x] Commit

**The result — and why the two metrics disagree**

Scored 149 held-out functions with the `claude_cli` (Haiku) judge:

| Metric | Base | Tuned | Δ |
| --- | --- | --- | --- |
| Judge score (0–10) | 5.810 | 5.664 | −0.145 |
| ROUGE-L (F1) | 0.235 | 0.328 | **+0.092** |
| Format-compliance | 0.980 | 0.973 | −0.007 |

ROUGE-L jumped ~39% relative; the judge dipped slightly. Not a contradiction — a finding. The
samples (`artifacts/samples/side_by_side.md`) show why: the **base** model pads with a verbose,
sometimes-hallucinated docstring *and re-emits the whole function*, while the **tuned** model
emits a bare docstring in the reference's terse style. That style shift is exactly what lifts
ROUGE-L. The judge, rewarding apparent completeness, doesn't see the terser tuned output as
better — it may even reward base's padding. Lesson: **ROUGE-L measures conformance to the target
wording; the judge measures intrinsic quality. They answer different questions, so report both.**

**What I learned / a real bug on the first T4 run**

Training died at step 0 with `_amp_foreach_non_finite_check_and_unscale_cuda not
implemented for 'BFloat16'`. The chase: I probed the machine instead of guessing —
`get_device_capability()` → `(7, 5)`, `supports_bf16()` → `False`, `compute_dtype()` →
`fp16`. So dtype detection was correct; the base was fp16, not bf16. Then I probed the
*trainable* params: all `fp32`.

That was the tell. A fp16 `GradScaler` exists only to keep **fp16** gradients from
underflowing. My adapter weights are fp32 — the scaler protects nothing, yet `fp16=True`
still builds it, and its unscale kernel then chokes on the T4. All cost, no benefit. Fix:
only enable mixed precision when bf16 hardware is present (`fp16=False` on Turing); the
fp32 adapter trains fine without a scaler. The base still computes in fp16 via bitsandbytes.

Lesson that generalized: **mixed-precision `fp16=True` is worth it only when your trainable
params are fp16.** With QLoRA the adapter is fp32, so a T4 wants plain fp32 optimization,
not fp16+scaler.

---

## Phase 9 — Docs & ship

- [ ] Finalize README (results, compute, scope); `mypy` + `ruff` + `black` clean
- [ ] Push to a private GitHub repo
- [ ] Commit

---

### Handy commands

```powershell
pytest                      # run tests
mypy                        # type-check
ruff check . ; black .      # lint + format
python -m docstring_tuner.data          # build data
python -m docstring_tuner.demo --limit 3
python -m docstring_tuner.evaluate --judge mock
```
