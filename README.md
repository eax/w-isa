# W-ISA

Reversible, gradient-free weight editing for local language models on CPU, Apple Silicon and CUDA.

**v0.1.0.** W-ISA provides a JSON instruction format, a compiler and a Python
runtime for named weight operations, request-scoped patches and snapshot restoration.
The repository includes executable experiments and their recorded measurements.

Compile a small JSON program, bind it to PyTorch parameter names, and change live
weights with deterministic linear algebra. The runtime freezes the model and
performs edits under `torch.no_grad()`.

The default loader targets `Qwen/Qwen2.5-0.5B-Instruct`. It also accepts a local
checkpoint path or another Transformers causal LM, including
`HuggingFaceTB/SmolLM-135M-Instruct`. Parameter names and dimensions come from the
loaded model.

## Run it

From a checkout, create a virtual environment and install the package:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/wisa demo
```

Installation downloads dependencies. The self-contained demo changes a fixed
matrix's output from `[5, 11]` to approximately
`[3, -1]`, then restores `[5, 11]` exactly.

Python 3.10+
is declared; the tested interpreter is Python 3.12.9. Direct runtime dependencies
are `torch`, `transformers`, and `numpy`, plus their normal transitive packages.

Inspect a cached model with
`.venv/bin/wisa inspect --local-files-only --contains mlp.down_proj`.
Use `--model /path/to/checkpoint` for a local checkpoint. Omitting
`--local-files-only` allows Transformers to download public model files from
Hugging Face; generation runs locally.

For a before/edit/after/undo comparison on cached Qwen weights:

`.venv/bin/wisa run --local-files-only --program examples/prune.json --prompt "What is a matrix?" --max-new-tokens 32 --undo`

Run `.venv/bin/python examples/roundtrip.py` for the pretrained SmolLM experiment.
The first run downloads the model when needed; subsequent runs can use
`--local-files-only`. All model-running commands accept `--device`.

The pruning example zeros small values in Qwen's first MLP down-projection.
Inspect parameter names before using the example with a different architecture.
CLI edits operate on in-memory weights for the current process.
For an application that keeps a model resident, use `WeightRuntime` directly.

`--device auto` selects CUDA, then MPS, then CPU according to PyTorch availability.
Use `--device cuda` for the current CUDA device or `--device cuda:1` for an indexed
device. `--device mps` and `--device cpu` select those backends explicitly.
The loader checks availability and CUDA indices before loading model files.
Weights default to FP32; `--dtype float16` and `--dtype bfloat16` select the other
supported model dtypes.

For NVIDIA GPUs, install the CUDA-enabled PyTorch build for your platform using
the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then
install W-ISA. Run `.venv/bin/wisa inspect --device cuda --contains mlp.down_proj`
to inspect a model, or
`.venv/bin/python examples/roundtrip.py --device cuda --output artifacts/cuda-roundtrip.json`
for the head-ablation and projection experiment. Its report records the selected
device, arithmetic dtype, weight changes, undo and replay results.

## Instructions

Source contains `"version": 1` and an `"instructions"` array. Each instruction
has `op` and `target`; the target is a full name from `model.named_parameters()`.
Operand fields refer to named arrays in an NPZ file supplied by `--operands`, or
to a mapping of numeric arrays/tensors passed to `compile_program()`.

For a matrix $W\in\mathbb{R}^{m\times n}$:

| Opcode | Extra fields | Operation |
| --- | --- | --- |
| `ADD` | `delta`, optional `alpha=1` | $W\leftarrow W+\alpha D$; exactly matching shape |
| `SCALE` | `factor` | $W\leftarrow fW$ |
| `RANK1` | `left`, `right`, optional `alpha=1` | $W\leftarrow W+\alpha uv^T$, with $u\in\mathbb{R}^m$, $v\in\mathbb{R}^n$ |
| `PROJECT_OUT` | `direction` | $W\leftarrow W-(W\hat v)\hat v^T$, with $\hat v=v/\lVert v\rVert_2$ |
| `WRITE_KEY` | `key`, `value` | $W\leftarrow W+(y-Wk)k^T/(k^Tk)$ |
| `PRUNE` | `threshold` | Set entries with $|W_{ij}|<t$ to zero; equality is retained |

`ADD`, `SCALE`, and `PRUNE` also work on vectors and other dense floating-point
parameter shapes. The rank-one and projection instructions require matrices
because their operands describe matrix input/output spaces. Thresholds are
nonnegative. The compiler checks parameter names, operand shapes, finite values,
instruction fields and nonzero projection directions before writing weights.
NPZ loading uses `allow_pickle=False`.

Every instruction also accepts an optional `slice` array, one entry per parameter
dimension. `null` selects that entire axis; `[start, stop]` selects a nonempty,
zero-based, half-open range within the axis. For example, `"slice": [null, [0, 64]]`
selects the first 64 columns of a matrix while preserving the other columns.
Operand shapes are checked against the selected region. With no `slice`, the
instruction retains its original whole-parameter behavior. Reports count changed
entries and the delta norm in the selected region; entries outside it have zero
delta. Backups still cover the whole target parameter, and overlapping slices
execute in source order.

`PROJECT_OUT` removes an input direction, so $W'v\approx0$.
`WRITE_KEY` gives $W'k\approx y$ and preserves the response to vectors orthogonal
to $k$. It is the minimum-Frobenius-norm single-key correction in exact
arithmetic. For a biased linear layer, pass the desired pre-bias output as `value`.

For the demo, $W=[[1,2],[3,4]]$, $k=[1,2]$, and $y=[3,-1]$ give
$W'=[[0.6,1.2],[0.6,-0.8]]$. The output changes from $[5,11]$ to approximately
$[3,-1]$; undo restores $[5,11]$ exactly.

## Embedding the runtime

Import `compile_program` and `WeightRuntime` from `wisa`. Construct a runtime
with your model, compile a source mapping against that model and an operand
mapping, then pass the returned `Program` to `runtime.execute(program)`.
`runtime.undo()` restores the last successful nonempty program.
`runtime.discard_undo()` releases its saved tensors without changing weights.

The [compiler](src/wisa/compiler.py) produces a tuple of bound instructions and
detached CPU FP64 operand copies. A compiled program owns its operand copies.
The compiled IR contains PyTorch tensors.
The [runtime](src/wisa/runtime.py) rechecks target names and shapes before writing.
The [model adapter](src/wisa/model.py) handles local loading, chat templates, and
greedy generation.

Execution preserves `Parameter` identity, including ordinary tied embedding/LM
head weights. Instructions run in source order; each result is rounded to the
parameter's original dtype before the next instruction reads it. Reports contain
the norm and number of actually changed entries after that rounding.

All edited targets are backed up before the first write. An execution exception
or cancellation restores those backups and keeps the preceding undo point.
Successful execution replaces the undo point; an empty program preserves it.
Undo restores the saved parameter values. Before restoration, the runtime checks
parameter identity, shape and dtype against the snapshot.

`runtime.temporary(program)` adds an owned request scope: it holds the model's
reentrant lock, applies the program, and restores the entry state in `finally`.
Nested scopes and additional runtime `execute()`/`undo()` calls are tracked;
the preceding public undo point survives the scope. Runtime instances wrapping
the same model object share this lock and snapshot state. This includes no-match
requests in the controller. Coordination is keyed by the model object and covers
operations routed through the shared runtime lock.

Use `runtime.lock` around other inference calls sharing that model. After a
weight change, start generation with a fresh KV cache, as the CLI and controller
do. Scope cleanup runs through Python's `finally` mechanism.

## Numerical behavior and cost

The model stays on its CPU, MPS or CUDA device while edits are calculated on CPU
in FP64. Host transfer and dtype conversion are separate operations. Results are
checked for finite values, rounded to the original weight dtype and copied back
into the existing parameter view. The runtime synchronizes each touched
accelerator before snapshots and after writes; restoration also waits for pending
device work. CUDA synchronization uses the actual device index of each parameter.

The arithmetic is deterministic for fixed inputs and execution settings. Reports
measure the changes after rounding to the model's dtype. The recorded CPU/MPS
tests compare weights bit for bit; generation uses greedy decoding.

Every matrix instruction takes $O(mn)$ arithmetic for the selected region,
including the rank-one ones: this implementation materializes that region's
updated dense matrix. Projection uses vector products and an outer product.
For instructions selecting $N_j$ elements each, execution costs
$O(P+B+\sum_j N_j+A)$, where $P$ is the number of named parameters inspected,
$B$ is the total element count of full target parameters backed up once, and
$A$ counts operand elements checked. Host/device transfers include both full target
backups and per-instruction selected regions.

Beyond model storage, peak memory includes the previous undo snapshot, the new
transaction backups, compiled FP64 operands, and a constant number of dense
FP64 work buffers for the largest target. Backups use the original weight dtype
on CPU. Snapshots copy the targeted parameters. During normal
idle operation, one undo snapshot remains resident. FP32 Qwen 0.5B weights alone
are roughly 2 GB; inference activations, caches, and edit workspaces add to that.

## Validation

Use the VS Code task **W-ISA: offline tests**, or run
`PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v`.
The tests use `unittest`, fixed matrices, and a tiny randomly initialized
`Qwen2ForCausalLM`. For debugging, place a
breakpoint in the runtime and launch the `wisa` module with the `demo` argument
using the workspace virtual environment.

The November 3, 2026 run used ARM64 macOS, Python 3.12.9, PyTorch 2.15.0,
Transformers 5.22.1 and NumPy 2.5.4: **62 passed; 6 CUDA hardware cases skipped
by device discovery**. Coverage includes independent NumPy/hand-derived
answers, extreme direction scales, rounding, rollback after partial execution,
tied parameters, repeated execution, MPS operands, all opcodes on CPU/MPS, and
Qwen logit changes with exact undo/replay. Seven additional slice tests cover
head-only ablation, noncontiguous MPS writes, operand shape checks, overlapping
slices, rollback, projection, and invalid bounds. New evaluation/profile/control
tests cover token accounting, paired activation means, threshold calibration,
nested request scopes, cancellation during writes, multiple runtime instances,
rule ordering, operand snapshots, and streaming close. Dependency checks and the installed
CLI demo passed. The loader uses the `torch_dtype` argument shared with the
declared Transformers 4.x range.

The tiny Qwen checkpoint loader test uses real locally saved weights and a mocked
tokenizer loader. The separate experiment below exercised real pretrained
SmolLM weights, its tokenizer
and chat template, and MPS generation. The weights were downloaded from Hugging
Face; all inference was local.

CUDA tests cover opcode parity, FP32/FP16/BF16 rounding, sliced writes, rollback,
request cancellation, stream completion, indexed devices and tiny-Qwen inference.
Hardware cases are enabled by `torch.cuda.is_available()` and device count.
Device selection and synchronization routing also have mocked backend tests.

## Pretrained SmolLM roundtrip

Run `.venv/bin/python examples/roundtrip.py --local-files-only` to reproduce the
experiment using the cached model. The [runner](examples/roundtrip.py) accepts
`--model`, `--device`, `--layer-index`, `--head-index`, repeatable `--prompt`, and
`--max-new-tokens` options. Defaults use the tenth transformer block
(`layers[9]`), query head 0, and four fixed prompts with greedy decoding.

`SCALE` with factor 0 zeros only `o_proj.weight[:, 0:64]`. With nine query heads
of dimension 64, these columns carry head 0's contribution to the residual
stream. Other heads' columns, and the shared GQA key/value projections, remain
unchanged. This is output ablation: attention computation runs and the selected
head's output contribution is zeroed.

`PROJECT_OUT` edits that block's MLP down-projection. Its fixed direction comes
from the measured down-projection input at the last token of the first prompt,
before either edit. The roundtrip measures the two operations together.

Two separate process runs matched exactly on saved logits, token sequences,
measured direction, and model-weight hashes. Within each process, the runner
checks baseline repeat, edited repeat, undo, and replay. All four prompts changed
logits; three changed their generated token sequences. Direction response norm
fell from 6.4653542 to $1.4276433\times10^{-7}$. Edit transactions took 9.2–14.6 ms
across the four measured executions, excluding model loading and inference.

See the [measured results](artifacts/roundtrip-report.md),
[first JSON report](artifacts/smol-roundtrip.json), and
[independent process report](artifacts/smol-roundtrip-repeat.json).
The runner also saves raw FP32 logits, the measured FP64 operand, and the exact
JSON program, together with the model revision and execution environment.

## Evaluation, profiles, and match-action

The [control-plane report](artifacts/control-plane-report.md) records full
WikiText-2 measurements, held-out contrastive-profile results, and real MPS
request-isolation tests. The combined original ablation raised perplexity by
0.3754%. A separate formal-minus-casual direction raised it by 0.0040%. The
controller gated four of twelve held-out requests: four logit distributions
changed and all twelve greedy token sequences were preserved. The report includes
completion-preference scores, trigger counts and generated text.

`perplexity(model, token_ids, context_length, stride)` computes token-weighted NLL.
It scores every token except the first exactly once and returns window-level
losses, counts, protocol settings, and elapsed time. The
[four-condition runner](examples/perplexity.py) compares original weights,
head-only ablation, MLP-only projection, and their combination on identical tokens.

`ActivationProfiler(runtime, tokenizer, ActivationSite(...))` captures an input
or output activation at a named module with last-token or mean-token pooling.
`contrast(positive, negative)` returns detached CPU FP64 class means and their
difference, plus source metadata. `profile.operands("v_diff")` feeds the compiler
directly; `profile.save(...)` stores numeric NPZ arrays and JSON provenance.
The [style example](examples/profile_style.py) uses disjoint topic sets for
profiling, threshold calibration, and evaluation. It also saves a symbolic
`PROJECT_OUT` program. Each profile records its capture location and feature space.

`Rule(name, site, direction, threshold, program)` defines one match-action row.
`MatchActionController(runtime, tokenizer, rules)` uses caller-supplied rule order:
the first rule with `cosine_score > threshold` wins. All scores are measured
before any patch; identical sites share an activation capture. Rule tensors are
copied on registration. Threshold calibration is a separate function operating
on the designated calibration set.

`controller.generate(prompt)` wraps local greedy generation. For other synchronous
inference engines, use `controller.run(prompt, callback)` or hold
`controller.session(prompt)` for the full inference lifetime. The result separates
inference time from control time, including restoration and any lock wait.
These APIs serialize requests sharing one mutable model.

`controller.stream(prompt, iterator_factory)` keeps the scope open while consuming
a synchronous iterator. Close the returned generator on early disconnect, using
the same thread, for example through `contextlib.closing`. The controller owns
cleanup of the supplied iterator. For an async server, run the entire synchronous
request in a worker rather than holding a thread-owned `RLock` across arbitrary
`await` points. The API runs inside the caller's Python application.

To reproduce the measurements, run the following from the workspace:

- `.venv/bin/python examples/fetch_wikitext.py` downloads the public test corpus
	once, using the standard library. See [data attribution](data/README.md).
- `.venv/bin/python examples/perplexity.py` evaluates the complete cached split.
	`--max-tokens` explicitly selects a shorter pilot; the report marks whether the
	full test split was evaluated.
- `.venv/bin/python examples/profile_style.py --corpus data/wikitext2-test.json`
	extracts the profile, evaluates held-out behavior, and measures its corpus cost.
- `.venv/bin/python examples/match_action.py` checks calibrated gating, exact
	undo/repeat, failure injection, stream closure, and two threaded requests.

Model-running examples use cached checkpoints; the roundtrip example also
supports initial model download. Runtime dependencies are torch, transformers
and numpy.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for test and experiment conventions and
[SECURITY.md](SECURITY.md) for the trust boundary. The GitHub workflow runs CPU
tests against an installed wheel. The linked experiment reports record local
MPS measurements and their execution environment.

Project code is licensed under the [MIT License](LICENSE), copyright 2026 Ekrem Aksoy. Downloaded model weights,
dependencies, and datasets retain their own licenses. WikiText is fetched on
demand and is excluded from Git; see [dataset attribution](data/README.md).
Reports and small numeric experiment artifacts are included for reproducibility.