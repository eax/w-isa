# SmolLM head ablation and MLP projection

Real pretrained `HuggingFaceTB/SmolLM-135M-Instruct`, revision
`b5f4b394abe60fe0e4d666139e57bfce520302a9`, was loaded on Apple MPS in FP32.
Parameter count: 134,515,008. The runs used ARM64 macOS 15.6.1, Python 3.12.9,
PyTorch 2.15.0, Transformers 5.22.1, and NumPy 2.5.4. Full environment and UTC
timestamps are retained in the JSON reports. The session's local date was
November 2, 2026; report timestamps fall on November 3 UTC.

## What changed

The tenth transformer block is `model.layers[9]`. It has nine query heads, three
KV heads, and 64 dimensions per head.

`SCALE`, factor `0.0`, targeted only columns `[0, 64)` of the 576×576 attention
output projection. That zeros query head 0's contribution after head
concatenation. Exactly **36,864** weight elements changed. Other head columns
were checked directly and remained equal; the rest of the model's parameter
hashes changed only for the two intended targets. Shared key/value projections
and Parameter identities were preserved.

`PROJECT_OUT` targeted the 576×1536 MLP down-projection in the same block. The
direction was captured from its input at the final token of the first prompt,
before editing. The captured activation vector was held fixed for all phases
and saved as an FP64 operand. Exactly **884,666** MLP weight entries changed
after FP32 rounding, out of 884,736 entries.

Both edits were applied together; the output measurements refer to that combined program.

## Actual device path

The evaluator was instrumented around the original implementation. For both
instructions, in both initial execution and replay, it observed:

| Boundary | Observed value |
| --- | --- |
| Live parameter device and dtype | `mps:0`, `torch.float32` |
| Arithmetic input device and dtype | `cpu`, `torch.float64` |
| Arithmetic result device and dtype | `cpu`, `torch.float64` |
| SCALE workspace | 576×64 |
| PROJECT_OUT workspace | 576×1536 |

The results were rounded back to FP32, copied into the existing MPS parameter
views, and synchronized before inference. The full model stayed on MPS.
Execution used `torch.no_grad()` and frozen parameters.

For the normalized captured direction $\hat v$, the projection measurements were:

| Measurement | Value |
| --- | ---: |
| $\lVert W\hat v\rVert_2$ before | 6.465354205884534 |
| $\lVert W'\hat v\rVert_2$ after FP32 storage | 1.427643281552695e-7 |
| Relative residual | 2.2081439563718027e-8 |
| Maximum difference from independent NumPy reference | 0.0 |

The independent NumPy reference used $W-(Wv)v^T/(v^Tv)$, then converted to FP32.

## Outputs and repeatability

Four prompts were fixed before the first run. Greedy decoding generated up to
48 new tokens with a fresh KV cache on each call. Logits below refer to the
49,152-way next-token distribution at the end of each prompt.

| Prompt | Maximum absolute logit change | Generated token sequence changed? |
| --- | ---: | --- |
| Explain in one sentence why the sky is blue. | 2.5427284240722656 | Yes |
| What is 17 + 25? Answer with just the number. | 1.9658713340759277 | Yes |
| Write a short definition of a matrix. | 2.2467260360717773 | Yes |
| Continue the sequence: 2, 4, 8, 16, | 1.7994508743286133 | No |

All 49,152 recorded logits changed for each prompt. The fourth prompt retained
its original greedy token sequence.

For the matrix prompt, the first generated sentence changed from:

> A matrix is a rectangular array of numbers, symbols, or other things, where each element is a combination of numbers, symbols, or other things.

to:

> A matrix is a rectangular array of numbers, symbols, or other data, where each element is a combination of numbers, symbols, or other data.

These excerpts come from the generated model output. Complete text and token IDs
for every phase and prompt are in the reports.

Each process checked baseline → baseline repeat → edited → edited repeat →
restored → replayed, then undid the final edit. The repeated states matched
exactly in logits and token IDs. Target parameter tensors were compared directly
on undo/replay in the second run; full-model parameter digests matched too.

The entire experiment was rerun in a fresh Python process. A separate NumPy check
read both saved logit archives, compared raw bytes for all six 4×49,152 arrays,
verified each row's SHA-256 against the JSON report, and independently recalculated
the logit changes above. Both processes matched exactly in raw logits, token
sequences, measured direction, edit reports, and model-weight digests. Timings
and timestamps differed, as expected.

The logit and token repeat comparisons used exact equality in the recorded
model and execution environment.

## Timing and storage

| Process | First edit transaction | Undo then replay transaction |
| --- | ---: | ---: |
| First | 14.571 ms | 9.219 ms |
| Independent repeat | 13.601 ms | 10.544 ms |

Times include both commands, CPU backups, arithmetic, delta reporting, FP32
conversion, copying back to MPS, and synchronization. They exclude model loading,
generation, and the independent verification/hashing passes. The table contains
the four measured edit transactions.

One full-target undo snapshot occupies 4,866,048 bytes, or about 4.64 MiB, despite
the SCALE instruction selecting only one head. The largest FP64 work matrix is
884,736 elements, or 6.75 MiB; execution holds several such buffers. The captured
FP64 operand occupies 12 KiB. Each instruction's arithmetic is linear in its
selected element count, with additional full-target backup cost. Projection
was computed using vector products and an outer product.

## Reproduce and inspect

Run `.venv/bin/python examples/roundtrip.py --device mps --local-files-only` from the workspace.
For a new process with separate evidence, add
`--output artifacts/smol-roundtrip-repeat.json`. The model is now cached. Omit
`--local-files-only` on another machine to permit the initial download; inference
still runs locally. The recorded results above use MPS. The runner also accepts
`--device cpu`, `--device cuda`, and indexed CUDA devices for additional runs.

The [runner](../examples/roundtrip.py) and [slice tests](../tests/test_slices.py)
are separate from the model cache. **34 offline tests passed**, including real
MPS slice writes and a small Qwen architecture test.

Evidence: [first report](smol-roundtrip.json),
[fresh-process report](smol-roundtrip-repeat.json),
[exact program](smol-roundtrip.program.json),
[measured direction](smol-roundtrip.operands.npz),
[raw logits](smol-roundtrip.logits.npz), and
[fresh-process raw logits](smol-roundtrip-repeat.logits.npz).