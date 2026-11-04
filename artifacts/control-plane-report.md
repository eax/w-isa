# Control-plane measurements

All inference below used the locally cached SmolLM-135M-Instruct revision
`b5f4b394abe60fe0e4d666139e57bfce520302a9`, FP32 weights on MPS, and CPU FP64
weight algebra. Environment: Python 3.12.9, PyTorch 2.15.0, Transformers 5.22.1,
NumPy 2.5.4, ARM64 macOS. The session date was November 2, 2026 locally.

## Full WikiText-2: original intervention

The complete `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split was used:
4,358 rows, joined with two newlines exactly as stored. SmolLM's tokenizer
produced 304,986 tokens; **304,985 targets were scored exactly once**. Tokenization
used the complete raw text with `add_special_tokens=False`.

Rolling context length was 512 input tokens, with stride 256. Each of the
1,191 windows emitted its loss sum and token count. FP32 cross-entropy losses
were summed in CPU FP64; perplexity is `exp(total_nll / scored_tokens)`.
The model processed windows of up to 512 input tokens.

| Condition | Perplexity | Change from original |
| --- | ---: | ---: |
| Original | 22.23531898 | — |
| Tenth-block query head 0 ablated | 22.31904706 | +0.37655% |
| Original single-activation MLP direction projected out | 22.23131240 | -0.01802% |
| Both edits | 22.31878257 | +0.37536% |

The head selection and activation vector are unchanged from the
[earlier roundtrip](roundtrip-report.md). Each condition began with original
weights and was restored exactly. Each complete pass took about 61 seconds.
The [raw report](perplexity.json) retains all window losses. An independent
`math.fsum` reconstruction verified token counts and aggregate perplexity.

Each condition uses the same model, tokenizer, corpus and window settings.
The initial 4,096-token
[pilot](perplexity-pilot.json) and its [scoped repeat](perplexity-pilot-scoped.json)
matched exactly in losses and counts after the transaction refactor.

## Formal-minus-casual activation profile

The [fixture](../examples/style_pairs.json) supplies matched content with formal
and casual instruction prefixes. Five topic pairs define the profile; five
different pairs calibrate the trigger threshold; six further topic pairs are
held out. The topic sets are disjoint. This handwritten fixture measures topic
transfer with a shared pair of style-instruction prefixes.

The profiler captured the last prompt token's input to
`model.layers.9.mlp.down_proj`, a 1,536-dimensional space. It calculated
$v=\mu_{formal}-\mu_{casual}$ directly in CPU FP64 under `torch.no_grad()`.
Its norm was 0.2804317523. The
[metadata](style-profile.json), [operand bank](style-profile.npz), and
[symbolic program](style-profile.program.json) are saved.

The cosine trigger uses the raw captured activation and this direction. The
strict rule is $\cos(x,v)>\tau$. Calibration maximized balanced accuracy, breaking
ties toward fewer false positives and then the larger threshold. Threshold
selection used the calibration split.

| Trigger measurement | Calibration | Held-out topics |
| --- | ---: | ---: |
| Threshold | -0.018948859620056394 | Same fixed threshold |
| True positives | 5 / 5 | 4 / 6 |
| False positives | 0 / 5 | 0 / 6 |
| False negatives | 0 | 2 |
| Balanced accuracy | 1.0 | 0.83333 |

### Completion measurements

Under an unconditional `PROJECT_OUT` of this direction,
teacher-forced scores were computed for a fixed formal and casual completion
for each held-out prompt. The diagnostic margin is mean-token log probability
of the formal completion minus that of the casual completion. It is length
normalized and evaluated against the fixture's paired completions.

| Diagnostic | Original | Projected |
| --- | ---: | ---: |
| Mean formal-preference margin | 0.34645676 | 0.34226227 |
| Formal-instruction margin minus casual-instruction margin | 0.21085805 | 0.22222082 |
| Full WikiText-2 perplexity | 22.23531898 | 22.23619810 |

The corpus perplexity change was **+0.0039537%**. The instruction-sensitivity
margin changed from 0.21085805 to 0.22222082.
Three of twelve greedy generations changed, all on casual-instruction prompts:
rescheduling an interview, acknowledging a bug report, and congratulating a
promotion. All original and modified text is preserved in the
[profile report](style-profile.report.json).

The projector is sign-invariant: `PROJECT_OUT(v)` and `PROJECT_OUT(-v)`
produce the same matrix update.
The [fresh-process repeat](style-profile-repeat.report.json) matched the direction,
calibration, held-out scores, completion scores and generated text exactly.
The full-corpus figures above come from the original profile experiment.

## Conditional requests

The calibrated rule was installed through `MatchActionController`. All scores
were measured on unpatched request-entry weights, before any selected program
was applied. Generation started with a fresh cache. Four requests matched;
eight used the original weights. Matching is based on the measured activation.

For the four matched requests, maximum absolute next-token logit changes were
0.265463, 0.307559, 0.313648 and 0.232894. All twelve conditional requests retained
their original greedy token sequences. The three unconditional changes
above were on casual prompts that the rule correctly left unpatched.

Unmatched requests retained exact original logits and token IDs. Every request
was repeated and then compared with restored inference. Two independently
constructed controllers wrapping the same model were also exercised in separate
threads, one matched and one unmatched. They produced the expected isolated
outputs, with access serialized by the shared model lock.

Injected generation errors and `asyncio.CancelledError` restored weights and
propagated the exception. Closing a synthetic two-item stream early restored
the model too. That two-item stream probe exercised lifecycle cleanup.
The full model's before/after hashes matched.
See [first request report](match-action.json) and
[fresh-process request report](match-action-repeat.json).

Measured in the latter run, averaged over the twelve first requests:

| Timing component | Mean |
| --- | ---: |
| Activation profiling | 13.288 ms |
| Apply phase, matched requests only | 10.635 ms |
| Total control overhead across all requests | 17.493 ms |
| Entire request, including inference callback | 442.535 ms |

Control overhead is measured as total request time minus callback time; it
includes profiling, applying/restoring the patch, and any lock wait. The callback
computes a diagnostic next-token forward and generates up to 48 tokens. The table
reports the averages of twelve first requests in the recorded run.

## Runtime changes and validation

`WeightRuntime.temporary()` owns snapshots for its lifetime and holds a shared
per-model lock. Nested scopes restore their own entry state, including additional
runtime writes made inside the scope. A preexisting manual undo point is preserved.
Multiple runtime objects wrapping the same `nn.Module` share scope/undo state.
An interruption halfway through `execute()` now restores completed writes even
for `KeyboardInterrupt` or `CancelledError`; the interruption is then reraised.

Controller rules are evaluated in caller order with first-match semantics.
Direction and program operand tensors are copied when rules are registered.
Matched and unmatched requests use the same isolation mechanism. Synchronous
stream consumers must close their generator on early exit, on the owning thread.
The control plane coordinates operations routed through the shared model lock
inside the calling Python application.

**54 offline tests pass**, including MPS request cancellation, nested scopes,
shared locks, half-applied cancellation rollback, operand copying, stream cleanup,
profile pooling, strict threshold semantics and independently derived NLL.
The real-model experiments run separately against cached weights, using torch,
transformers and numpy with local inference.

Snapshot memory is proportional to touched weights: each active
nested scope can retain snapshots of its touched parameters. Profiling requires
one full no-cache forward per distinct capture site per request; rules sharing a
site reuse a capture. Windowed evaluation retains at most a context-sized model
forward/logit workspace plus token IDs and per-window summaries. The projector
uses vector products and an outer product.