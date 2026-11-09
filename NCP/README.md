# NCP — Neural Control Plane

NCP runs a closed loop over W-ISA: observe development examples, derive a failure
direction, synthesize candidate programs, evaluate them, and admit rules into a
versioned flow table. Accepted rules are applied within W-ISA request scopes;
the base model's stored weights return to their entry state after each request.

The first implementation uses deterministic Python synthesis. Its default
candidate set contains an MLP projection, an attention-head scale, and their
ordered combination. The model, layers, heads, scale factors, match threshold
and admission budgets are configurable.

## Research context and attribution

We credit **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab** and their
November 2, 2026 article,
[*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/),
for the weight-space programming perspective and its connection to recursive
self-improvement. NCP develops a verifier-guided flow-policy loop on top of W-ISA.

Additional context from that article's bibliography includes
[*Large Language Models as Optimizers*](https://arxiv.org/abs/2309.03409)
for evaluated candidate generation,
[*DSPy*](https://arxiv.org/abs/2310.03714) for declarative program optimization,
[*Wait, Wait, Wait... Why Do Reasoning Models Loop?*](https://arxiv.org/abs/2512.12895)
for repetition analysis, and
[*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717)
for directional intervention studies.

## Run from the repository root

NCP imports the parent project's `wisa` package. Use the existing workspace
environment, or install the parent checkout using `.venv/bin/python -m pip install -e .`.
The experiment uses a cached SmolLM checkpoint and the local WikiText-2 test cache
created by [the parent downloader](../examples/fetch_wikitext.py).

Run the local loop:

```sh
PYTHONPATH=NCP:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m ncp --device mps --output local/ncp/run.json
```

Use `--device cpu`, `--device cuda`, or an indexed CUDA device to select another
backend. `--help` lists the experiment options. For a second cycle add `--cycles 2`;
for a saved table, pass `--resume local/ncp/run.flows.json` and a new output path.
Reports and flow snapshots default to the repository's ignored local directory.

Run the offline NCP tests:

```sh
PYTHONPATH=NCP:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m unittest discover -s NCP/tests -v
```

## Components

| Component | Role |
| --- | --- |
| [schema](ncp/schema.py) | Model identity, named rules, trigger direction, W-ISA source and operands |
| [flow table](ncp/flows.py) | Candidate/active/rejected/expired/retired states, priority, TTL, hits and atomic JSON persistence |
| [telemetry](ncp/telemetry.py) | Final-answer scoring, repeated n-grams, adjacent-repeat position, repetition run length and observation records |
| [backend](ncp/backend.py) | Local generation, per-step decoding-score entropy, activation capture and request-scoped routing |
| [synthesis](ncp/synthesis.py) | Candidate construction from development outcomes and actual model dimensions |
| [verifier](ncp/verifier.py) | Routed task comparison, forced-action perplexity, spectral measurements and admission decision |
| [controller](ncp/controller.py) | Observe → synthesize → verify → promote/archive cycle and active-rule inference |

`NeuralControlPlane`, `NeuralRule`, `FlowTable`, `TorchBackend`, `TaskCase`,
`GatePolicy` and `Verifier` are exported by `ncp`.

## Flow lifecycle

Rules bind to a model identifier, revision and SHA-256 digest of parameter names,
shapes, dtypes and bytes. A definition includes its activation site, cosine
direction, threshold, priority, program and numeric operands. The table owns
copies of submitted definitions. Candidate IDs are derived deterministically
from their intervention and matching configuration.

Higher numeric priority runs first; equal priorities retain registration order.
The first matching rule is selected. Activations are captured before applying
that rule. The request then enters `WeightRuntime.temporary()`, generates using
a fresh cache, and restores the parameter state on scope exit.

Promotion persists the rule definition and admission evidence in the table.
The physical weight patch remains scoped to each selected request.

`hard_timeout_seconds` measures elapsed wall-clock time since promotion.
`idle_timeout_seconds` measures time since promotion or the most recent completed
selected request. Expiration is applied when the table is inspected for admission
or dispatch. An in-flight request completes under its selected scope. Timestamps
remain intact across save/load, so elapsed time continues to count across restart.

`token_budget` caps generation for a selected rule, separately from the timeouts.
The CLI exposes these as `--hard-timeout`, `--idle-timeout`, and
`--rule-token-budget`. Model-wide generation uses `--max-new-tokens`.

Admission evidence records the rule fingerprint and active-table version.
Promotion checks both; a changed active table requires fresh evidence. Snapshots
and saved tables preserve rule order, status, timestamps, hits and evaluation
records. The current loop recognizes already-recorded candidate IDs and retains
their earlier decisions.

## Telemetry and synthesis

The [reasoning fixture](examples/reasoning_cases.json) contains three development,
three validation and three final-test questions with disjoint IDs and prompts.
The expected response ends with `Final answer: <number>`. Scoring compares the
last captured answer, or the entire response when the pattern is absent, with
the expected answer after case and trailing-period normalization.

Generation records text, token IDs, elapsed time, token-budget completion,
per-step decoding-score entropy in nats, repeated n-gram fraction and adjacent
repetition runs. Default n-gram size is four. These are output-token measurements.

Development outcomes label prompt-end activations. With both success and failure
examples, synthesis uses mean(failure) − mean(success). With an all-failure
development set it uses the failure centroid and records that choice explicitly.
The direction and trigger are measured at the same module input and pooling site.

For each selected block, the generator creates `PROJECT_OUT` on the MLP
down-projection, `SCALE` on a selected head's output-projection columns, and their
combination. Dimensions come from the loaded model. The default targets are
zero-based layer 9, head 0, scale factor 0.0, and cosine threshold 0.8.

## Admission policy

The verifier compares active-table routing with prospective routing after adding
the candidate. It records selection counts, corrected and regressed answers,
and the change in mean repeated n-gram fraction. Validation uses the candidate's
actual priority and the existing rules' order.

For corpus evaluation, the candidate action is **forced active across the entire
verifier slice**, independently of its trigger. W-ISA evaluates the original and
edited parameters under identical tokenization and rolling-window settings.
This gives a direct action-cost measurement alongside routed task effects.

Defaults require at least one corrected validation answer, zero regressed answers,
a nonnegative reduction in repetition, and perplexity growth of at most 0.5%.
Admission also requires a selected candidate, positive task/repetition progress,
and exact restoration of the base parameter digest. CLI options configure each
numeric budget.

The verifier records $\lVert W\rVert_2$, $\lVert\Delta W\rVert_2$,
$\lVert\Delta W\rVert_F$, and the relative spectral change. A supplied
`--max-relative-spectral-delta` adds a spectral admission threshold.

The default corpus protocol partitions the cached text into two disjoint
4,096-token slices. The first is used during admission; the second is evaluated
after candidate selection. Each slice scores 4,095 targets with context 512 and
stride 256. Final task examples are also evaluated after selection.

## Recorded first run

The initial run used SmolLM-135M-Instruct revision
`b5f4b394abe60fe0e4d666139e57bfce520302a9`, FP32 weights on MPS, CPU FP64 edits,
and a 128-token generation budget. Development scoring recorded zero exact
answers out of three, so synthesis used a 1,536-dimensional failure centroid
with norm 3.6363433948. All three candidates were selected on all three
validation prompts.

The baseline validation repetition fraction was 0.41333333. Baseline verifier
perplexity was 23.74291741 on the first 4,095 scored targets.

| Candidate | Validation repetition fraction | Corrected answers | Forced-action PPL | PPL change | Admission |
| --- | ---: | ---: | ---: | ---: | --- |
| MLP projection | 0.21906173 | 0 | 23.72929551 | -0.0573725% | Archived |
| Head scale 0.0 | 0.36800000 | 0 | 23.78039795 | +0.1578599% | Archived |
| Combined | 0.51733333 | 0 | 23.77012175 | +0.1145787% | Archived |

The corrected-answer admission threshold was one; the active table remained at
zero rules. Final exact-answer scoring was 0/3 for original and routed execution;
both had a repetition fraction of 0.256. Final-slice baseline perplexity was
33.72129164. Every trial and the complete run restored the original parameter
digest. The initial end-to-end experiment took 44.38 seconds.

A fresh-process two-cycle run matched first-cycle observations, token entropies,
candidate fingerprints, spectral measurements and perplexity values exactly.
The second cycle recognized the three recorded candidate IDs. Loading the saved
flow table in a third process preserved those decisions and final outputs.
The raw reports are generated under local/ncp by the CLI.

**25 NCP tests passed.** Controlled fixtures exercise both promotion and
rejection using real W-ISA weight writes. The suite also covers TTL expiration
across restart, stale evidence, priority shadowing, rule copying, token-budget
dispatch, forced PPL on unmatched inputs, cancellation and scalar parameter hashes.
The parent W-ISA regression run recorded 62 passing tests and six hardware-gated
CUDA cases skipped by device discovery.

## Cost and state ownership

Admission runs task inference, forced-corpus evaluation and spectral norms as
separate measured operations. Original-corpus scores are cached per token digest
and scoring protocol. Candidate task evaluations use fresh request scopes.
Model parameter digests are checked around verification.

The table serializes definitions and evidence as JSON; saving uses a flushed
temporary file followed by atomic replacement. Tables are trusted local control
data. Application code supplies task definitions, rule inputs and model objects.
The parent runtime coordinates model access through its shared lock.

NCP is part of the parent [MIT-licensed project](../LICENSE), copyright 2026
Ekrem Aksoy. Implementation files and fixtures live in this directory; execution
artifacts are kept under the ignored local directory.