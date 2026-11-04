# Contributing

Small, reproducible changes are easiest to review. Describe the expected behavior,
include a minimal example, and state which device and dtype you tested.

Install from a checkout using `python3 -m venv .venv` followed by
`.venv/bin/python -m pip install -e .`. Run the offline suite with:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests use local synthetic tensors and a tiny random Qwen model. CPU, MPS and CUDA
test groups are selected through PyTorch device discovery. Include the test
counts, selected devices and dtypes in the pull request. Python 3.12.9 is the
locally measured interpreter; the declared minimum is 3.10. The CI matrix uses
Python 3.10 and 3.12 with CPU PyTorch.

For weight operations, test rounding, rejected inputs, exact restoration, and
changes outside the requested slice. A known matrix or independently calculated
NumPy result makes a stronger reference than calling the implementation twice.
For request scopes, include failure and cancellation paths. Use separate profile,
calibration and evaluation splits for activation-based experiments.

Pretrained experiments belong under examples.
Record model revision, prompt or corpus provenance, scoring protocol and device.
Report logit changes, generated token sequences and perplexity as separate
measurements, retaining the complete results for each experiment condition.

Pull requests contain source, tests, user documentation and experiment evidence.
Keep credentials, environments, downloaded corpora and model caches in their local
locations. Project contributions use the MIT License; third-party material carries
its own attribution and applicable license terms.