"""Token-weighted causal NLL with explicit, overlapping evaluation windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch
import torch.nn.functional as F


def prediction_windows(length: int, context_length: int, stride: int):
    """Yield (input_begin, target_end, first_scored_target), globally indexed.

    Input ends at target_end-1, so the final input logit predicts the final
    target. Every token except token 0 is scored exactly once, including tails.
    """
    if length < 2:
        raise ValueError("at least two tokens are required")
    if context_length < 1 or not 1 <= stride <= context_length:
        raise ValueError("require 1 <= stride <= context_length")
    scored = 1
    end = min(length, context_length + 1)
    while scored < length:
        begin = max(0, end - 1 - context_length)
        yield begin, end, scored
        scored = end
        end = min(length, end + stride)


@dataclass(frozen=True)
class PerplexityResult:
    scored_tokens: int
    total_nll: float
    mean_nll: float
    perplexity: float
    context_length: int
    stride: int
    seconds: float
    windows: tuple[dict, ...]

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def perplexity(model, token_ids: torch.Tensor, context_length=512, stride=256) -> PerplexityResult:
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a one-dimensional int64 tensor")
    if model.training:
        raise ValueError("perplexity requires model.eval()")
    device = next(model.parameters()).device
    ids = token_ids.detach().cpu()
    start_time = time.perf_counter()
    windows = []
    total = 0.0
    count = 0
    for begin, end, first in prediction_windows(len(ids), context_length, stride):
        inputs = ids[begin:end - 1].unsqueeze(0).to(device)
        logits = model(input_ids=inputs, use_cache=False).logits[0]
        selected = logits[first - begin - 1:]
        labels = ids[first:end].to(device)
        losses = F.cross_entropy(selected.float(), labels, reduction="none").cpu().double()
        if not torch.isfinite(losses).all().item():
            raise ValueError("nonfinite token loss")
        nll = losses.sum().item()
        windows.append({"input_begin": begin, "target_begin": first, "target_end": end,
                        "tokens": len(labels), "nll": nll})
        total += nll
        count += len(labels)
    if count != len(ids) - 1:
        raise RuntimeError("token accounting mismatch")
    mean = total / count
    return PerplexityResult(count, total, mean, math.exp(mean), context_length, stride,
                            time.perf_counter() - start_time, tuple(windows))