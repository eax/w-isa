"""Contrastive activation measurement without training or backward passes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .model import encode_prompt
from .runtime import WeightRuntime


@dataclass(frozen=True)
class ActivationSite:
    module: str
    location: str = "input"
    pooling: str = "last"

    def __post_init__(self):
        if self.location not in {"input", "output"}:
            raise ValueError("activation location must be input or output")
        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling must be last or mean")


@dataclass(frozen=True)
class ContrastProfile:
    direction: torch.Tensor
    positive_mean: torch.Tensor
    negative_mean: torch.Tensor
    metadata: dict

    def operands(self, name="v_diff"):
        return {name: self.direction.detach().clone()}

    def save(self, stem: str | Path):
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stem.with_suffix(".npz"), v_diff=self.direction.numpy(),
                 positive_mean=self.positive_mean.numpy(), negative_mean=self.negative_mean.numpy())
        stem.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2) + "\n")


class ActivationProfiler:
    def __init__(self, runtime: WeightRuntime, tokenizer, site: ActivationSite):
        self.runtime, self.tokenizer, self.site = runtime, tokenizer, site
        self.module = runtime.model.get_submodule(site.module)

    @torch.no_grad()
    def capture(self, prompt: str) -> torch.Tensor:
        with self.runtime.lock:
            model = self.runtime.model
            encoded = encode_prompt(self.tokenizer, prompt, next(model.parameters()).device)
            captured = []

            def collect(value):
                if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
                    raise ValueError("activation site must expose a batch-one [batch, token, feature] tensor")
                mask = encoded["attention_mask"][0].to(device=value.device, dtype=torch.bool)
                if value.shape[1] != len(mask) or not mask.any().item():
                    raise ValueError("activation and prompt token positions do not match")
                tokens = value[0, mask].detach().cpu().double()
                vector = tokens[-1] if self.site.pooling == "last" else tokens.mean(dim=0)
                if not torch.isfinite(vector).all().item():
                    raise ValueError("nonfinite activation")
                captured.append(vector.clone())

            if self.site.location == "input":
                handle = self.module.register_forward_pre_hook(lambda _module, args: collect(args[0]))
            else:
                handle = self.module.register_forward_hook(lambda _module, _args, output: collect(output))
            try:
                model(**encoded, use_cache=False)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise ValueError("activation site must execute exactly once per prompt forward")
            return captured[0]

    @torch.no_grad()
    def contrast(self, positive: list[str], negative: list[str], model_revision=None) -> ContrastProfile:
        if not positive or not negative:
            raise ValueError("both contrast sets must be nonempty")
        with self.runtime.lock:
            pos = torch.stack([self.capture(prompt) for prompt in positive])
            neg = torch.stack([self.capture(prompt) for prompt in negative])
            positive_mean, negative_mean = pos.mean(dim=0), neg.mean(dim=0)
            direction = positive_mean - negative_mean
            if not torch.isfinite(direction).all().item() or torch.linalg.vector_norm(direction).item() == 0:
                raise ValueError("contrast sets produced a zero or nonfinite direction")
            prompt_digest = hashlib.sha256(json.dumps([positive, negative]).encode()).hexdigest()
            metadata = {"site": asdict(self.site), "positive_count": len(positive),
                        "negative_count": len(negative), "feature_count": len(direction),
                        "dtype": str(direction.dtype), "model_revision": model_revision,
                        "model_id": getattr(getattr(self.runtime.model, "config", None), "_name_or_path", None),
                        "prompt_sha256": prompt_digest,
                        "direction_sha256": hashlib.sha256(direction.numpy().tobytes()).hexdigest(),
                        "direction_norm": torch.linalg.vector_norm(direction).item()}
            return ContrastProfile(direction, positive_mean, negative_mean, metadata)


def cosine_score(activation: torch.Tensor, direction: torch.Tensor) -> float:
    x, v = activation.detach().cpu().double(), direction.detach().cpu().double()
    if x.ndim != 1 or x.shape != v.shape or x.numel() == 0:
        raise ValueError("trigger and activation must be nonempty vectors in the same feature space")
    if not torch.isfinite(x).all().item() or not torch.isfinite(v).all().item():
        raise ValueError("trigger vectors must be finite")
    if v.abs().max().item() == 0:
        raise ValueError("trigger direction must be nonzero")
    if x.abs().max().item() == 0:
        return 0.0
    x, v = x / x.abs().max(), v / v.abs().max()
    score = torch.dot(x / torch.linalg.vector_norm(x), v / torch.linalg.vector_norm(v)).item()
    return max(-1.0, min(1.0, score))


def classification_metrics(scores: list[float], labels: list[bool], threshold: float) -> dict:
    if not scores or len(scores) != len(labels) or not all(math.isfinite(x) for x in scores):
        raise ValueError("finite scores and matching nonempty labels are required")
    if not math.isfinite(threshold) or any(type(label) is not bool for label in labels):
        raise ValueError("finite threshold and boolean labels are required")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("metrics require both positive and negative examples")
    tp = sum(score > threshold and label for score, label in zip(scores, labels))
    fp = sum(score > threshold and not label for score, label in zip(scores, labels))
    return {"threshold": threshold, "true_positive": tp, "false_positive": fp,
            "false_negative": positives - tp, "true_negative": negatives - fp,
            "tpr": tp / positives, "fpr": fp / negatives,
            "balanced_accuracy": 0.5 * (tp / positives + 1 - fp / negatives)}


def calibrate_threshold(scores: list[float], labels: list[bool]) -> dict:
    classification_metrics(scores, labels, 0.0)
    values = sorted(set(scores))
    candidates = [math.nextafter(values[0], -math.inf), *values]
    metrics = [classification_metrics(scores, labels, threshold) for threshold in candidates]
    return max(metrics, key=lambda row: (row["balanced_accuracy"], -row["fpr"], row["threshold"]))