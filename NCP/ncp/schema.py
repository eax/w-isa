"""Serializable flow definitions and model identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

import torch

from wisa.profiling import ActivationSite, cosine_score


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    weights_sha256: str


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class NeuralRule:
    rule_id: str
    identity: ModelIdentity
    site: ActivationSite
    direction: torch.Tensor
    threshold: float
    source: dict
    operands: dict[str, torch.Tensor] = field(default_factory=dict)
    priority: int = 0
    hard_timeout_seconds: float | None = None
    idle_timeout_seconds: float | None = None
    token_budget: int | None = None
    hypothesis: str = ""

    def __post_init__(self):
        if not self.rule_id or not math.isfinite(self.threshold):
            raise ValueError("a rule ID and finite threshold are required")
        if type(self.priority) is not int:
            raise ValueError("priority must be an integer")
        cosine_score(self.direction, self.direction)
        for value in (self.hard_timeout_seconds, self.idle_timeout_seconds):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("timeouts must be positive finite seconds")
        if self.token_budget is not None and (type(self.token_budget) is not int or self.token_budget <= 0):
            raise ValueError("token budget must be a positive integer")

    def payload(self) -> dict:
        return {"rule_id": self.rule_id, "identity": asdict(self.identity), "site": asdict(self.site),
                "direction": self.direction.detach().cpu().double().tolist(), "threshold": self.threshold,
                "source": self.source,
                "operands": {k: v.detach().cpu().double().tolist() for k, v in self.operands.items()},
                "priority": self.priority, "hard_timeout_seconds": self.hard_timeout_seconds,
                "idle_timeout_seconds": self.idle_timeout_seconds, "token_budget": self.token_budget,
                "hypothesis": self.hypothesis}

    @classmethod
    def from_payload(cls, payload: dict) -> NeuralRule:
        data = dict(payload)
        data["identity"] = ModelIdentity(**data["identity"])
        data["site"] = ActivationSite(**data["site"])
        data["direction"] = torch.tensor(data["direction"], dtype=torch.float64, device="cpu")
        data["operands"] = {k: torch.tensor(v, dtype=torch.float64, device="cpu")
                            for k, v in data["operands"].items()}
        return cls(**data)

    def clone(self) -> NeuralRule:
        return self.from_payload(json.loads(json.dumps(self.payload(), allow_nan=False)))

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.payload())