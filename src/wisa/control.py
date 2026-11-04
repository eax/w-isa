"""Ordered match-action rules and model-scoped request isolation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
import time

import torch

from .compiler import Program
from .model import generate
from .profiling import ActivationProfiler, ActivationSite, cosine_score
from .runtime import WeightRuntime


@dataclass(frozen=True)
class Rule:
    name: str
    site: ActivationSite
    direction: torch.Tensor
    threshold: float
    program: Program


@dataclass(frozen=True)
class Decision:
    matched_rule: str | None
    scores: tuple[tuple[str, float], ...]
    profile_seconds: float
    apply_seconds: float


class MatchActionController:
    """First matching rule wins, in caller-supplied order.

    All scores are computed before applying the selected patch. Requests with
    no match acquire the same model lock as patched requests. No cache from the
    profiling forward is reused by generation. Rules are copied on registration.
    """

    def __init__(self, runtime: WeightRuntime, tokenizer, rules: list[Rule]):
        if len({rule.name for rule in rules}) != len(rules):
            raise ValueError("rule names must be unique")
        self.runtime, self.tokenizer = runtime, tokenizer
        copied = []
        for rule in rules:
            if not rule.name or not math.isfinite(rule.threshold):
                raise ValueError("rule name and finite threshold are required")
            direction = rule.direction.detach().cpu().double().clone()
            cosine_score(direction, direction)
            runtime.model.get_submodule(rule.site.module)
            program = Program(tuple(replace(instruction, operands=tuple(value.detach().clone()
                                    for value in instruction.operands)) for instruction in rule.program.instructions))
            copied.append(Rule(rule.name, rule.site, direction, rule.threshold, program))
        self.rules = tuple(copied)

    @contextmanager
    def session(self, prompt: str):
        with self.runtime.lock:
            start = time.perf_counter()
            activations = {}
            scores = []
            selected = None
            for rule in self.rules:
                if rule.site not in activations:
                    activations[rule.site] = ActivationProfiler(self.runtime, self.tokenizer, rule.site).capture(prompt)
                score = cosine_score(activations[rule.site], rule.direction)
                scores.append((rule.name, score))
                if selected is None and score > rule.threshold:
                    selected = rule
            profile_seconds = time.perf_counter() - start
            start = time.perf_counter()
            with self.runtime.temporary(selected.program if selected else None):
                yield Decision(selected.name if selected else None, tuple(scores), profile_seconds,
                               time.perf_counter() - start)

    def run(self, prompt: str, inference):
        """Run a synchronous inference callback inside the request scope."""
        start = time.perf_counter()
        with self.session(prompt) as decision:
            inference_start = time.perf_counter()
            value = inference()
            inference_seconds = time.perf_counter() - inference_start
        total = time.perf_counter() - start
        return {"value": value, "decision": decision, "total_seconds": total,
                "inference_seconds": inference_seconds, "control_seconds": total - inference_seconds}

    def generate(self, prompt: str, max_new_tokens=64):
        return self.run(prompt, lambda: generate(self.runtime.model, self.tokenizer, prompt, max_new_tokens))

    def stream(self, prompt: str, inference_iterator):
        """Consume a synchronous iterator within the scope; close on disconnect.

        Consumers must close this generator (or use contextlib.closing) when
        stopping early, on the same thread that began iteration.
        """
        with self.session(prompt):
            iterator = iter(inference_iterator())
            try:
                for item in iterator:
                    yield item
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()