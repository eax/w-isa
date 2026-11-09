"""Admission gate: routed task progress, forced-action PPL and spectral measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time

import torch

from .backend import model_digest, route_observe
from .telemetry import summarize


class ModelStateError(RuntimeError):
    """The controller's registered base-state invariant changed."""


@dataclass(frozen=True)
class GatePolicy:
    max_ppl_increase_percent: float = 0.5
    min_corrected_cases: int = 1
    max_regressed_cases: int = 0
    min_repeat_reduction: float = 0.0
    max_relative_spectral_delta: float | None = None

    def __post_init__(self):
        values = [self.max_ppl_increase_percent, self.min_repeat_reduction]
        if self.max_relative_spectral_delta is not None:
            values.append(self.max_relative_spectral_delta)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("gate thresholds must be nonnegative and finite")
        for count in (self.min_corrected_cases, self.max_regressed_cases):
            if type(count) is not int or count < 0:
                raise ValueError("case thresholds must be nonnegative integers")


def decide(policy: GatePolicy, before, after, candidate_id, ppl_change, spectral, restored=True):
    if not before or [row.case_id for row in before] != [row.case_id for row in after]:
        raise ValueError("gate observations must cover the same nonempty ordered case set")
    corrected = sum(not a.correct and b.correct for a, b in zip(before, after))
    regressed = sum(a.correct and not b.correct for a, b in zip(before, after))
    repetition_gain = summarize(before)["mean_repeat_fraction"] - summarize(after)["mean_repeat_fraction"]
    matches = sum(row.matched_rule == candidate_id for row in after)
    reasons = []
    if matches == 0:
        reasons.append("candidate_not_selected")
    if corrected < policy.min_corrected_cases:
        reasons.append("corrected_case_requirement")
    if regressed > policy.max_regressed_cases:
        reasons.append("task_regression")
    if repetition_gain < policy.min_repeat_reduction:
        reasons.append("repetition_requirement")
    if corrected <= regressed and repetition_gain <= 0:
        reasons.append("task_progress_requirement")
    if not math.isfinite(ppl_change) or ppl_change > policy.max_ppl_increase_percent:
        reasons.append("perplexity_budget")
    if policy.max_relative_spectral_delta is not None:
        if any(row["relative_spectral_delta"] is None
               or row["relative_spectral_delta"] > policy.max_relative_spectral_delta for row in spectral):
            reasons.append("spectral_budget")
    if not restored:
        reasons.append("restoration_check")
    return {"accepted": not reasons, "reasons": reasons, "corrected_cases": corrected,
            "regressed_cases": regressed, "repeat_reduction": repetition_gain, "candidate_matches": matches}


class Verifier:
    def __init__(self, backend, policy=GatePolicy(), context_length=512, stride=256):
        self.backend, self.policy = backend, policy
        self.context_length, self.stride = context_length, stride
        self._baseline_ppl = {}

    @torch.no_grad()
    def evaluate(self, rule, table, cases, corpus_tokens):
        if len({case.case_id for case in cases}) != len(cases) or not cases:
            raise ValueError("validation case IDs must be unique and nonempty")
        start = time.perf_counter()
        rule = rule.clone()
        backend = self.backend
        with backend.runtime.lock:
            version, active, prospective = table.trial_snapshot(rule.rule_id)
            bound = next(candidate for candidate in prospective if candidate.rule_id == rule.rule_id)
            if bound.fingerprint != rule.fingerprint:
                raise ValueError("trial rule differs from its submitted definition")
            initial_hash = model_digest(backend.runtime.model)
            if initial_hash != backend.identity.weights_sha256:
                raise ModelStateError("base model state differs from its registered identity")
            program = backend.compile(rule)
            corpus = corpus_tokens.detach().cpu().contiguous()
            corpus_hash = hashlib.sha256(corpus.numpy().tobytes()).hexdigest()
            key = (corpus_hash, self.context_length, self.stride)
            if key not in self._baseline_ppl:
                self._baseline_ppl[key] = backend.score_corpus(corpus, self.context_length, self.stride)
            baseline_ppl = self._baseline_ppl[key]
            before = [route_observe(backend, case, active) for case in cases]
            after = [route_observe(backend, case, prospective) for case in cases]
            parameters = dict(backend.runtime.model.named_parameters(remove_duplicate=False))
            originals = {instruction.target: parameters[instruction.target].detach().cpu().double().clone()
                         for instruction in program.instructions}
            spectral = []
            # The candidate is forced active across the entire verifier corpus,
            # independent of whether the prompt-level trigger matches its text.
            with backend.runtime.temporary(program) as edit_reports:
                for target, original in originals.items():
                    edited = parameters[target].detach().cpu().double()
                    delta = edited - original
                    norm = torch.linalg.matrix_norm if original.ndim == 2 else torch.linalg.vector_norm
                    base_norm = norm(original, ord=2).item()
                    delta_norm = norm(delta, ord=2).item()
                    relative = delta_norm / base_norm if base_norm else (0.0 if delta_norm == 0 else None)
                    spectral.append({"target": target, "base_spectral_norm": base_norm,
                                     "delta_spectral_norm": delta_norm, "relative_spectral_delta": relative,
                                     "delta_frobenius_norm": torch.linalg.vector_norm(delta).item()})
                forced_ppl = backend.score_corpus(corpus, self.context_length, self.stride)
            restored_hash = model_digest(backend.runtime.model)
            restored = restored_hash == initial_hash
            ppl_change = 100 * math.expm1(forced_ppl["mean_nll"] - baseline_ppl["mean_nll"])
            decision = decide(self.policy, before, after, rule.rule_id, ppl_change, spectral, restored)
            report = {**decision, "rule_id": rule.rule_id, "rule_fingerprint": rule.fingerprint,
                      "table_version": version, "model_identity": asdict(backend.identity),
                      "policy": asdict(self.policy), "baseline": [o.to_dict() for o in before],
                      "routed_candidate": [o.to_dict() for o in after],
                      "baseline_summary": summarize(before), "candidate_summary": summarize(after),
                      "baseline_perplexity": baseline_ppl, "forced_candidate_perplexity": forced_ppl,
                      "ppl_change_percent": ppl_change, "ppl_tokens_sha256": corpus_hash,
                      "spectral": spectral, "restored_exact": restored,
                      "edit_reports": [asdict(row) for row in edit_reports],
                      "seconds": time.perf_counter() - start}
            if not restored:
                raise ModelStateError("model restoration check failed")
            return report