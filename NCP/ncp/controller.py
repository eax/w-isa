"""Telemetry -> candidates -> verification -> flow admission."""

from __future__ import annotations

from dataclasses import asdict

from .backend import route_observe
from .flows import FlowTable
from .synthesis import synthesize
from .telemetry import summarize
from .verifier import GatePolicy, ModelStateError, Verifier


class NeuralControlPlane:
    def __init__(self, backend, table: FlowTable | None = None, policy=GatePolicy(),
                 context_length=512, stride=256):
        self.backend = backend
        self.table = table if table is not None else FlowTable(backend.identity)
        if self.table.identity != backend.identity:
            raise ValueError("controller and table model identity differ")
        self.verifier = Verifier(backend, policy, context_length, stride)

    def infer(self, case):
        with self.backend.runtime.lock:
            _, rules = self.table.snapshot()
            observation = route_observe(self.backend, case, rules)
            if observation.matched_rule:
                self.table.record_hit(observation.matched_rule)
            return observation

    def step(self, development_cases, validation_cases, corpus_tokens, **synthesis_options):
        development_ids = {case.case_id for case in development_cases}
        validation_ids = {case.case_id for case in validation_cases}
        if (len(development_ids) != len(development_cases) or len(validation_ids) != len(validation_cases)
                or not development_ids or not validation_ids or development_ids & validation_ids
                or {case.prompt for case in development_cases} & {case.prompt for case in validation_cases}):
            raise ValueError("development and validation sets require disjoint unique cases")
        telemetry = [self.infer(case) for case in development_cases]
        candidates, profile = synthesize(self.backend, development_cases, telemetry, **synthesis_options)
        trials = []
        for rule in candidates:
            if self.table.contains(rule.rule_id):
                trials.append({"rule_id": rule.rule_id, "action": "already_recorded"})
                continue
            self.table.submit(rule)
            try:
                result = self.verifier.evaluate(rule, self.table, validation_cases, corpus_tokens)
                if result["accepted"]:
                    self.table.promote(rule.rule_id, result)
                    action = "RULE PROMOTED"
                else:
                    self.table.reject(rule.rule_id, result)
                    action = "RULE REJECTED - REVERTED"
            except Exception as error:
                result = {"accepted": False, "rule_id": rule.rule_id,
                          "rule_fingerprint": rule.fingerprint, "table_version": self.table.version,
                          "model_identity": asdict(self.backend.identity), "policy": asdict(self.verifier.policy),
                          "error": type(error).__name__, "message": str(error)}
                self.table.reject(rule.rule_id, result)
                if isinstance(error, ModelStateError):
                    raise
                action = "RULE REJECTED - REVERTED"
            trials.append({"rule_id": rule.rule_id, "action": action, "verification": result})
        return {"development": [o.to_dict() for o in telemetry], "development_summary": summarize(telemetry),
                "profile": profile, "trials": trials, "table_version": self.table.version}