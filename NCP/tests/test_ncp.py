import asyncio
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from wisa import WeightRuntime, compile_program
from wisa.profiling import ActivationSite

from ncp.backend import model_digest, route_observe
from ncp.controller import NeuralControlPlane
from ncp.flows import FlowTable
from ncp.schema import ModelIdentity, NeuralRule
from ncp.synthesis import synthesize
from ncp.telemetry import Observation, TaskCase, repetition_metrics
from ncp.verifier import GatePolicy, ModelStateError, Verifier, decide


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class FixtureBackend:
    def __init__(self, ppl_penalty=0.0):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        self.runtime = WeightRuntime(model)
        self.identity = ModelIdentity("fixture", "1", model_digest(model))
        self.ppl_penalty = ppl_penalty
        self.ppl_weights = []

    def capture(self, prompt, site):
        return torch.ones(1, dtype=torch.float64)

    def compile(self, rule):
        return compile_program(rule.source, self.runtime.model, rule.operands)

    def observe(self, case, token_budget=None):
        edited = self.runtime.model.weight.item() == 0
        text = "Final answer: 1" if edited else "Final answer: 0"
        tokens = [1, 2, 3, 4] if edited else [1, 2] * 8
        return Observation(case.case_id, text, tokens, case.score(text), [0.1] * len(tokens),
                           repetition_metrics(tokens), False, 0.0)

    def score_corpus(self, tokens, context_length=512, stride=256):
        weight = self.runtime.model.weight.item()
        self.ppl_weights.append(weight)
        mean = 1.0 + (self.ppl_penalty if weight == 0 else 0)
        return {"mean_nll": mean, "perplexity": math.exp(mean), "scored_tokens": len(tokens) - 1}


def rule_for(backend, name="candidate", priority=0, threshold=0.8, **kwargs):
    return NeuralRule(name, backend.identity, ActivationSite("weight"), torch.ones(1), threshold,
                      {"version": 1, "instructions": [{"op": "SCALE", "target": "weight", "factor": 0}]},
                      priority=priority, **kwargs)


def accepted(rule, table):
    return {"accepted": True, "rule_fingerprint": rule.fingerprint, "table_version": table.version}


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.backend = FixtureBackend()
        self.clock = Clock()
        self.table = FlowTable(self.backend.identity, self.clock)

    def test_priority_and_stable_order(self):
        for name, priority in (("first", 2), ("high", 5), ("second", 2)):
            rule = rule_for(self.backend, name, priority)
            self.table.submit(rule)
            self.table.promote(name, accepted(rule, self.table))
        self.assertEqual([r.rule_id for r in self.table.snapshot()[1]], ["high", "first", "second"])

    def test_strict_promotion_and_stale_evidence(self):
        first, second = rule_for(self.backend, "first"), rule_for(self.backend, "second")
        self.table.submit(first)
        self.table.submit(second)
        old = accepted(second, self.table)
        self.table.promote(first.rule_id, accepted(first, self.table))
        with self.assertRaises(ValueError):
            self.table.promote(second.rule_id, old)
        with self.assertRaises(ValueError):
            self.table.promote(second.rule_id, {**accepted(second, self.table), "accepted": False})
        self.assertEqual(len(self.table.snapshot()[1]), 1)

    def test_submit_and_snapshot_copy_mutable_values(self):
        rule = rule_for(self.backend)
        self.table.submit(rule)
        fingerprint = rule.fingerprint
        rule.direction.fill_(7)
        rule.source["instructions"][0]["factor"] = 3
        self.table.promote(rule.rule_id, {"accepted": True, "rule_fingerprint": fingerprint, "table_version": 0})
        snapshot = self.table.snapshot()[1][0]
        snapshot.direction.fill_(19)
        self.assertEqual(self.table.snapshot()[1][0].direction.item(), 1)
        self.assertEqual(self.table.snapshot()[1][0].source["instructions"][0]["factor"], 0)

    def test_hard_and_idle_expiry(self):
        rule = rule_for(self.backend, hard_timeout_seconds=10, idle_timeout_seconds=4)
        self.table.submit(rule)
        self.table.promote(rule.rule_id, accepted(rule, self.table))
        self.clock.now = 103
        self.table.record_hit(rule.rule_id)
        self.clock.now = 106
        self.assertEqual(len(self.table.snapshot()[1]), 1)
        self.clock.now = 107
        self.assertEqual(len(self.table.snapshot()[1]), 0)
        self.assertEqual(self.table.export()["entries"][0]["status"], "expired")
        self.assertEqual(self.table.version, 2)

    def test_persistence_roundtrip_and_identity_check(self):
        rule = rule_for(self.backend)
        self.table.submit(rule)
        self.table.promote(rule.rule_id, accepted(rule, self.table))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.json"
            self.table.save(path)
            loaded = FlowTable.load(path, self.backend.identity, self.clock)
            self.assertEqual(loaded.export(), self.table.export())
            with self.assertRaises(ValueError):
                FlowTable.load(path, ModelIdentity("other", "1", "hash"))
            data = json.loads(path.read_text())
            data["entries"][0]["rule"]["threshold"] = 0.2
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                FlowTable.load(path, self.backend.identity)

    def test_reject_retire_and_invalid_timeouts(self):
        rule = rule_for(self.backend)
        self.table.submit(rule)
        self.table.reject(rule.rule_id, {"accepted": False})
        self.assertEqual(self.table.snapshot()[1], ())
        self.table.retire(rule.rule_id)
        self.assertEqual(self.table.export()["entries"][0]["status"], "retired")
        with self.assertRaises(ValueError):
            rule_for(self.backend, hard_timeout_seconds=0)
        with self.assertRaises(ValueError):
            self.table.submit(rule)

    def test_trial_order_uses_registration_order(self):
        early = rule_for(self.backend, "early")
        late = rule_for(self.backend, "late")
        self.table.submit(early)
        self.table.submit(late)
        self.table.promote(late.rule_id, accepted(late, self.table))
        _, before, after = self.table.trial_snapshot(early.rule_id)
        self.assertEqual([r.rule_id for r in before], ["late"])
        self.assertEqual([r.rule_id for r in after], ["early", "late"])

    def test_expiry_invalidates_previously_measured_evidence(self):
        active = rule_for(self.backend, "active", hard_timeout_seconds=1)
        candidate = rule_for(self.backend, "new")
        self.table.submit(active)
        self.table.promote(active.rule_id, accepted(active, self.table))
        self.table.submit(candidate)
        old_evidence = accepted(candidate, self.table)
        self.clock.now += 1
        with self.assertRaises(ValueError):
            self.table.promote(candidate.rule_id, old_evidence)
        self.assertEqual(self.table.snapshot()[1], ())

    def test_persisted_ttl_counts_elapsed_time_across_restart(self):
        rule = rule_for(self.backend, hard_timeout_seconds=5)
        self.table.submit(rule)
        self.table.promote(rule.rule_id, accepted(rule, self.table))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flows.json"
            self.table.save(path)
            self.clock.now += 5
            loaded = FlowTable.load(path, self.backend.identity, self.clock)
            self.assertEqual(loaded.snapshot()[1], ())
            self.assertEqual(loaded.export()["entries"][0]["status"], "expired")

    def test_invalid_evidence_cannot_partially_activate_a_rule(self):
        rule = rule_for(self.backend)
        self.table.submit(rule)
        with self.assertRaises(ValueError):
            self.table.promote(rule.rule_id, {**accepted(rule, self.table), "bad": float("nan")})
        self.assertEqual(self.table.export()["entries"][0]["status"], "candidate")
        self.assertEqual(self.table.version, 0)


class TelemetryTests(unittest.TestCase):
    def test_repetition_counts_and_short_sequences(self):
        result = repetition_metrics([1, 2] * 4, 2)
        self.assertEqual(result["ngram_count"], 7)
        self.assertEqual(result["repeated_ngrams"], 5)
        self.assertEqual(result["max_consecutive_repeats"], 4)
        self.assertEqual(result["first_adjacent_repeat_token"], 2)
        self.assertEqual(repetition_metrics([1, 2], 4)["repeat_fraction"], 0)

    def test_final_answer_scoring(self):
        case = TaskCase("math", "question", "42")
        self.assertTrue(case.score("Reasoning.\nFinal answer: 42."))
        self.assertFalse(case.score("I see 42 somewhere but the answer is 0"))
        self.assertTrue(case.score("42"))
        self.assertFalse(case.score("Final answer: 42\nFinal answer: 0"))

    def test_scalar_parameter_fingerprint(self):
        model = nn.Module()
        model.scalar = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        original = model_digest(model)
        model.scalar.fill_(2)
        self.assertNotEqual(model_digest(model), original)
        model.scalar.fill_(1)
        self.assertEqual(model_digest(model), original)


class VerificationTests(unittest.TestCase):
    def evaluate(self, backend, threshold=0.8):
        table = FlowTable(backend.identity)
        rule = rule_for(backend, threshold=threshold)
        table.submit(rule)
        cases = [TaskCase("validation", "held-out", "1")]
        report = Verifier(backend).evaluate(rule, table, cases, torch.tensor([0, 1, 2]))
        return rule, table, report

    def test_real_wisa_trial_promotes_and_restores(self):
        backend = FixtureBackend()
        rule, table, report = self.evaluate(backend)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["corrected_cases"], 1)
        self.assertEqual(report["spectral"][0]["relative_spectral_delta"], 1)
        self.assertEqual(backend.runtime.model.weight.item(), 1)
        table.promote(rule.rule_id, report)
        served = route_observe(backend, TaskCase("request", "new", "1"), table.snapshot()[1])
        self.assertTrue(served.correct)
        self.assertEqual(backend.runtime.model.weight.item(), 1)

    def test_ppl_rejects_even_when_task_is_corrected(self):
        _, _, report = self.evaluate(FixtureBackend(ppl_penalty=0.01))
        self.assertEqual(report["corrected_cases"], 1)
        self.assertFalse(report["accepted"])
        self.assertIn("perplexity_budget", report["reasons"])

    def test_forced_ppl_runs_when_trigger_never_matches(self):
        backend = FixtureBackend(ppl_penalty=0.01)
        _, _, report = self.evaluate(backend, threshold=1.0)
        self.assertEqual(report["candidate_matches"], 0)
        self.assertEqual(backend.ppl_weights, [1.0, 0.0])
        self.assertIn("perplexity_budget", report["reasons"])
        self.assertEqual(backend.runtime.model.weight.item(), 1)

    def test_wrong_base_identity_stops_evaluation(self):
        backend = FixtureBackend()
        with torch.no_grad():
            backend.runtime.model.weight.fill_(2)
        with self.assertRaises(ModelStateError):
            self.evaluate(backend)

    def test_cancellation_inside_routed_generation_restores(self):
        backend = FixtureBackend()
        rule = rule_for(backend)
        with patch.object(backend, "observe", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                route_observe(backend, TaskCase("cancel", "prompt", "1"), (rule,))
        self.assertEqual(backend.runtime.model.weight.item(), 1)

    def test_selected_rule_passes_its_token_budget(self):
        backend = FixtureBackend()
        rule = rule_for(backend, token_budget=3)
        with patch.object(backend, "observe", wraps=backend.observe) as observe:
            route_observe(backend, TaskCase("request", "prompt", "1"), (rule,))
        self.assertEqual(observe.call_args.args[1], 3)
        self.assertEqual(backend.runtime.model.weight.item(), 1)

    def test_controller_cycle_then_serves_promoted_rule(self):
        backend = FixtureBackend()
        rule = rule_for(backend)
        controller = NeuralControlPlane(backend)
        development, validation = [TaskCase("dev", "dev prompt", "1")], [TaskCase("val", "val prompt", "1")]
        with patch("ncp.controller.synthesize", return_value=([rule], {"fixture": True})):
            report = controller.step(development, validation, torch.tensor([0, 1, 2]))
            self.assertEqual(report["trials"][0]["action"], "RULE PROMOTED")
            again = controller.step(development, validation, torch.tensor([0, 1, 2]))
            self.assertEqual(again["trials"][0]["action"], "already_recorded")
        result = controller.infer(TaskCase("new", "new prompt", "1"))
        self.assertTrue(result.correct)
        self.assertEqual(controller.table.export()["entries"][0]["hits"], 2)
        self.assertEqual(backend.runtime.model.weight.item(), 1)

    def test_controller_rejects_overlapping_splits(self):
        backend = FixtureBackend()
        case = TaskCase("same", "same", "1")
        with self.assertRaises(ValueError):
            NeuralControlPlane(backend).step([case], [case], torch.tensor([0, 1]))

    def test_spectral_policy_and_noop_progress_requirement(self):
        backend = FixtureBackend()
        case = TaskCase("case", "prompt", "1")
        before = backend.observe(case)
        rule = rule_for(backend)
        with backend.runtime.temporary(backend.compile(rule)):
            after = backend.observe(case)
        after.matched_rule = rule.rule_id
        result = decide(GatePolicy(max_relative_spectral_delta=0.25), [before], [after], rule.rule_id,
                        0.0, [{"relative_spectral_delta": 1.0}])
        self.assertIn("spectral_budget", result["reasons"])
        before.matched_rule = rule.rule_id
        result = decide(GatePolicy(min_corrected_cases=0), [before], [before], rule.rule_id, 0.0, [])
        self.assertIn("task_progress_requirement", result["reasons"])

    def test_existing_priority_can_shadow_a_candidate(self):
        backend = FixtureBackend()
        table = FlowTable(backend.identity)
        active = rule_for(backend, "active", priority=5)
        candidate = rule_for(backend, "candidate", priority=0)
        table.submit(active)
        table.promote(active.rule_id, accepted(active, table))
        table.submit(candidate)
        report = Verifier(backend).evaluate(candidate, table, [TaskCase("val", "held-out", "1")],
                                            torch.tensor([0, 1, 2]))
        self.assertEqual(report["routed_candidate"][0]["matched_rule"], "active")
        self.assertIn("candidate_not_selected", report["reasons"])

    def test_rejected_exception_has_rule_and_model_provenance(self):
        backend = FixtureBackend()
        controller = NeuralControlPlane(backend)
        rule = rule_for(backend)
        with patch("ncp.controller.synthesize", return_value=([rule], {})), \
             patch.object(controller.verifier, "evaluate", side_effect=ValueError("fixture failure")):
            report = controller.step([TaskCase("dev", "dev", "1")], [TaskCase("val", "val", "1")],
                                     torch.tensor([0, 1, 2]))
        evidence = report["trials"][0]["verification"]
        self.assertEqual(evidence["rule_fingerprint"], rule.fingerprint)
        self.assertEqual(evidence["model_identity"]["weights_sha256"], backend.identity.weights_sha256)
        self.assertEqual(evidence["message"], "fixture failure")


class SynthesisTests(unittest.TestCase):
    def test_actual_parameter_shapes_and_deterministic_candidate_ids(self):
        model = nn.Module()
        model.model = nn.Module()
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.head_dim = 1
        block.self_attn.o_proj = nn.Linear(2, 2, bias=False)
        block.mlp = nn.Module()
        block.mlp.down_proj = nn.Linear(3, 2, bias=False)
        model.model.layers = nn.ModuleList([block])
        backend = FixtureBackend()
        backend.runtime = WeightRuntime(model)
        backend.identity = ModelIdentity("fixture", "1", model_digest(model))
        backend.capture = lambda prompt, site: torch.tensor([1., 2., 3.], dtype=torch.float64)
        case = TaskCase("failed", "question", "1")
        observation = Observation("failed", "0", [0], False, [], repetition_metrics([0]), False, 0)
        first, metadata = synthesize(backend, [case], [observation], layers=[0], heads=[1])
        second, _ = synthesize(backend, [case], [observation], layers=[0], heads=[1])
        self.assertEqual(len(first), 3)
        self.assertEqual([r.fingerprint for r in first], [r.fingerprint for r in second])
        self.assertEqual(metadata["profiles"][0]["source"], "failure_centroid")
        self.assertEqual(first[1].source["instructions"][0]["slice"], [None, [1, 2]])
        with self.assertRaises(ValueError):
            synthesize(backend, [case], [observation, observation], layers=[0], heads=[1])


if __name__ == "__main__":
    unittest.main()