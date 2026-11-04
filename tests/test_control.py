import asyncio
import threading
import unittest
from unittest.mock import patch

import torch
from torch import nn

from wisa import WeightRuntime, compile_program
from wisa.control import MatchActionController, Rule
from wisa.profiling import ActivationSite
import wisa.runtime as implementation
from test_profiling import Model, Tokenizer


def program(model, factor, target="layer.weight"):
    return compile_program({"version": 1, "instructions": [
        {"op": "SCALE", "target": target, "factor": factor},
    ]}, model)


class ScopeTests(unittest.TestCase):
    def test_nested_scopes_restore_entry_and_prior_undo(self):
        model = Model()
        runtime = WeightRuntime(model)
        original = model.layer.weight.clone()
        runtime.execute(program(model, 2))
        with runtime.temporary(program(model, 3)):
            self.assertTrue(torch.equal(model.layer.weight, original * 6))
            with WeightRuntime(model).temporary(program(model, 5)):
                self.assertTrue(torch.equal(model.layer.weight, original * 30))
                runtime.execute(program(model, 7, "embedding.weight"))
            self.assertTrue(torch.equal(model.layer.weight, original * 6))
        self.assertTrue(torch.equal(model.layer.weight, original * 2))
        self.assertTrue(runtime.undo())
        self.assertTrue(torch.equal(model.layer.weight, original))

    def test_undo_inside_empty_scope_is_also_restored(self):
        model = Model()
        runtime = WeightRuntime(model)
        original = model.layer.weight.clone()
        runtime.execute(program(model, 2))
        with runtime.temporary():
            runtime.undo()
            self.assertTrue(torch.equal(model.layer.weight, original))
        self.assertTrue(torch.equal(model.layer.weight, original * 2))
        self.assertTrue(runtime.undo())
        self.assertTrue(torch.equal(model.layer.weight, original))

    def test_inference_errors_and_cancellation_restore(self):
        for error in (RuntimeError, KeyboardInterrupt, asyncio.CancelledError):
            with self.subTest(error=error):
                model = Model()
                runtime = WeightRuntime(model)
                original = model.layer.weight.clone()
                with self.assertRaises(error):
                    with runtime.temporary(program(model, 0)):
                        raise error("synthetic interruption")
                self.assertTrue(torch.equal(model.layer.weight, original))
                self.assertFalse(runtime.undo())

    def test_cancellation_halfway_through_execute_rolls_back(self):
        for error in (KeyboardInterrupt, asyncio.CancelledError):
            model = Model()
            runtime = WeightRuntime(model)
            original = model.layer.weight.clone()
            compiled = compile_program({"version": 1, "instructions": [
                {"op": "SCALE", "target": "layer.weight", "factor": 2},
                {"op": "SCALE", "target": "layer.weight", "factor": 3},
            ]}, model)
            evaluate = implementation._evaluate
            calls = []

            def interrupted(weight, instruction):
                calls.append(1)
                if len(calls) == 2:
                    raise error("cancel")
                return evaluate(weight, instruction)

            with patch.object(implementation, "_evaluate", side_effect=interrupted):
                with self.assertRaises(error):
                    runtime.execute(compiled)
            self.assertTrue(torch.equal(model.layer.weight, original))

    def test_two_runtime_instances_share_request_lock(self):
        model = Model()
        first, second = WeightRuntime(model), WeightRuntime(model)
        original = model.layer.weight.clone()
        started, finished = threading.Event(), threading.Event()
        seen = []

        def observer():
            started.set()
            with second.temporary():
                seen.append(model.layer.weight.clone())
            finished.set()

        with first.temporary(program(model, 0)):
            thread = threading.Thread(target=observer)
            thread.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(finished.wait(0.05))
        thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertTrue(torch.equal(seen[0], original))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.runtime = WeightRuntime(self.model)
        self.rule = Rule("first", ActivationSite("layer"), torch.tensor([1., 0.]), 0.5,
                         program(self.model, 0))
        self.controller = MatchActionController(self.runtime, Tokenizer(), [self.rule])

    def test_match_and_no_match_do_not_leak_weights(self):
        original = self.model.layer.weight.clone()
        matched = self.controller.run("0", lambda: self.model.layer.weight.clone())
        unmatched = self.controller.run("1", lambda: self.model.layer.weight.clone())
        self.assertEqual(matched["decision"].matched_rule, "first")
        self.assertEqual(torch.count_nonzero(matched["value"]).item(), 0)
        self.assertIsNone(unmatched["decision"].matched_rule)
        self.assertTrue(torch.equal(unmatched["value"], original))
        self.assertTrue(torch.equal(self.model.layer.weight, original))

    def test_first_matching_rule_wins(self):
        other = Rule("second", self.rule.site, self.rule.direction, 0.5, program(self.model, 3))
        controller = MatchActionController(self.runtime, Tokenizer(), [self.rule, other])
        result = controller.run("0", lambda: self.model.layer.weight.clone())
        self.assertEqual(result["decision"].matched_rule, "first")
        self.assertEqual(len(result["decision"].scores), 2)
        self.assertEqual(torch.count_nonzero(result["value"]).item(), 0)

    def test_inference_exception_and_stream_close_restore(self):
        original = self.model.layer.weight.clone()

        def broken():
            raise RuntimeError("generation failed")

        with self.assertRaises(RuntimeError):
            self.controller.run("0", broken)
        self.assertTrue(torch.equal(self.model.layer.weight, original))
        stream = self.controller.stream("0", lambda: iter(["one", "two"]))
        self.assertEqual(next(stream), "one")
        self.assertEqual(torch.count_nonzero(self.model.layer.weight).item(), 0)
        stream.close()
        self.assertTrue(torch.equal(self.model.layer.weight, original))

    def test_invalid_rule_and_trigger_failure_never_mutate(self):
        original = self.model.layer.weight.clone()
        with self.assertRaises(ValueError):
            MatchActionController(self.runtime, Tokenizer(), [self.rule, self.rule])
        bad = Rule("bad", self.rule.site, torch.ones(3), 0.0, self.rule.program)
        with self.assertRaises(ValueError):
            MatchActionController(self.runtime, Tokenizer(), [bad]).run("0", lambda: None)
        self.assertTrue(torch.equal(self.model.layer.weight, original))

    def test_stream_closes_underlying_iterator_once(self):
        closed = []

        class Source:
            def __iter__(self):
                return self

            def __next__(self):
                return "token"

            def close(self):
                closed.append(True)

        stream = self.controller.stream("0", Source)
        self.assertEqual(next(stream), "token")
        stream.close()
        self.assertEqual(len(closed), 1)

    def test_registration_copies_program_operands(self):
        compiled = compile_program({"version": 1, "instructions": [
            {"op": "ADD", "target": "layer.weight", "delta": "d"},
        ]}, self.model, {"d": torch.ones(2, 2)})
        rule = Rule("copy", self.rule.site, self.rule.direction, 0.5, compiled)
        controller = MatchActionController(self.runtime, Tokenizer(), [rule])
        compiled.instructions[0].operands[0].fill_(99)
        original = self.model.layer.weight.clone()
        result = controller.run("0", lambda: self.model.layer.weight.clone())
        self.assertTrue(torch.equal(result["value"], original + 1))
        self.assertTrue(torch.equal(self.model.layer.weight, original))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_real_mps_request_scope_and_cancellation(self):
        model = Model().to("mps")
        runtime = WeightRuntime(model)
        rule = Rule("mps", ActivationSite("layer"), torch.tensor([1., 0.]), 0.5, program(model, 0))
        controller = MatchActionController(runtime, Tokenizer(), [rule])
        original = model.layer.weight.detach().cpu().clone()
        result = controller.run("0", lambda: model.layer.weight.detach().cpu().clone())
        self.assertEqual(torch.count_nonzero(result["value"]).item(), 0)
        self.assertTrue(torch.equal(model.layer.weight.detach().cpu(), original))
        with self.assertRaises(asyncio.CancelledError):
            with controller.session("0"):
                raise asyncio.CancelledError()
        self.assertTrue(torch.equal(model.layer.weight.detach().cpu(), original))


if __name__ == "__main__":
    unittest.main()