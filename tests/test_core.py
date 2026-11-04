from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from wisa import CompileError, ExecutionError, WeightRuntime, compile_program, load_operands
from wisa.cli import demo
from wisa.model import select_device


def source(*instructions):
    return {"version": 1, "instructions": list(instructions)}


def matrix(dtype=torch.float64, device="cpu"):
    model = nn.Linear(2, 2, bias=False, dtype=dtype, device=device)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device))
    return model


class AlgebraTests(unittest.TestCase):
    def setUp(self):
        self.model = matrix()
        self.runtime = WeightRuntime(self.model)

    def execute(self, op, operands=None, **fields):
        program = compile_program(source({"op": op, "target": "weight", **fields}),
                                  self.model, operands)
        return self.runtime.execute(program)

    def test_add(self):
        reports = self.execute("ADD", {"d": [[1, -1], [2, -2]]}, delta="d", alpha=0.5)
        torch.testing.assert_close(self.model.weight, torch.tensor(
            [[1.5, 1.5], [4.0, 3.0]], dtype=torch.float64))
        self.assertEqual(reports[0].changed_elements, 4)
        self.assertAlmostEqual(reports[0].delta_norm, np.sqrt(2.5))

    def test_rank_one_matches_numpy(self):
        before = self.model.weight.detach().numpy().copy()
        u, v = np.array([2.0, -3.0]), np.array([0.5, 4.0])
        self.execute("RANK1", {"u": u, "v": v}, left="u", right="v", alpha=-0.25)
        np.testing.assert_array_equal(self.model.weight.numpy(), before - 0.25 * np.outer(u, v))

    def test_project_out_preserves_orthogonal_input(self):
        original = self.model.weight.detach().clone()
        self.execute("PROJECT_OUT", {"d": [1, 2]}, direction="d")
        direction = torch.tensor([1.0, 2.0], dtype=torch.float64)
        orthogonal = torch.tensor([2.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close(self.model.weight @ direction, torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(self.model.weight @ orthogonal, original @ orthogonal)
        projected = self.model.weight.detach().clone()
        self.execute("PROJECT_OUT", {"d": [1, 2]}, direction="d")
        torch.testing.assert_close(self.model.weight, projected)

    def test_write_key_independent_expected_matrix(self):
        self.execute("WRITE_KEY", {"k": [1, 2], "v": [3, -1]}, key="k", value="v")
        expected = torch.tensor([[0.6, 1.2], [0.6, -0.8]], dtype=torch.float64)
        torch.testing.assert_close(self.model.weight, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(self.model.weight.numpy() @ np.array([1, 2]), [3, -1], atol=1e-14)

    def test_write_key_leaves_orthogonal_input_unchanged(self):
        original = self.model.weight.detach().clone()
        self.execute("WRITE_KEY", {"k": [1, 2], "v": [3, -1]}, key="k", value="v")
        orthogonal = torch.tensor([2.0, -1.0], dtype=torch.float64)
        torch.testing.assert_close(self.model.weight @ orthogonal, original @ orthogonal)

    def test_scaled_directions_avoid_denominator_overflow_and_underflow(self):
        for scale in (1e-200, 1e200, -1e200):
            with self.subTest(scale=scale):
                self.execute("WRITE_KEY", {"k": [scale, 2 * scale],
                             "v": [3 * scale, -scale]}, key="k", value="v")
                torch.testing.assert_close(self.model.weight, torch.tensor(
                    [[0.6, 1.2], [0.6, -0.8]], dtype=torch.float64))
                self.runtime.undo()
                self.execute("PROJECT_OUT", {"d": [scale, 2 * scale]}, direction="d")
                torch.testing.assert_close(self.model.weight @ torch.tensor(
                    [1.0, 2.0], dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
                self.runtime.undo()

    def test_prune_strict_boundary(self):
        self.execute("PRUNE", threshold=2.0)
        torch.testing.assert_close(self.model.weight, torch.tensor(
            [[0.0, 2.0], [3.0, 4.0]], dtype=torch.float64))
        report = self.execute("PRUNE", threshold=0.0)[0]
        self.assertEqual(report.changed_elements, 0)

    def test_scale_and_add_support_bias_vectors(self):
        model = nn.Linear(2, 2, dtype=torch.float64)
        with torch.no_grad():
            model.bias.copy_(torch.tensor([2, -3]))
        runtime = WeightRuntime(model)
        program = compile_program(source(
            {"op": "SCALE", "target": "bias", "factor": -2},
            {"op": "ADD", "target": "bias", "delta": "d"},
        ), model, {"d": [1, 1]})
        runtime.execute(program)
        torch.testing.assert_close(model.bias, torch.tensor([-3.0, 7.0], dtype=torch.float64))

    def test_sequential_rounding_and_deterministic_replay(self):
        model = matrix(torch.float16)
        runtime = WeightRuntime(model)
        program = compile_program(source(
            {"op": "SCALE", "target": "weight", "factor": 0.3333},
            {"op": "SCALE", "target": "weight", "factor": 3.0},
        ), model)
        expected = (model.weight.double() * 0.3333).half().double().mul(3).half()
        first = runtime.execute(program)
        result = model.weight.detach().clone()
        self.assertTrue(torch.equal(result, expected))
        runtime.undo()
        self.assertEqual(first, runtime.execute(program))
        self.assertTrue(torch.equal(result, model.weight))

    def test_no_autograd_and_parameter_identity_preserved(self):
        identity = id(self.model.weight)
        operand = torch.ones((2, 2), requires_grad=True)
        self.execute("ADD", {"d": operand}, delta="d")
        self.assertEqual(id(self.model.weight), identity)
        self.assertFalse(self.model.weight.requires_grad)
        self.assertIsNone(self.model.weight.grad)
        self.assertIsNone(self.model.weight.grad_fn)
        self.assertIsNone(operand.grad)
        self.assertTrue(torch.is_grad_enabled())


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.model = matrix()

    def test_invalid_programs_do_not_write(self):
        original = self.model.weight.detach().clone()
        invalid = [
            None, {}, {"version": True, "instructions": []},
            {"version": 2, "instructions": []}, {"version": 1, "instructions": {}},
            source(None), source({"op": []}), source({"op": "BACKPROP", "target": "weight"}),
            source({"op": "SCALE", "target": "missing", "factor": 2}),
            source({"op": "SCALE", "target": "weight"}),
            source({"op": "SCALE", "target": "weight", "factor": True}),
            source({"op": "SCALE", "target": "weight", "factor": float("inf")}),
            source({"op": "SCALE", "target": "weight", "factor": 10**400}),
            source({"op": "SCALE", "target": "weight", "factor": 2, "typo": 1}),
            source({"op": "PRUNE", "target": "weight", "threshold": -1}),
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(CompileError):
                    compile_program(payload, self.model)
                self.assertTrue(torch.equal(self.model.weight, original))

    def test_bad_operands(self):
        for delta in ([[1, 2]], [[float("nan"), 0], [0, 0]], [[1j, 0], [0, 0]]):
            with self.subTest(delta=delta), self.assertRaises(CompileError):
                compile_program(source({"op": "ADD", "target": "weight", "delta": "d"}),
                                self.model, {"d": delta})
        for key in ([0, 0], [1, 2, 3]):
            with self.subTest(key=key), self.assertRaises(CompileError):
                compile_program(source({"op": "WRITE_KEY", "target": "weight",
                                        "key": "k", "value": "v"}),
                                self.model, {"k": key, "v": [1, 1]})
        with self.assertRaises(CompileError):
            compile_program(source({"op": "RANK1", "target": "weight", "left": "u",
                                    "right": "v"}), self.model)

    def test_operand_bank_is_copied(self):
        delta = np.ones((2, 2))
        program = compile_program(source({"op": "ADD", "target": "weight", "delta": "d"}),
                                  self.model, {"d": delta})
        delta[:] = 100
        WeightRuntime(self.model).execute(program)
        np.testing.assert_array_equal(self.model.weight.numpy(), [[2, 3], [4, 5]])

    def test_npz_numeric_roundtrip_and_pickle_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operands.npz"
            np.savez(path, key=np.array([1, 2]))
            np.testing.assert_array_equal(load_operands(path)["key"], [1, 2])
            np.savez(path, key=np.array([{"object": True}], dtype=object))
            with self.assertRaises(ValueError):
                load_operands(path)

    def test_program_binds_by_actual_shape_not_model_name(self):
        model = nn.Linear(2, 3, dtype=torch.float64)
        with self.assertRaises(CompileError):
            compile_program(source({"op": "PROJECT_OUT", "target": "bias",
                                    "direction": "d"}), model, {"d": [1, 2, 3]})
        program = compile_program(source({"op": "SCALE", "target": "weight", "factor": 2}),
                                  self.model)
        with self.assertRaises(ExecutionError):
            WeightRuntime(model).execute(program)


class TransactionTests(unittest.TestCase):
    def test_multi_target_failure_rolls_back_and_keeps_prior_undo(self):
        model = nn.Sequential(matrix(torch.float16), matrix(torch.float16))
        runtime = WeightRuntime(model)
        original = {name: p.clone() for name, p in model.named_parameters()}
        runtime.execute(compile_program(source(
            {"op": "SCALE", "target": "0.weight", "factor": 2},
        ), model))
        committed = {name: p.clone() for name, p in model.named_parameters()}
        with self.assertRaisesRegex(ExecutionError, "rolled back"):
            runtime.execute(compile_program(source(
                {"op": "SCALE", "target": "0.weight", "factor": 3},
                {"op": "SCALE", "target": "1.weight", "factor": 1e10},
            ), model))
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, committed[name]))
        self.assertTrue(runtime.undo())
        for name, p in model.named_parameters():
            self.assertTrue(torch.equal(p, original[name]))
        self.assertFalse(runtime.undo())

    def test_tied_parameters_remain_tied_and_execute_in_order(self):
        model = nn.Sequential(matrix(), matrix())
        model[1].weight = model[0].weight
        runtime = WeightRuntime(model)
        original = model[0].weight.clone()
        runtime.execute(compile_program(source(
            {"op": "SCALE", "target": "0.weight", "factor": 2},
            {"op": "SCALE", "target": "1.weight", "factor": 3},
        ), model))
        self.assertIs(model[0].weight, model[1].weight)
        self.assertTrue(torch.equal(model[0].weight, original * 6))
        runtime.undo()
        self.assertTrue(torch.equal(model[0].weight, original))

    def test_empty_program_keeps_undo_and_discard_releases_it(self):
        model = matrix()
        runtime = WeightRuntime(model)
        runtime.execute(compile_program(source({"op": "SCALE", "target": "weight",
                                                "factor": 2}), model))
        self.assertEqual(runtime.execute(compile_program(source(), model)), ())
        self.assertTrue(runtime.undo())
        runtime.discard_undo()
        self.assertFalse(runtime.undo())

    def test_nonfinite_target_is_unchanged_on_failure(self):
        model = matrix()
        with torch.no_grad():
            model.weight[0, 0] = float("inf")
        runtime = WeightRuntime(model)
        original = model.weight.clone()
        with self.assertRaises(ExecutionError):
            runtime.execute(compile_program(source({"op": "SCALE", "target": "weight",
                                                    "factor": 0}), model))
        self.assertTrue(torch.equal(model.weight, original))

    def test_undo_refuses_replaced_parameters_before_writing(self):
        model = matrix()
        runtime = WeightRuntime(model)
        runtime.execute(compile_program(source({"op": "SCALE", "target": "weight",
                                                "factor": 2}), model))
        model.weight = nn.Parameter(torch.zeros_like(model.weight), requires_grad=False)
        with self.assertRaises(ExecutionError):
            runtime.undo()
        self.assertEqual(torch.count_nonzero(model.weight).item(), 0)

    def test_demo_is_repeatable(self):
        first, second = demo(), demo()
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(first["before"], first["restored"])


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
class MPSTests(unittest.TestCase):
    def test_all_opcodes_match_cpu_weights_bitwise_and_undo(self):
        cpu, mps = matrix(torch.float32), matrix(torch.float32, "mps")
        payload = source(
            {"op": "ADD", "target": "weight", "delta": "d", "alpha": 0.5},
            {"op": "RANK1", "target": "weight", "left": "u", "right": "k"},
            {"op": "PROJECT_OUT", "target": "weight", "direction": "k"},
            {"op": "WRITE_KEY", "target": "weight", "key": "k", "value": "v"},
            {"op": "SCALE", "target": "weight", "factor": -1},
            {"op": "PRUNE", "target": "weight", "threshold": 0.5},
        )
        operands = {"d": [[1, 0], [0, 1]], "u": [2, 3], "k": [1, 2], "v": [3, -1]}
        program = compile_program(payload, cpu, operands)
        cpu_runtime, mps_runtime = WeightRuntime(cpu), WeightRuntime(mps)
        original = mps.weight.cpu().clone()
        self.assertEqual(cpu_runtime.execute(program), mps_runtime.execute(program))
        self.assertTrue(torch.equal(cpu.weight, mps.weight.cpu()))
        self.assertEqual(mps.weight.device.type, "mps")
        mps_runtime.undo()
        self.assertTrue(torch.equal(mps.weight.cpu(), original))

    def test_auto_selects_mps(self):
        self.assertEqual(select_device().type, "cuda" if torch.cuda.is_available() else "mps")

    def test_mps_operand_compiles_without_mps_float64(self):
        model = matrix(torch.float32, "mps")
        delta = torch.ones((2, 2), device="mps")
        program = compile_program(source({"op": "ADD", "target": "weight", "delta": "d"}),
                                  model, {"d": delta})
        self.assertEqual(program.instructions[0].operands[0].device.type, "cpu")
        self.assertEqual(program.instructions[0].operands[0].dtype, torch.float64)
        WeightRuntime(model).execute(program)
        np.testing.assert_array_equal(model.weight.cpu().numpy(), [[2, 3], [4, 5]])


if __name__ == "__main__":
    unittest.main()