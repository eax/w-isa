"""Head-sized parameter slices, without replacing or zeroing whole matrices."""

import unittest

import torch
from torch import nn

from wisa import CompileError, ExecutionError, WeightRuntime, compile_program


def source(*instructions):
    return {"version": 1, "instructions": list(instructions)}


def model_on(device="cpu", dtype=torch.float32):
    model = nn.Linear(6, 4, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, 25, device=device).reshape(4, 6))
    return model


class SliceTests(unittest.TestCase):
    def check_head_ablation(self, device):
        model = model_on(device)
        runtime = WeightRuntime(model)
        original = model.weight.detach().cpu().clone()
        identity = id(model.weight)
        # Three concatenated query heads, two dimensions each; remove head 1.
        program = compile_program(source({"op": "SCALE", "target": "weight",
                                          "slice": [None, [2, 4]], "factor": 0}), model)
        report = runtime.execute(program)[0]
        edited = model.weight.cpu()
        self.assertEqual(id(model.weight), identity)
        self.assertEqual(torch.count_nonzero(edited[:, 2:4]).item(), 0)
        self.assertTrue(torch.equal(edited[:, :2], original[:, :2]))
        self.assertTrue(torch.equal(edited[:, 4:], original[:, 4:]))
        self.assertEqual(report.changed_elements, 8)
        heads = torch.arange(1.0, 7.0)
        zeroed = heads.clone()
        zeroed[2:4] = 0
        torch.testing.assert_close(edited @ heads, original @ zeroed, rtol=0, atol=0)
        self.assertTrue(runtime.undo())
        self.assertTrue(torch.equal(model.weight.cpu(), original))

    def test_cpu_head_scale_changes_only_selected_columns(self):
        self.check_head_ablation("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_noncontiguous_head_slice_roundtrip(self):
        self.check_head_ablation("mps")

    def test_operand_dimensions_use_selected_shape(self):
        model = model_on()
        runtime = WeightRuntime(model)
        program = compile_program(source({"op": "ADD", "target": "weight",
                                          "slice": [[1, 3], [2, 4]], "delta": "d"}),
                                  model, {"d": [[10, 20], [30, 40]]})
        original = model.weight.clone()
        runtime.execute(program)
        expected = original.clone()
        expected[1:3, 2:4] += torch.tensor([[10, 20], [30, 40]])
        self.assertTrue(torch.equal(model.weight, expected))
        with self.assertRaises(CompileError):
            compile_program(source({"op": "ADD", "target": "weight",
                                    "slice": [None, [0, 2]], "delta": "d"}),
                            model, {"d": torch.ones(4, 6)})

    def test_projection_on_slice_preserves_other_rows_and_columns(self):
        model = model_on(dtype=torch.float64)
        runtime = WeightRuntime(model)
        original = model.weight.clone()
        program = compile_program(source({"op": "PROJECT_OUT", "target": "weight",
                                          "slice": [[1, 3], [2, 4]], "direction": "v"}),
                                  model, {"v": [1, 2]})
        runtime.execute(program)
        mask = torch.ones_like(original, dtype=torch.bool)
        mask[1:3, 2:4] = False
        self.assertTrue(torch.equal(original[mask], model.weight[mask]))
        torch.testing.assert_close(model.weight[1:3, 2:4] @ torch.tensor(
            [1.0, 2.0], dtype=torch.float64), torch.zeros(2, dtype=torch.float64))

    def test_overlapping_slices_execute_in_order_and_undo_once(self):
        model = model_on()
        runtime = WeightRuntime(model)
        original = model.weight.clone()
        runtime.execute(compile_program(source(
            {"op": "SCALE", "target": "weight", "slice": [None, [0, 4]], "factor": 2},
            {"op": "SCALE", "target": "weight", "slice": [None, [2, 6]], "factor": 3},
        ), model))
        expected = original * torch.tensor([2, 2, 6, 6, 3, 3])
        self.assertTrue(torch.equal(model.weight, expected))
        runtime.undo()
        self.assertTrue(torch.equal(model.weight, original))

    def test_failed_second_slice_restores_whole_target(self):
        model = model_on(dtype=torch.float16)
        runtime = WeightRuntime(model)
        original = model.weight.clone()
        with self.assertRaises(ExecutionError):
            runtime.execute(compile_program(source(
                {"op": "SCALE", "target": "weight", "slice": [None, [0, 2]], "factor": 0},
                {"op": "SCALE", "target": "weight", "slice": [None, [4, 6]], "factor": 1e10},
            ), model))
        self.assertTrue(torch.equal(model.weight, original))

    def test_invalid_slice_bounds_fail_before_mutation(self):
        model = model_on()
        original = model.weight.detach().clone()
        for selection in (None, [], [None], [None, [0, 7]], [None, [-1, 2]],
                          [None, [2, 2]], [None, [4, 2]], [None, [False, 2]],
                          [None, [0, 2, 1]], [None, "0:2"]):
            with self.subTest(selection=selection), self.assertRaises(CompileError):
                compile_program(source({"op": "SCALE", "target": "weight",
                                        "slice": selection, "factor": 0}), model)
            self.assertTrue(torch.equal(model.weight, original))


if __name__ == "__main__":
    unittest.main()