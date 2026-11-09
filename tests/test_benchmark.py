import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from wisa import WeightRuntime, compile_program
from examples.intervention_benchmark import intervention_scope, rank_one_vectors, fixed_decode, MemorySampler, write_figures
from test_transformers import tiny_qwen


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(3, 2, dtype=torch.float64))
        self.runtime = WeightRuntime(self.model)
        self.module = self.model[0]
        self.left, self.right = rank_one_vectors(2, 3)
        self.program = compile_program({"version": 1, "instructions": [
            {"op": "RANK1", "target": "0.weight", "left": "u", "right": "v", "alpha": 0.2},
        ]}, self.model, {"u": self.left, "v": self.right})

    def scope(self, mode):
        return intervention_scope(mode, self.runtime, self.module, self.program, self.left, self.right, 0.2)

    def test_hook_and_materialized_rank_one_with_bias(self):
        inputs = torch.arange(12, dtype=torch.float64).reshape(1, 4, 3)
        original = self.module.weight.clone()
        baseline = self.model(inputs)
        expected = baseline + 0.2 * (inputs @ self.right).unsqueeze(-1) * self.left
        for mode in ("wisa", "hook"):
            with self.scope(mode):
                torch.testing.assert_close(self.model(inputs), expected, rtol=1e-12, atol=1e-12)
            self.assertTrue(torch.equal(self.module.weight, original))
        self.assertFalse(self.module._forward_hooks)

    def test_exception_restores_both_execution_modes(self):
        original = self.module.weight.clone()
        for mode in ("wisa", "hook"):
            with self.assertRaises(RuntimeError):
                with self.scope(mode):
                    raise RuntimeError("fixture")
            self.assertFalse(self.module._forward_hooks)
            self.assertTrue(torch.equal(self.module.weight, original))

    def test_fixed_decode_has_exact_length_and_repeats(self):
        model = tiny_qwen()
        encoded = {"input_ids": torch.tensor([[1, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        for count in (1, 5):
            a, b = fixed_decode(model, encoded, count), fixed_decode(model, encoded, count)
            self.assertEqual(len(a["token_ids"]), count)
            self.assertEqual(a["token_ids"], b["token_ids"])
            self.assertGreater(a["first_token_at"], 0)
            self.assertGreaterEqual(a["decode_seconds_per_token"], 0)

    def test_memory_sampler_and_figure_outputs(self):
        with MemorySampler(torch.device("cpu")) as sampler:
            tensor = torch.ones(128)
        self.assertIn("sampled_peak", sampler.result())
        summaries = []
        for length in (32, 128):
            for mode in ("baseline", "wisa", "hook"):
                summaries.append({"mode": mode, "tokens": length,
                                  "request_seconds": {"median": length / 100, "q25": length / 110, "q75": length / 90},
                                  "decode_seconds_per_token": {"median": 0.01, "q25": 0.009, "q75": 0.011}})
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "figure"
            write_figures({"summary": summaries}, stem)
            self.assertIn('<svg', stem.with_suffix('.svg').read_text())
            self.assertIn(r'\begin{groupplot}', stem.with_suffix('.tex').read_text())
        self.assertEqual(tensor.sum().item(), 128)


if __name__ == "__main__":
    unittest.main()