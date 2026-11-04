"""Offline integration with real Qwen2 architecture and tiny random weights."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from wisa import WeightRuntime, compile_program
from wisa.model import load_model


def tiny_qwen():
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=True,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            attn_implementation="eager",
        )).eval().requires_grad_(False)


class QwenIntegrationTests(unittest.TestCase):
    @torch.no_grad()
    def check_device(self, device):
        model = tiny_qwen().to(device)
        runtime = WeightRuntime(model)
        tokens = torch.tensor([[1, 4, 7, 8]], device=device)
        before = model(tokens, use_cache=False).logits.cpu().clone()
        target = "model.layers.0.mlp.down_proj.weight"
        weight = dict(model.named_parameters())[target]
        original = weight.cpu().clone()
        program = compile_program({"version": 1, "instructions": [
            {"op": "RANK1", "target": target, "left": "u", "right": "v", "alpha": 0.2},
        ]}, model, {"u": torch.arange(16), "v": torch.arange(32)})
        first = runtime.execute(program)
        after = model(tokens, use_cache=False).logits.cpu()
        self.assertTrue(torch.isfinite(after).all().item())
        self.assertGreater(torch.max(torch.abs(after - before)).item(), 1e-5)
        self.assertTrue(runtime.undo())
        self.assertTrue(torch.equal(weight.cpu(), original))
        restored = model(tokens, use_cache=False).logits.cpu()
        torch.testing.assert_close(restored, before, rtol=0, atol=0)
        self.assertEqual(runtime.execute(program), first)
        torch.testing.assert_close(model(tokens, use_cache=False).logits.cpu(), after, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        runtime.undo()
        greedy_args = {"input_ids": tokens, "max_new_tokens": 4, "do_sample": False,
                       "num_beams": 1, "pad_token_id": 0}
        self.assertTrue(torch.equal(model.generate(**greedy_args), model.generate(**greedy_args)))

    def test_cpu_qwen_edit_logits_undo_replay_and_greedy(self):
        self.check_device("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_qwen_edit_logits_undo_replay_and_greedy(self):
        self.check_device("mps")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware test")
    def test_cuda_qwen_edit_logits_undo_replay_and_greedy(self):
        self.check_device("cuda:0")

    def test_load_model_from_local_checkpoint(self):
        model = tiny_qwen()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value="tokenizer"):
                loaded, tokenizer = load_model(directory, device="cpu", local_files_only=True)
            self.assertEqual(tokenizer, "tokenizer")
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertEqual(loaded.model.embed_tokens.weight.data_ptr(), loaded.lm_head.weight.data_ptr())
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()