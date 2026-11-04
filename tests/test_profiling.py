from types import SimpleNamespace
import unittest

import torch
from torch import nn

from wisa.profiling import ActivationProfiler, ActivationSite, calibrate_threshold, classification_metrics, cosine_score
from wisa import WeightRuntime


class Tokens(dict):
    def to(self, device):
        return Tokens({key: value.to(device) for key, value in self.items()})


class Tokenizer:
    chat_template = None

    def __call__(self, prompt, **kwargs):
        ids = torch.tensor([[int(x) for x in prompt.split()]])
        return Tokens(input_ids=ids, attention_mask=torch.ones_like(ids))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor([[1., 0.], [0., 1.], [2., 3.]]))
        self.layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(2))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.layer(self.embedding(input_ids)))


class ProfilingTests(unittest.TestCase):
    def test_paired_means_and_last_pooling(self):
        profiler = ActivationProfiler(WeightRuntime(Model()), Tokenizer(), ActivationSite("layer"))
        profile = profiler.contrast(["0 2", "1 2"], ["2 0", "1 0"])
        torch.testing.assert_close(profile.direction, torch.tensor([1., 3.], dtype=torch.float64))
        self.assertEqual(profile.metadata["feature_count"], 2)
        self.assertEqual(profile.metadata["positive_count"], 2)
        self.assertFalse(profile.direction.requires_grad)
        self.assertFalse(profiler.module._forward_pre_hooks)

    def test_mean_pooling_and_output_site(self):
        profiler = ActivationProfiler(WeightRuntime(Model()), Tokenizer(), ActivationSite("layer", "output", "mean"))
        torch.testing.assert_close(profiler.capture("0 1"), torch.tensor([0.5, 0.5], dtype=torch.float64))
        self.assertFalse(profiler.module._forward_hooks)

    def test_failed_forward_removes_hook(self):
        model = Model()
        profiler = ActivationProfiler(WeightRuntime(model), Tokenizer(), ActivationSite("layer"))
        with self.assertRaises(IndexError):
            profiler.capture("9")
        self.assertFalse(profiler.module._forward_pre_hooks)

    def test_cosine_space_and_zero_handling(self):
        self.assertEqual(cosine_score(torch.tensor([1., 0.]), torch.tensor([1., 0.])), 1.0)
        self.assertEqual(cosine_score(torch.zeros(2), torch.ones(2)), 0.0)
        with self.assertRaises(ValueError):
            cosine_score(torch.ones(3), torch.ones(2))
        with self.assertRaises(ValueError):
            cosine_score(torch.ones(2), torch.zeros(2))

    def test_threshold_is_strict_and_calibrated_separately(self):
        calibration = calibrate_threshold([0.8, 0.6, 0.4, 0.2], [True, True, False, False])
        self.assertEqual(calibration["threshold"], 0.4)
        self.assertEqual(calibration["balanced_accuracy"], 1.0)
        heldout = classification_metrics([0.4, 0.5], [True, False], calibration["threshold"])
        self.assertEqual(heldout["false_positive"], 1)
        self.assertEqual(heldout["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()