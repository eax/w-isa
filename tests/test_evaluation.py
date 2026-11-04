import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from wisa.evaluation import perplexity, prediction_windows


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[2., 0., -1.], [-1., 3., 0.], [1., -2., 2.]]),
                                   requires_grad=False)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.logits[input_ids])


class EvaluationTests(unittest.TestCase):
    def test_windows_score_every_token_once(self):
        for length in (2, 3, 7, 8, 13):
            for context in (1, 2, 4, 8):
                for stride in range(1, context + 1):
                    scored = []
                    for begin, end, first in prediction_windows(length, context, stride):
                        self.assertLessEqual(end - 1 - begin, context)
                        self.assertGreaterEqual(first - begin - 1, 0)
                        scored.extend(range(first, end))
                    self.assertEqual(scored, list(range(1, length)))

    def test_bigram_independent_token_weighted_reference(self):
        model = Bigram().eval()
        ids = torch.tensor([0, 1, 2, 2, 0, 1, 1, 2, 0])
        expected = sum(math.log(sum(math.exp(x) for x in model.logits[a].tolist()))
                       - model.logits[a, b].item() for a, b in zip(ids[:-1], ids[1:]))
        for context, stride in ((1, 1), (4, 2), (8, 3), (4, 4)):
            result = perplexity(model, ids, context, stride)
            self.assertEqual(result.scored_tokens, len(ids) - 1)
            self.assertAlmostEqual(result.total_nll, expected, places=5)
            self.assertAlmostEqual(result.perplexity, math.exp(expected / (len(ids) - 1)), places=5)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_protocol_rejected(self):
        for values in ((1, 4, 2), (10, 0, 1), (10, 4, 0), (10, 4, 5)):
            with self.assertRaises(ValueError):
                list(prediction_windows(*values))
        with self.assertRaises(ValueError):
            perplexity(Bigram(), torch.tensor([0, 1]))


if __name__ == "__main__":
    unittest.main()