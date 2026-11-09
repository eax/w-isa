"""Local PyTorch observation backend and W-ISA flow execution."""

from __future__ import annotations

import hashlib
import time

import torch

from wisa import WeightRuntime, compile_program
from wisa.evaluation import perplexity
from wisa.model import encode_prompt
from wisa.profiling import ActivationProfiler, ActivationSite, cosine_score

from .schema import ModelIdentity, NeuralRule
from .telemetry import Observation, TaskCase, repetition_metrics


@torch.no_grad()
def model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TorchBackend:
    def __init__(self, model, tokenizer, model_id: str, revision: str,
                 max_new_tokens: int = 128, ngram_size: int = 4):
        if max_new_tokens <= 0 or ngram_size <= 0:
            raise ValueError("token budget and ngram size must be positive")
        self.runtime = WeightRuntime(model)
        self.tokenizer = tokenizer
        self.max_new_tokens, self.ngram_size = max_new_tokens, ngram_size
        self.identity = ModelIdentity(model_id, revision, model_digest(model))

    def capture(self, prompt: str, site: ActivationSite) -> torch.Tensor:
        return ActivationProfiler(self.runtime, self.tokenizer, site).capture(prompt)

    def compile(self, rule: NeuralRule):
        if rule.identity != self.identity:
            raise ValueError("rule model identity mismatch")
        return compile_program(rule.source, self.runtime.model, rule.operands)

    @torch.no_grad()
    def observe(self, case: TaskCase, token_budget: int | None = None) -> Observation:
        budget = min(self.max_new_tokens, token_budget) if token_budget is not None else self.max_new_tokens
        start = time.perf_counter()
        model = self.runtime.model
        with self.runtime.lock:
            inputs = encode_prompt(self.tokenizer, case.prompt, next(model.parameters()).device)
            output = model.generate(**inputs, max_new_tokens=budget, do_sample=False, num_beams=1,
                                    return_dict_in_generate=True, output_scores=True,
                                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                    else self.tokenizer.eos_token_id)
            tokens = output.sequences[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            entropies = []
            for scores in output.scores:
                log_probs = torch.log_softmax(scores[0].detach().cpu().double(), dim=-1)
                probs = log_probs.exp()
                entropy = -torch.where(probs > 0, probs * log_probs, 0).sum()
                if not torch.isfinite(entropy).item():
                    raise ValueError("nonfinite output entropy")
                entropies.append(entropy.item())
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
            hit_budget = len(tokens) >= budget and (not tokens or tokens[-1] not in eos_ids)
            body = tokens[:-1] if tokens and tokens[-1] in eos_ids else tokens
            return Observation(case.case_id, text, tokens, case.score(text), entropies,
                               repetition_metrics(body, self.ngram_size), hit_budget, time.perf_counter() - start)

    def score_corpus(self, token_ids: torch.Tensor, context_length=512, stride=256):
        with self.runtime.lock:
            return perplexity(self.runtime.model, token_ids, context_length, stride).to_dict()


def route_observe(backend, case: TaskCase, rules: tuple[NeuralRule, ...]) -> Observation:
    """Priority order is supplied by the table; capture before editing."""
    with backend.runtime.lock:
        captures = {}
        scores = {}
        selected = None
        for rule in rules:
            if rule.identity != backend.identity:
                raise ValueError("flow model identity mismatch")
            if rule.site not in captures:
                captures[rule.site] = backend.capture(case.prompt, rule.site)
            score = cosine_score(captures[rule.site], rule.direction)
            scores[rule.rule_id] = score
            if selected is None and score > rule.threshold:
                selected = rule
        program = backend.compile(selected) if selected else None
        with backend.runtime.temporary(program):
            observation = backend.observe(case, selected.token_budget if selected else None)
        observation.matched_rule = selected.rule_id if selected else None
        observation.match_scores = scores
        return observation