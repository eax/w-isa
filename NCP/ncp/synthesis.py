"""Deterministic candidates derived from observed development failures."""

from __future__ import annotations

from dataclasses import asdict

import torch
from wisa.profiling import ActivationSite

from .schema import NeuralRule, canonical_hash


def synthesize(backend, cases, observations, layers=(9,), heads=(0,), factors=(0.0,),
               threshold=0.8, priority=0, hard_timeout_seconds=None, idle_timeout_seconds=None,
               token_budget=None):
    labels = {row.case_id: row.correct for row in observations}
    if (len(labels) != len(observations) or len(observations) != len(cases)
            or set(labels) != {case.case_id for case in cases}):
        raise ValueError("development cases and observations differ")
    failed = [case for case in cases if not labels[case.case_id]]
    successful = [case for case in cases if labels[case.case_id]]
    if not failed:
        return [], {"failure_cases": 0, "success_cases": len(successful), "profiles": []}
    candidates, profiles = [], []
    model = backend.runtime.model
    with backend.runtime.lock:
        for layer_index in layers:
            if not 0 <= layer_index < len(model.model.layers):
                raise ValueError("layer index is outside the model")
            block = model.model.layers[layer_index]
            site = ActivationSite(f"model.layers.{layer_index}.mlp.down_proj")
            failures = torch.stack([backend.capture(case.prompt, site) for case in failed])
            direction = failures.mean(dim=0)
            kind = "failure_centroid"
            if successful:
                successes = torch.stack([backend.capture(case.prompt, site) for case in successful])
                direction = direction - successes.mean(dim=0)
                kind = "failure_minus_success"
            norm = torch.linalg.vector_norm(direction).item()
            profiles.append({"site": asdict(site), "source": kind, "norm": norm,
                             "failure_case_ids": [case.case_id for case in failed],
                             "success_case_ids": [case.case_id for case in successful]})
            if norm == 0:
                continue
            projection = {"op": "PROJECT_OUT", "target": site.module + ".weight", "direction": "v_error"}

            def candidate(instructions, label):
                source = {"version": 1, "instructions": instructions}
                uses_direction = any(row["op"] == "PROJECT_OUT" for row in instructions)
                operands = {"v_error": direction.clone()} if uses_direction else {}
                payload = {"source": source, "direction": direction.tolist(), "threshold": threshold,
                           "model": backend.identity.__dict__, "priority": priority,
                           "hard_timeout_seconds": hard_timeout_seconds, "idle_timeout_seconds": idle_timeout_seconds}
                if token_budget is not None:
                    payload["token_budget"] = token_budget
                rule = NeuralRule("rule-" + canonical_hash(payload)[:20], backend.identity, site,
                                  direction.clone(), threshold, source, operands, priority,
                                  hard_timeout_seconds, idle_timeout_seconds,
                                  token_budget=token_budget, hypothesis=label)
                backend.compile(rule)
                candidates.append(rule)

            candidate([projection], f"Project the {kind} input direction out of block {layer_index}'s MLP")
            head_dim = block.self_attn.head_dim
            output = block.self_attn.o_proj.weight
            for head in heads:
                if head < 0 or (head + 1) * head_dim > output.shape[1]:
                    raise ValueError(f"head {head} is outside block {layer_index}'s output projection")
                for factor in factors:
                    scale = {"op": "SCALE", "target": f"model.layers.{layer_index}.self_attn.o_proj.weight",
                             "slice": [None, [head * head_dim, (head + 1) * head_dim]], "factor": factor}
                    candidate([scale], f"Scale block {layer_index} head {head} output by {factor}")
                    candidate([scale, projection], f"Scale block {layer_index} head {head} and project its MLP")
    return candidates, {"failure_cases": len(failed), "success_cases": len(successful), "profiles": profiles}