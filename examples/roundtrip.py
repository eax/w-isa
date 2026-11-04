"""Measure a pretrained head ablation and MLP projection on CPU, MPS or CUDA.

This is an executable experiment, not an automatically downloaded unit test.
It saves the source program, measured operand, raw logits, and a JSON report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import torch
import transformers
from transformers.utils.hub import cached_file

from wisa import WeightRuntime, compile_program
from wisa.model import load_model
import wisa.runtime as runtime_module


DEFAULT_PROMPTS = [
    "Explain in one sentence why the sky is blue.",
    "What is 17 + 25? Answer with just the number.",
    "Write a short definition of a matrix.",
    "Continue the sequence: 2, 4, 8, 16,",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def manifest(model):
    return {name: tensor_hash(parameter) for name, parameter in model.named_parameters()}


def manifest_hash(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def encode(tokenizer, prompt, device):
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    else:
        tokens = tokenizer(prompt, return_tensors="pt")
    return tokens.to(device)


@torch.no_grad()
def observe(model, tokenizer, inputs, max_new_tokens):
    rows = []
    for encoded in inputs:
        logits = model(**encoded, use_cache=False).logits[0, -1].detach().cpu().clone()
        require(torch.isfinite(logits).all().item(), "inference produced nonfinite logits")
        # Each generation call starts a fresh KV cache, including after undo.
        sequence = model.generate(
            **encoded, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )[0, encoded["input_ids"].shape[1]:].detach().cpu()
        rows.append({
            "logits": logits,
            "token_ids": sequence.tolist(),
            "text": tokenizer.decode(sequence, skip_special_tokens=True),
        })
    return rows


def identical(left, right):
    return len(left) == len(right) and all(
        torch.equal(a["logits"], b["logits"]) and a["token_ids"] == b["token_ids"]
        for a, b in zip(left, right)
    )


@torch.no_grad()
def experiment(args):
    start_time = time.perf_counter()
    model, tokenizer = load_model(
        args.model, device=args.device, dtype="float32", local_files_only=args.local_files_only,
    )
    runtime = WeightRuntime(model)
    device = next(model.parameters()).device
    require(device.type in {"cpu", "mps", "cuda"}, "unsupported model device")
    require(0 <= args.layer_index < len(model.model.layers), "layer index out of bounds")
    layer = model.model.layers[args.layer_index]
    heads = model.config.num_attention_heads
    head_dim = layer.self_attn.head_dim
    require(0 <= args.head_index < heads, "head index out of bounds")
    attention_name = f"model.layers.{args.layer_index}.self_attn.o_proj.weight"
    mlp_name = f"model.layers.{args.layer_index}.mlp.down_proj.weight"
    parameters = dict(model.named_parameters())
    attention = parameters[attention_name]
    mlp = parameters[mlp_name]
    require(attention.shape[1] == heads * head_dim, "unexpected concatenated head layout")
    first, last = args.head_index * head_dim, (args.head_index + 1) * head_dim
    original_attention = attention.detach().cpu().clone()
    original_mlp = mlp.detach().cpu().clone()
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    baseline_manifest = manifest(model)
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    inputs = [encode(tokenizer, prompt, device) for prompt in prompts]

    captured = []

    def capture_direction(_module, arguments):
        captured.append(arguments[0][0, -1].detach().cpu().double().clone())

    handle = layer.mlp.down_proj.register_forward_pre_hook(capture_direction)
    try:
        model(**inputs[0], use_cache=False)
    finally:
        handle.remove()
    require(len(captured) == 1, "expected one MLP activation capture")
    direction = captured[0]
    require(torch.linalg.vector_norm(direction).item() > 0, "captured direction is zero")
    source = {"version": 1, "instructions": [
        {"op": "SCALE", "target": attention_name, "slice": [None, [first, last]], "factor": 0.0},
        {"op": "PROJECT_OUT", "target": mlp_name, "direction": "mlp_direction"},
    ]}
    program = compile_program(source, model, {"mlp_direction": direction})

    observations = {}
    timings = {}

    def take_observation(name):
        start = time.perf_counter()
        observations[name] = observe(model, tokenizer, inputs, args.max_new_tokens)
        timings[name] = time.perf_counter() - start

    take_observation("baseline")
    take_observation("baseline_repeat")
    require(identical(observations["baseline"], observations["baseline_repeat"]),
            "baseline inference is not bitwise repeatable")

    algebra_trace = []
    real_evaluate = runtime_module._evaluate

    def execute_traced(phase):
        # Instrument the actual arithmetic boundary rather than labeling a
        # presumed path. The original evaluator still performs every edit.
        def evaluate(weight, instruction):
            result = real_evaluate(weight, instruction)
            entry = {
                "phase": phase, "op": instruction.op, "target": instruction.target,
                "shape": list(weight.shape), "compute_device": str(weight.device),
                "compute_dtype": str(weight.dtype), "result_device": str(result.device),
                "result_dtype": str(result.dtype),
                "parameter_device": str(parameters[instruction.target].device),
                "parameter_dtype": str(parameters[instruction.target].dtype),
            }
            algebra_trace.append(entry)
            require(weight.device.type == result.device.type == "cpu", "edit left CPU")
            require(weight.dtype == result.dtype == torch.float64, "edit did not use FP64")
            return result

        start = time.perf_counter()
        with patch.object(runtime_module, "_evaluate", side_effect=evaluate):
            reports = runtime.execute(program)
        timings[phase + "_edit"] = time.perf_counter() - start
        return reports

    reports = execute_traced("first")
    edited_attention = attention.detach().cpu().clone()
    edited_mlp = mlp.detach().cpu().clone()
    first_manifest = manifest(model)
    changed_names = sorted(name for name in baseline_manifest if baseline_manifest[name] != first_manifest[name])
    require(changed_names == sorted([attention_name, mlp_name]), "unexpected parameter changed")
    require(torch.count_nonzero(edited_attention[:, first:last]).item() == 0, "head was not zeroed")
    require(torch.equal(original_attention[:, :first], edited_attention[:, :first])
            and torch.equal(original_attention[:, last:], edited_attention[:, last:]),
            "an unselected attention head changed")

    # Independent NumPy reference: direct unnormalized rank-one formula.
    v = direction.numpy()
    w = original_mlp.double().numpy()
    reference = (w - np.outer(w @ v, v) / np.dot(v, v)).astype(np.float32)
    np.testing.assert_allclose(edited_mlp.numpy(), reference, rtol=1e-6, atol=1e-7)
    unit = direction / torch.linalg.vector_norm(direction)
    response_before = torch.linalg.vector_norm(original_mlp.double() @ unit).item()
    response_after = torch.linalg.vector_norm(edited_mlp.double() @ unit).item()
    relative_residual = response_after / response_before
    require(relative_residual < 1e-5, "PROJECT_OUT did not remove the measured direction")

    take_observation("edited")
    take_observation("edited_repeat")
    require(identical(observations["edited"], observations["edited_repeat"]),
            "edited inference is not bitwise repeatable")
    require(runtime.undo(), "first undo missing")
    restored_manifest = manifest(model)
    require(restored_manifest == baseline_manifest, "undo did not restore all model weights")
    require(torch.equal(attention.detach().cpu(), original_attention)
            and torch.equal(mlp.detach().cpu(), original_mlp), "edited tensors differ after undo")
    take_observation("restored")
    require(identical(observations["baseline"], observations["restored"]),
            "undo did not restore original logits/tokens")
    replay_reports = execute_traced("replay")
    replay_manifest = manifest(model)
    require(first_manifest == replay_manifest and reports == replay_reports, "weight replay differs")
    require(torch.equal(attention.detach().cpu(), edited_attention)
            and torch.equal(mlp.detach().cpu(), edited_mlp), "edited tensors differ on replay")
    take_observation("replayed")
    require(identical(observations["edited"], observations["replayed"]), "inference replay differs")
    require(runtime.undo(), "final undo missing")
    require(manifest(model) == baseline_manifest, "final undo failed")
    require(all(id(parameter) == identities[name] for name, parameter in model.named_parameters()),
            "a Parameter object was replaced")
    require(all(p.device == device and not p.requires_grad and p.grad is None
                for p in model.parameters()), "model device changed or gradients appeared")

    comparisons = []
    for index, prompt in enumerate(prompts):
        before, after = observations["baseline"][index], observations["edited"][index]
        delta = (after["logits"] - before["logits"]).double()
        comparisons.append({
            "prompt": prompt, "input_token_ids": inputs[index]["input_ids"][0].cpu().tolist(),
            "max_abs_logit_change": delta.abs().max().item(),
            "logit_l2_change": torch.linalg.vector_norm(delta).item(),
            "changed_logit_count": int(torch.count_nonzero(delta).item()),
            "generated_tokens_changed": before["token_ids"] != after["token_ids"],
            "baseline_next_token_id": int(before["logits"].argmax().item()),
            "edited_next_token_id": int(after["logits"].argmax().item()),
            "phases": {
                phase: {"text": rows[index]["text"], "token_ids": rows[index]["token_ids"],
                        "logits_sha256": tensor_hash(rows[index]["logits"])}
                for phase, rows in observations.items()
            },
        })
    require(all(row["max_abs_logit_change"] > 0 for row in comparisons), "no logit change")

    config_path = cached_file(args.model, "config.json", local_files_only=True)
    config_directory = Path(config_path).parent if config_path else None
    revision = config_directory.name if config_directory and config_directory.parent.name == "snapshots" else None
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "cached_revision": revision,
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "numpy": np.__version__},
        "model_device": str(device), "model_dtype": str(next(model.parameters()).dtype),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "layer_index_zero_based": args.layer_index, "layer_ordinal": args.layer_index + 1,
        "head_index_zero_based": args.head_index, "query_heads": heads,
        "kv_heads": model.config.num_key_value_heads, "head_dim": head_dim,
        "head_columns": [first, last], "program": source,
        "direction_source": "down_proj input at the final prompt token, baseline prompt 0",
        "direction_shape": list(direction.shape), "direction_sha256": tensor_hash(direction),
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "projection": {"response_norm_before": response_before, "response_norm_after": response_after,
                       "relative_residual": relative_residual,
                       "max_abs_numpy_reference_error": float(np.max(np.abs(edited_mlp.numpy() - reference)))},
        "algebra_trace": algebra_trace, "edit_reports": [asdict(row) for row in reports],
        "changed_parameter_names": changed_names,
        "model_weight_sha256": {"baseline": manifest_hash(baseline_manifest),
                                "edited": manifest_hash(first_manifest),
                                "restored": manifest_hash(restored_manifest),
                                "replayed": manifest_hash(replay_manifest)},
        "checks": {"all_passed": True, "baseline_repeat_bitwise": True,
                   "edited_repeat_bitwise": True, "undo_weights_logits_tokens_bitwise": True,
                   "replay_weights_logits_tokens_bitwise": True, "other_parameters_unchanged": True,
                   "other_attention_heads_unchanged": True, "head_exactly_zero": True,
                   "parameter_identity_preserved": True, "gradients_absent": True,
                   "model_stayed_on_device": True, "final_undo": True},
        "comparisons": comparisons, "timings_seconds": timings,
        "total_seconds": time.perf_counter() - start_time,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    program_path = args.output.with_suffix(".program.json")
    operands_path = args.output.with_suffix(".operands.npz")
    logits_path = args.output.with_suffix(".logits.npz")
    report["artifacts"] = {"program": str(program_path), "operands": str(operands_path),
                           "logits": str(logits_path)}
    program_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    np.savez(operands_path, mlp_direction=direction.numpy())
    np.savez_compressed(logits_path, **{
        name: torch.stack([row["logits"] for row in rows]).numpy()
        for name, rows in observations.items()
    })
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--layer-index", type=int, default=9, help="zero-based; 9 is the tenth layer")
    parser.add_argument("--head-index", type=int, default=0)
    parser.add_argument("--prompt", action="append", help="repeat for multiple prompts; first supplies direction")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/smol-roundtrip.json"))
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    report = experiment(args)
    print(json.dumps({
        "report": str(args.output), "checks": report["checks"],
        "projection": report["projection"], "timings_seconds": report["timings_seconds"],
        "comparisons": [
            {"prompt": row["prompt"], "max_abs_logit_change": row["max_abs_logit_change"],
             "generated_tokens_changed": row["generated_tokens_changed"],
             "before": row["phases"]["baseline"]["text"], "after": row["phases"]["edited"]["text"]}
            for row in report["comparisons"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()