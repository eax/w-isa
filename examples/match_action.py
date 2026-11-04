"""Exercise a calibrated rule with real local inference, failure and concurrency."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers.utils.hub import cached_file

from wisa import MatchActionController, Rule, WeightRuntime, compile_program, load_operands
from wisa.model import encode_prompt, load_model
from wisa.profiling import ActivationSite
from roundtrip import identical, manifest, manifest_hash, observe, tensor_hash


def run(args):
    profile = json.loads(args.profile.with_suffix(".report.json").read_text())
    metadata = json.loads(args.profile.with_suffix(".json").read_text())
    operands = load_operands(args.profile.with_suffix(".npz"))
    if hashlib.sha256(operands["v_diff"].tobytes()).hexdigest() != metadata["direction_sha256"]:
        raise ValueError("profile direction checksum mismatch")
    model, tokenizer = load_model(profile["model"], device=args.device, local_files_only=True)
    config = Path(cached_file(profile["model"], "config.json", local_files_only=True))
    revision = config.parent.name if config.parent.parent.name == "snapshots" else None
    if revision != profile["revision"]:
        raise ValueError("model revision differs from calibrated profile")
    runtime = WeightRuntime(model)
    source = json.loads(args.profile.with_suffix(".program.json").read_text())
    program = compile_program(source, model, operands)
    site = ActivationSite(**profile["site"])
    threshold = profile["calibration"]["threshold"]
    rule = Rule("formal-difference-projection", site, torch.from_numpy(operands["v_diff"]), threshold, program)
    controller = MatchActionController(runtime, tokenizer, [rule])
    original_manifest = manifest(model)
    target = dict(model.named_parameters())[source["instructions"][0]["target"]]
    original = target.detach().cpu().clone()
    rows = []
    cached = {}

    def inference(prompt):
        encoded = encode_prompt(tokenizer, prompt, next(model.parameters()).device)
        return observe(model, tokenizer, [encoded], args.max_new_tokens)

    for item in profile["heldout_scores"]:
        prompt = item["prompt"]
        start = time.perf_counter()
        baseline = inference(prompt)
        baseline_seconds = time.perf_counter() - start
        result = controller.run(prompt, lambda: inference(prompt))
        if not torch.equal(target.detach().cpu(), original):
            raise AssertionError("request leaked edited weights")
        repeated = controller.run(prompt, lambda: inference(prompt))
        restored = inference(prompt)
        if not identical(result["value"], repeated["value"]) or not identical(baseline, restored):
            raise AssertionError("request repeat or restored inference differs")
        decision = result["decision"]
        reference_score_delta = decision.scores[0][1] - item["score"]
        if decision.matched_rule is None and not identical(baseline, result["value"]):
            raise AssertionError("unmatched request output changed")
        cached[prompt] = result["value"]
        before, after = baseline[0], result["value"][0]
        rows.append({"prompt": prompt, "positive_label": item["positive"],
                     "profile_reference_score": item["score"],
                     "profile_reference_score_delta": reference_score_delta,
                     "decision": asdict(decision), "baseline_seconds": baseline_seconds,
                     "request_seconds": result["total_seconds"],
                     "inference_seconds": result["inference_seconds"],
                     "control_seconds": result["control_seconds"],
                     "repeat_request_seconds": repeated["total_seconds"],
                     "max_abs_logit_change": (after["logits"] - before["logits"]).abs().max().item(),
                     "baseline": {"text": before["text"], "token_ids": before["token_ids"],
                                  "logit_sha256": tensor_hash(before["logits"])},
                     "conditional": {"text": after["text"], "token_ids": after["token_ids"],
                                     "logit_sha256": tensor_hash(after["logits"])},
                     "repeat_exact": True, "undo_weights_logits_tokens_exact": True})
    matched_prompts = [row["prompt"] for row in rows if row["decision"]["matched_rule"] is not None]
    unmatched_prompts = [row["prompt"] for row in rows if row["decision"]["matched_rule"] is None]
    if not matched_prompts or not unmatched_prompts:
        raise AssertionError("this fixture must exercise both match and no-match paths")
    matched_prompt, unmatched_prompt = matched_prompts[0], unmatched_prompts[0]

    def fail(error):
        if torch.equal(target.detach().cpu(), original):
            raise AssertionError("failure injection did not occur inside an applied patch")
        raise error("injected request failure")

    for error in (RuntimeError, asyncio.CancelledError):
        try:
            controller.run(matched_prompt, lambda: fail(error))
        except error:
            pass
        else:
            raise AssertionError("injected failure was not propagated")
        if not torch.equal(target.detach().cpu(), original):
            raise AssertionError("failure or cancellation leaked weights")
    stream = controller.stream(matched_prompt, lambda: iter(["probe", "unused"]))
    next(stream)
    if torch.equal(target.detach().cpu(), original):
        raise AssertionError("stream probe did not observe applied patch")
    stream.close()
    if not torch.equal(target.detach().cpu(), original):
        raise AssertionError("stream close leaked weights")
    second_controller = MatchActionController(WeightRuntime(model), tokenizer, [rule])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(controller.run, matched_prompt, lambda: inference(matched_prompt))
        second = pool.submit(second_controller.run, unmatched_prompt, lambda: inference(unmatched_prompt))
        if not identical(first.result()["value"], cached[matched_prompt]):
            raise AssertionError("concurrent matched output differs")
        if not identical(second.result()["value"], cached[unmatched_prompt]):
            raise AssertionError("concurrent unmatched output differs")
    final_manifest = manifest(model)
    if final_manifest != original_manifest:
        raise AssertionError("final model weights differ")
    report = {"model": profile["model"], "revision": revision, "device": str(next(model.parameters()).device),
              "policy": "first matching rule wins; scores measured before any edit",
              "rule": {"name": rule.name, "site": asdict(site), "threshold": threshold, "program": source,
                       "profile": str(args.profile)},
              "requests": rows, "max_new_tokens": args.max_new_tokens,
              "checks": {"repeat_exact": True, "unmatched_unchanged": True, "undo_exact": True,
                         "generation_failure_restored": True, "cancellation_restored": True,
                         "synthetic_stream_close_restored": True, "concurrent_requests_isolated": True,
                         "full_model_restored": True},
              "model_hash_before": manifest_hash(original_manifest), "model_hash_after": manifest_hash(final_manifest)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": report["checks"], "matched": len(matched_prompts), "unmatched": len(unmatched_prompts),
                      "changed_generations": sum(row["baseline"]["token_ids"] != row["conditional"]["token_ids"] for row in rows),
                      "mean_profile_ms": 1000 * sum(row["decision"]["profile_seconds"] for row in rows) / len(rows),
                      "mean_apply_ms": 1000 * sum(row["decision"]["apply_seconds"] for row in rows if row["decision"]["matched_rule"]) / len(matched_prompts),
                      "mean_control_ms": 1000 * sum(row["control_seconds"] for row in rows) / len(rows),
                      "mean_request_ms": 1000 * sum(row["request_seconds"] for row in rows) / len(rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("artifacts/style-profile"))
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, default=Path("artifacts/match-action.json"))
    run(parser.parse_args())