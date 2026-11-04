"""Extract a formal-minus-casual direction and evaluate it on disjoint topics."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils.hub import cached_file

from wisa import WeightRuntime, compile_program
from wisa.evaluation import perplexity
from wisa.model import encode_prompt, generate, load_model
from wisa.profiling import ActivationProfiler, ActivationSite, calibrate_threshold, classification_metrics, cosine_score


@torch.no_grad()
def completion_logprob(model, tokenizer, prompt, completion):
    device = next(model.parameters()).device
    prefix = encode_prompt(tokenizer, prompt, device)["input_ids"]
    suffix = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    ids = torch.cat([prefix, suffix], dim=1)
    logits = model(input_ids=ids, use_cache=False).logits[0, prefix.shape[1] - 1:-1]
    return -F.cross_entropy(logits.float(), suffix[0], reduction="mean").item()


def prompts(data, split):
    return [(data[prefix] + row["question"], label, row)
            for row in data[split]
            for prefix, label in (("positive_prefix", True), ("negative_prefix", False))]


def run(args):
    fixture_bytes = args.pairs.read_bytes()
    data = json.loads(fixture_bytes)
    sets = [{row["question"] for row in data[split]} for split in ("profile", "calibration", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("profile/calibration/test topics must be disjoint")
    model, tokenizer = load_model(args.model, device=args.device, local_files_only=True)
    runtime = WeightRuntime(model)
    site = ActivationSite(args.module, "input", args.pooling)
    profiler = ActivationProfiler(runtime, tokenizer, site)
    config = Path(cached_file(args.model, "config.json", local_files_only=True))
    revision = config.parent.name if config.parent.parent.name == "snapshots" else None
    train = prompts(data, "profile")
    profile = profiler.contrast([p for p, label, _ in train if label],
                                [p for p, label, _ in train if not label], revision)
    profile.save(args.output)
    source = {"version": 1, "instructions": [
        {"op": "PROJECT_OUT", "target": args.module + ".weight", "direction": "v_diff"},
    ]}
    program = compile_program(source, model, profile.operands())
    args.output.with_suffix(".program.json").write_text(json.dumps(source, indent=2) + "\n")
    calibration = prompts(data, "calibration")
    scores = [cosine_score(profiler.capture(p), profile.direction) for p, _, _ in calibration]
    calibrated = calibrate_threshold(scores, [label for _, label, _ in calibration])
    testing = prompts(data, "test")
    test_scores = [cosine_score(profiler.capture(p), profile.direction) for p, _, _ in testing]
    heldout = classification_metrics(test_scores, [label for _, label, _ in testing], calibrated["threshold"])

    def evaluate():
        rows = []
        for prompt, label, row in testing:
            formal = completion_logprob(model, tokenizer, prompt, row["formal"])
            casual = completion_logprob(model, tokenizer, prompt, row["casual"])
            rows.append({"prompt": prompt, "formal_instruction": label,
                         "formal_completion_mean_logprob": formal,
                         "casual_completion_mean_logprob": casual,
                         "formal_preference_margin": formal - casual,
                         "generation": generate(model, tokenizer, prompt, args.max_new_tokens)})
        return rows

    before = evaluate()
    corpus_ids = None
    corpus_metadata = None
    baseline_ppl = None
    if args.corpus:
        corpus = json.loads(args.corpus.read_text())
        text = corpus["separator"].join(corpus["rows"])
        if hashlib.sha256(text.encode()).hexdigest() != corpus["text_sha256"]:
            raise ValueError("corpus checksum mismatch")
        corpus_metadata = {key: value for key, value in corpus.items() if key != "rows"}
        corpus_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)["input_ids"][0]
        baseline_ppl = perplexity(model, corpus_ids).to_dict()
    parameter = dict(model.named_parameters())[args.module + ".weight"]
    original = parameter.detach().cpu().clone()
    with runtime.temporary(program) as reports:
        after = evaluate()
        projected_ppl = perplexity(model, corpus_ids).to_dict() if corpus_ids is not None else None
    if not torch.equal(parameter.detach().cpu(), original):
        raise AssertionError("profile experiment failed to restore weights")
    replay = evaluate()
    if before != replay:
        raise AssertionError("profile experiment failed to restore output scores/text")
    margins_before = np.array([row["formal_preference_margin"] for row in before])
    margins_after = np.array([row["formal_preference_margin"] for row in after])
    formal_mask = np.array([label for _, label, _ in testing])
    report = {"model": args.model, "revision": revision, "site": asdict(site),
              "profile": profile.metadata, "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
              "fixture": str(args.pairs), "calibration": calibrated, "heldout_trigger": heldout,
              "calibration_scores": [{"prompt": p, "positive": label, "score": score}
                                     for (p, label, _), score in zip(calibration, scores)],
              "heldout_scores": [{"prompt": p, "positive": label, "score": score}
                                  for (p, label, _), score in zip(testing, test_scores)],
              "baseline": before, "projected": after, "restored": replay,
              "style_metric": "mean-token teacher-forced logp(formal completion) minus logp(casual completion); diagnostic, not a style classifier",
              "metrics": {"mean_formal_margin_before": float(margins_before.mean()),
                          "mean_formal_margin_after": float(margins_after.mean()),
                          "instruction_sensitivity_before": float(margins_before[formal_mask].mean() - margins_before[~formal_mask].mean()),
                          "instruction_sensitivity_after": float(margins_after[formal_mask].mean() - margins_after[~formal_mask].mean()),
                          "changed_generations": sum(a["generation"] != b["generation"] for a, b in zip(before, after))},
              "max_new_tokens": args.max_new_tokens, "undo_weights_outputs_exact": True,
              "edit_reports": [asdict(row) for row in reports], "corpus": corpus_metadata,
              "baseline_perplexity": baseline_ppl, "projected_perplexity": projected_ppl}
    if baseline_ppl and projected_ppl:
        report["metrics"]["ppl_change_percent"] = 100 * (projected_ppl["perplexity"] / baseline_ppl["perplexity"] - 1)
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"profile": profile.metadata, "calibration": calibrated, "heldout": heldout,
                      "metrics": report["metrics"], "undo_exact": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--pairs", type=Path, default=Path("examples/style_pairs.json"))
    parser.add_argument("--module", default="model.layers.9.mlp.down_proj")
    parser.add_argument("--pooling", default="last", choices=("last", "mean"))
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/style-profile"))
    run(parser.parse_args())