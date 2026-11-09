"""Run the first local Neural Control Plane optimization loop."""

from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers.utils.hub import cached_file

from wisa.model import load_model

from .backend import TorchBackend, model_digest
from .controller import NeuralControlPlane
from .flows import FlowTable
from .telemetry import TaskCase, summarize
from .verifier import GatePolicy


ROOT = Path(__file__).resolve().parents[2]


def load_cases(path):
    raw = json.loads(path.read_text())
    groups = {name: [TaskCase(**case) for case in raw[name]] for name in ("development", "validation", "test")}
    ids, prompts = set(), set()
    for cases in groups.values():
        if not cases:
            raise ValueError("each data split must contain cases")
        for case in cases:
            if case.case_id in ids or case.prompt in prompts:
                raise ValueError("case IDs and prompts must be disjoint across splits")
            ids.add(case.case_id)
            prompts.add(case.prompt)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--cases", type=Path, default=ROOT / "NCP/examples/reasoning_cases.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/wikitext2-test.json")
    parser.add_argument("--output", type=Path, default=ROOT / "local/ncp/run.json")
    parser.add_argument("--resume", type=Path, help="saved flow table for the identical base model")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[9])
    parser.add_argument("--heads", type=int, nargs="+", default=[0])
    parser.add_argument("--head-factors", type=float, nargs="+", default=[0.0])
    parser.add_argument("--match-threshold", type=float, default=0.8)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--hard-timeout", type=float)
    parser.add_argument("--idle-timeout", type=float)
    parser.add_argument("--rule-token-budget", type=int, help="generation cap for a matched rule")
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-ppl-increase-percent", type=float, default=0.5)
    parser.add_argument("--min-corrected-cases", type=int, default=1)
    parser.add_argument("--max-regressed-cases", type=int, default=0)
    parser.add_argument("--min-repeat-reduction", type=float, default=0.0)
    parser.add_argument("--max-relative-spectral-delta", type=float)
    args = parser.parse_args()
    if args.cycles <= 0 or args.ppl_tokens < 2:
        parser.error("cycles must be positive and ppl-tokens must be at least two")
    started = time.perf_counter()
    cases = load_cases(args.cases)
    corpus = json.loads(args.corpus.read_text())
    text = corpus["separator"].join(corpus["rows"])
    if hashlib.sha256(text.encode()).hexdigest() != corpus["text_sha256"]:
        raise ValueError("corpus checksum mismatch")
    model, tokenizer = load_model(args.model, args.device, local_files_only=True)
    config = Path(cached_file(args.model, "config.json", local_files_only=True))
    revision = config.parent.name if config.parent.parent.name == "snapshots" else "local"
    backend = TorchBackend(model, tokenizer, args.model, revision, args.max_new_tokens)
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)["input_ids"][0]
    if len(tokens) < 2 * args.ppl_tokens:
        raise ValueError("corpus must contain disjoint verifier and final-evaluation token slices")
    verifier_tokens = tokens[:args.ppl_tokens].clone()
    final_tokens = tokens[args.ppl_tokens:2 * args.ppl_tokens].clone()
    table = FlowTable.load(args.resume, backend.identity) if args.resume else FlowTable(backend.identity)
    policy = GatePolicy(args.max_ppl_increase_percent, args.min_corrected_cases, args.max_regressed_cases,
                        args.min_repeat_reduction, args.max_relative_spectral_delta)
    controller = NeuralControlPlane(backend, table, policy, args.context_length, args.stride)
    report = {"identity": asdict(backend.identity), "device": str(next(model.parameters()).device),
              "torch": torch.__version__, "policy": asdict(policy), "max_new_tokens": args.max_new_tokens,
              "task_scoring": "Exact expected-answer comparison using the final answer-pattern capture, or the whole response when absent; case-insensitive and trailing period normalized",
              "fixture_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              "cases": {name: [asdict(case) for case in group] for name, group in cases.items()},
              "corpus": {k: v for k, v in corpus.items() if k != "rows"},
              "verifier_token_range": [0, args.ppl_tokens],
              "final_token_range": [args.ppl_tokens, 2 * args.ppl_tokens],
              "verifier_token_sha256": hashlib.sha256(verifier_tokens.numpy().tobytes()).hexdigest(),
              "final_token_sha256": hashlib.sha256(final_tokens.numpy().tobytes()).hexdigest(),
              "context_length": args.context_length, "stride": args.stride,
              "synthesis": {"layers": args.layers, "heads": args.heads, "factors": args.head_factors,
                            "threshold": args.match_threshold, "priority": args.priority,
                            "hard_timeout_seconds": args.hard_timeout,
                            "idle_timeout_seconds": args.idle_timeout,
                            "rule_token_budget": args.rule_token_budget}, "cycles": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_path = args.output.with_suffix(".flows.json")
    for index in range(args.cycles):
        result = controller.step(cases["development"], cases["validation"], verifier_tokens,
                                 layers=args.layers, heads=args.heads, factors=args.head_factors,
                                 threshold=args.match_threshold, priority=args.priority,
                                 hard_timeout_seconds=args.hard_timeout, idle_timeout_seconds=args.idle_timeout,
                                 token_budget=args.rule_token_budget)
        report["cycles"].append(result)
        controller.table.save(table_path)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for trial in result["trials"]:
            verification = trial.get("verification", {})
            print(json.dumps({"cycle": index + 1, "rule": trial["rule_id"], "action": trial["action"],
                              "reasons": verification.get("reasons", []),
                              "corrected_cases": verification.get("corrected_cases"),
                              "ppl_change_percent": verification.get("ppl_change_percent")}), flush=True)
        if not any("verification" in trial for trial in result["trials"]):
            break
    # Final data is first evaluated after candidate selection has finished.
    before = [backend.observe(case) for case in cases["test"]]
    after = [controller.infer(case) for case in cases["test"]]
    final_ppl = backend.score_corpus(final_tokens, args.context_length, args.stride)
    _, active = controller.table.snapshot()
    active_ppl = {}
    for rule in active:
        with backend.runtime.temporary(backend.compile(rule)):
            active_ppl[rule.rule_id] = backend.score_corpus(final_tokens, args.context_length, args.stride)
    report["final_evaluation"] = {"baseline": [o.to_dict() for o in before],
                                  "routed": [o.to_dict() for o in after],
                                  "baseline_summary": summarize(before), "routed_summary": summarize(after),
                                  "baseline_perplexity": final_ppl, "forced_rule_perplexity": active_ppl}
    report["table"] = controller.table.export()
    report["model_restored_exact"] = model_digest(model) == backend.identity.weights_sha256
    if not report["model_restored_exact"]:
        raise RuntimeError("base model changed")
    report["seconds"] = time.perf_counter() - started
    controller.table.save(table_path)
    report["flow_table_file"] = str(table_path)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output), "active_rules": len(active),
                      "final_baseline": summarize(before), "final_routed": summarize(after),
                      "model_restored_exact": report["model_restored_exact"], "seconds": report["seconds"]}, indent=2))