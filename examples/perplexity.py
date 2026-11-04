"""Evaluate baseline/head-only/MLP-only/combined against the same held-out tokens."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wisa import Program, WeightRuntime, compile_program, load_operands
from wisa.evaluation import perplexity
from wisa.model import load_model


def run(args):
    corpus = json.loads(args.corpus.read_text())
    text = corpus["separator"].join(corpus["rows"])
    if hashlib.sha256(text.encode()).hexdigest() != corpus["text_sha256"]:
        raise ValueError("corpus checksum mismatch")
    model, tokenizer = load_model(args.model, device=args.device, local_files_only=True)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)["input_ids"][0]
    full_count = len(ids)
    if args.max_tokens is not None:
        if args.max_tokens < 2:
            raise ValueError("max-tokens must be at least two")
        ids = ids[:args.max_tokens]
    source = json.loads(args.program.read_text())
    program = compile_program(source, model, load_operands(args.operands))
    runtime = WeightRuntime(model)
    params = dict(model.named_parameters())
    originals = {ins.target: params[ins.target].detach().cpu().clone() for ins in program.instructions}
    conditions = {"baseline": Program(()),
                  "head_only": Program(tuple(ins for ins in program.instructions if ins.op == "SCALE")),
                  "mlp_only": Program(tuple(ins for ins in program.instructions if ins.op == "PROJECT_OUT")),
                  "combined": program}
    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "device": str(next(model.parameters()).device),
              "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
              "dataset": {key: value for key, value in corpus.items() if key != "rows"},
              "corpus_tokens": full_count, "evaluated_tokens": len(ids),
              "full_test_split": len(ids) == full_count,
              "token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
              "program": source, "operand_file": str(args.operands),
              "protocol": "raw text, no BOS/chat template; rolling context; score each token except first once",
              "conditions": results}
    for name, instructions in conditions.items():
        print(f"Evaluating {name}: {len(ids)} tokens", flush=True)
        with runtime.temporary(instructions):
            result = perplexity(model, ids, args.context_length, args.stride)
        for target, original in originals.items():
            if not torch.equal(params[target].detach().cpu(), original):
                raise AssertionError(f"{name}: undo differs for {target}")
        results[name] = result.to_dict()
        if name != "baseline":
            results[name]["ppl_change_percent"] = 100 * (result.perplexity / results["baseline"]["perplexity"] - 1)
            results[name]["mean_nll_change"] = result.mean_nll - results["baseline"]["mean_nll"]
        results[name]["undo_exact"] = True
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in results[name].items() if k != "windows"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--corpus", type=Path, default=Path("data/wikitext2-test.json"))
    parser.add_argument("--program", type=Path, default=Path("artifacts/smol-roundtrip.program.json"))
    parser.add_argument("--operands", type=Path, default=Path("artifacts/smol-roundtrip.operands.npz"))
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/perplexity.json"))
    run(parser.parse_args())