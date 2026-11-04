"""Small CLI: offline demo, parameter inspection, and live model edits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .compiler import compile_program, load_operands
from .model import DEFAULT_MODEL, generate, load_model
from .runtime import WeightRuntime


@torch.no_grad()
def demo() -> dict[str, Any]:
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    runtime = WeightRuntime(model)
    key = torch.tensor([1.0, 2.0], dtype=torch.float64)
    value = torch.tensor([3.0, -1.0], dtype=torch.float64)
    program = compile_program({"version": 1, "instructions": [
        {"op": "WRITE_KEY", "target": "weight", "key": "k", "value": "v"},
    ]}, model, {"k": key, "v": value})
    before = model(key).tolist()
    reports = runtime.execute(program)
    after = model(key).tolist()
    weight = model.weight.tolist()
    runtime.undo()
    return {
        "before": before, "requested": value.tolist(), "after": after,
        "edited_weight": weight, "restored": model(key).tolist(),
        "reports": [asdict(report) for report in reports],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient-free local weight-space instructions")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="offline, fixed 2x2 matrix edit and undo")
    for name in ("inspect", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local path")
        sub.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
        sub.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
        sub.add_argument("--local-files-only", action="store_true", help="never download model files")
        if name == "inspect":
            sub.add_argument("--contains", default="", help="parameter name substring")
        else:
            sub.add_argument("--program", type=Path, required=True)
            sub.add_argument("--operands", type=Path, help="NPZ named arrays (no pickle)")
            sub.add_argument("--prompt", help="compare greedy generation before and after the edit")
            sub.add_argument("--max-new-tokens", type=int, default=64)
            sub.add_argument("--undo", action="store_true", help="restore after the comparison")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            output = demo()
        else:
            model, tokenizer = load_model(args.model, args.device, args.dtype, args.local_files_only)
            if args.command == "inspect":
                output = {
                    name: {"shape": list(p.shape), "dtype": str(p.dtype), "device": str(p.device)}
                    for name, p in model.named_parameters(remove_duplicate=False)
                    if args.contains in name
                }
            else:
                source = json.loads(args.program.read_text(encoding="utf-8"))
                operands = load_operands(args.operands) if args.operands else {}
                runtime = WeightRuntime(model)
                program = compile_program(source, model, operands)
                output = {"model": args.model, "device": str(next(model.parameters()).device)}
                if args.prompt is not None:
                    output["before"] = generate(model, tokenizer, args.prompt, args.max_new_tokens)
                reports = runtime.execute(program)
                output["reports"] = [asdict(report) for report in reports]
                if args.prompt is not None:
                    output["after"] = generate(model, tokenizer, args.prompt, args.max_new_tokens)
                if args.undo:
                    output["undone"] = runtime.undo()
                    if args.prompt is not None:
                        output["restored"] = generate(model, tokenizer, args.prompt, args.max_new_tokens)
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"wisa: {exc}\n")


if __name__ == "__main__":
    main()