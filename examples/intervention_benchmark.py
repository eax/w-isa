"""Compare baseline, W-ISA materialization and an equivalent rank-one hook."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time

import numpy as np
import torch
import transformers
from transformers.utils.hub import cached_file

from wisa import WeightRuntime, compile_program
from wisa.model import encode_prompt, load_model


MODES = ("baseline", "wisa", "hook")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class ProcessMemory:
    """Current RSS from proc_pidinfo on macOS or procfs on Linux."""

    def __init__(self):
        self.library = ctypes.CDLL("/usr/lib/libproc.dylib") if platform.system() == "Darwin" else None
        if self.library is not None:
            self.library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                                  ctypes.c_void_p, ctypes.c_int]
            self.library.proc_pidinfo.restype = ctypes.c_int

    def rss(self):
        if self.library is not None:
            buffer = (ctypes.c_uint64 * 12)()
            size = self.library.proc_pidinfo(os.getpid(), 4, 0, buffer, ctypes.sizeof(buffer))
            if size != ctypes.sizeof(buffer):
                raise RuntimeError("proc_pidinfo failed to return process memory")
            return int(buffer[1])
        statm = Path("/proc/self/statm")
        if statm.exists():
            return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return None


class MemorySampler:
    """Sample current process/device allocations through one measured request."""

    def __init__(self, device, interval=0.01):
        self.device, self.interval = device, interval
        self.process = ProcessMemory()
        self.stop = threading.Event()
        self.thread = None
        self.maximum = {}
        self.errors = []

    def sample(self):
        values = {"rss_bytes": self.process.rss()}
        if self.device.type == "mps":
            values.update(device_active_bytes=torch.mps.current_allocated_memory(),
                          device_driver_bytes=torch.mps.driver_allocated_memory())
        elif self.device.type == "cuda":
            values.update(device_active_bytes=torch.cuda.memory_allocated(self.device),
                          device_driver_bytes=torch.cuda.memory_reserved(self.device))
        for key, value in values.items():
            if value is not None:
                self.maximum[key] = max(value, self.maximum.get(key, 0))
        return values

    def worker(self):
        try:
            while not self.stop.wait(self.interval):
                self.sample()
        except Exception as error:
            self.errors.append(str(error))

    def __enter__(self):
        self.initial = self.sample()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.final = self.sample()
        if self.errors:
            raise RuntimeError("memory sampling failed: " + "; ".join(self.errors))

    def result(self):
        return {"sampling_interval_seconds": self.interval, "start": self.initial,
                "sampled_peak": self.maximum, "end": self.final,
                "sampled_peak_above_start": {key: value - self.initial[key]
                                             for key, value in self.maximum.items()
                                             if self.initial.get(key) is not None}}


def rank_one_vectors(rows, columns):
    left = torch.sin(torch.arange(1, rows + 1, dtype=torch.float64, device="cpu"))
    right = torch.cos(torch.arange(1, columns + 1, dtype=torch.float64, device="cpu"))
    return left / torch.linalg.vector_norm(left), right / torch.linalg.vector_norm(right)


@contextmanager
def intervention_scope(mode, runtime, module, program, left, right, alpha):
    with runtime.lock, torch.no_grad():
        if mode == "wisa":
            with runtime.temporary(program):
                yield
        elif mode == "hook":
            weight = module.weight
            u = left.to(dtype=weight.dtype).to(weight.device)
            v = right.to(dtype=weight.dtype).to(weight.device)

            def intervene(_module, inputs, output):
                return output + alpha * (inputs[0] @ v).unsqueeze(-1) * u

            handle = module.register_forward_hook(intervene)
            try:
                yield
            finally:
                synchronize(weight.device)
                handle.remove()
        elif mode == "baseline":
            yield
        else:
            raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def fixed_decode(model, encoded, count):
    """Exactly count greedy tokens; synchronization defines each timestamp."""
    if count < 1:
        raise ValueError("token count must be positive")
    device = encoded["input_ids"].device
    prefix_length = encoded["input_ids"].shape[1]
    mask = torch.ones((1, prefix_length + count), dtype=torch.long, device=device)
    mask[:, :prefix_length] = encoded["attention_mask"]
    token = encoded["input_ids"]
    cache = None
    outputs = []
    times = []
    extra = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(model.forward).parameters else {}
    synchronize(device)
    start = time.perf_counter()
    for step in range(count):
        result = model(input_ids=token, attention_mask=mask[:, :prefix_length + step],
                       past_key_values=cache, use_cache=True, **extra)
        token = result.logits[:, -1:].argmax(dim=-1)
        cache = result.past_key_values
        outputs.append(token)
        synchronize(device)
        times.append(time.perf_counter())
    ids = torch.cat(outputs, dim=1)[0].cpu().tolist()
    return {"token_ids": ids, "prefill_seconds": times[0] - start,
            "first_token_at": times[0],
            "decode_seconds": times[-1] - times[0],
            "decode_seconds_per_token": (times[-1] - times[0]) / (count - 1) if count > 1 else 0.0,
            "forward_seconds": times[-1] - start}


@torch.no_grad()
def validate_equivalence(model, encoded, runtime, module, program, left, right, alpha):
    original = module.weight.detach().cpu().clone()
    identity = id(module.weight)
    before = model(**encoded, use_cache=False).logits.detach().cpu()
    with intervention_scope("wisa", runtime, module, program, left, right, alpha):
        edited = module.weight.detach().cpu().clone()
        materialized = model(**encoded, use_cache=False).logits.detach().cpu()
    with intervention_scope("hook", runtime, module, program, left, right, alpha):
        hooked = model(**encoded, use_cache=False).logits.detach().cpu()
    restored = model(**encoded, use_cache=False).logits.detach().cpu()
    reference64 = original.double().numpy() + alpha * np.outer(left.numpy(), right.numpy())
    reference = torch.from_numpy(reference64).to(original.dtype).float().numpy()
    np.testing.assert_allclose(edited.float().numpy(), reference, rtol=0, atol=torch.finfo(original.dtype).eps)
    if not all(torch.isfinite(value).all().item() for value in (before, materialized, hooked, restored)):
        raise AssertionError("nonfinite validation logits")
    if not torch.equal(before, restored) or not torch.equal(module.weight.cpu(), original):
        raise AssertionError("validation restoration differs")
    if identity != id(module.weight):
        raise AssertionError("parameter identity changed")
    delta = (materialized - hooked).double()
    return {"weight_reference_max_abs_error": float(np.max(np.abs(reference - edited.float().numpy()))),
            "hook_vs_materialized_max_abs_logit_error": delta.abs().max().item(),
            "hook_vs_materialized_rms_logit_error": delta.square().mean().sqrt().item(),
            "hook_vs_materialized_argmax_agreement": (materialized.argmax(-1) == hooked.argmax(-1)).double().mean().item(),
            "intervention_max_abs_logit_change": (materialized - before).abs().max().item(),
            "weight_restored_exact": True, "logits_restored_exact": True,
            "parameter_identity_preserved": True}


def aggregate(trials):
    summaries = []
    for count in sorted({row["tokens"] for row in trials}):
        for mode in MODES:
            group = [row for row in trials if row["tokens"] == count and row["mode"] == mode]
            summary = {"tokens": count, "mode": mode, "repetitions": len(group)}
            for key in ("apply_seconds", "restore_seconds", "request_seconds", "request_ttft_seconds",
                        "prefill_seconds", "decode_seconds_per_token"):
                values = np.array([row[key] for row in group])
                summary[key] = {"median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                                "q75": float(np.quantile(values, 0.75)), "min": float(values.min()), "max": float(values.max())}
            summary["sampled_peak_bytes"] = {key: max(row["memory"]["sampled_peak"].get(key, 0) for row in group)
                                             for key in group[0]["memory"]["sampled_peak"]}
            summaries.append(summary)
    return summaries


def write_figures(report, stem):
    """SVG preview and a standalone PGFPlots figure from measured medians/IQR."""
    summaries = report["summary"]
    colors = {"baseline": "#59636e", "wisa": "#1769aa", "hook": "#d46b08"}
    panels = [("request_seconds", "Request time (s)", 1),
              ("decode_seconds_per_token", "Decode time (ms/token)", 1000)]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420" viewBox="0 0 1000 420">',
           '<rect width="1000" height="420" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="13" fill="#20262e">']
    lengths = sorted({row["tokens"] for row in summaries})
    for panel, (metric, title, multiplier) in enumerate(panels):
        left, top, width, height = 65 + panel * 500, 60, 395, 260
        maximum = max(row[metric]["q75"] * multiplier for row in summaries) * 1.15 or 1
        maximum = math.ceil(maximum * 10) / 10
        x = lambda value: left + (value - lengths[0]) / max(1, lengths[-1] - lengths[0]) * width
        y = lambda value: top + height - value / maximum * height
        svg.append(f'<text x="{left}" y="28" font-size="17">{title}</text>')
        for tick in range(6):
            value = tick * maximum / 5
            svg.append(f'<line x1="{left}" y1="{y(value):.2f}" x2="{left+width}" y2="{y(value):.2f}" stroke="#e3e7eb"/>')
            svg.append(f'<text x="{left-9}" y="{y(value)+4:.2f}" text-anchor="end">{value:.2f}</text>')
        for length in lengths:
            svg.append(f'<text x="{x(length):.2f}" y="343" text-anchor="middle">{length}</text>')
        for mode in MODES:
            points = [row for row in summaries if row["mode"] == mode]
            points.sort(key=lambda row: row["tokens"])
            coordinates = ' '.join(f'{x(row["tokens"]):.2f},{y(row[metric]["median"]*multiplier):.2f}' for row in points)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{colors[mode]}" stroke-width="2"/>')
            for row in points:
                cx, cy = x(row["tokens"]), y(row[metric]["median"] * multiplier)
                svg.append(f'<line x1="{cx}" x2="{cx}" y1="{y(row[metric]["q25"]*multiplier)}" y2="{y(row[metric]["q75"]*multiplier)}" stroke="{colors[mode]}" stroke-width="3"/>')
                svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="{colors[mode]}"/>')
        svg.append(f'<text x="{left+width/2}" y="369" text-anchor="middle">Generated tokens</text>')
    for i, mode in enumerate(MODES):
        svg.append(f'<text x="{360+i*110}" y="405" fill="{colors[mode]}" font-size="13">{mode}</text>')
    svg.append('</g></svg>')
    stem.with_suffix(".svg").write_text('\n'.join(svg) + '\n')
    tex = [r'\documentclass[tikz,border=6pt]{standalone}', r'\usepackage{pgfplots}',
           r'\usepgfplotslibrary{groupplots}', r'\pgfplotsset{compat=1.18}', r'\begin{document}',
           r'\begin{tikzpicture}',
           r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.8cm},width=7.5cm,height=6cm,xlabel={Generated tokens},xtick={' + ','.join(map(str, lengths)) + r'},grid=major,legend columns=3,legend style={font=\footnotesize,at={(0.5,-0.32)},anchor=north},tick label style={font=\small}]']
    for metric, title, multiplier in panels:
        tex.append(r'\nextgroupplot[ylabel={' + title + '},ymin=0]')
        for mode, color in (("baseline", "gray"), ("wisa", "blue"), ("hook", "orange")):
            points = sorted((row for row in summaries if row["mode"] == mode), key=lambda row: row["tokens"])
            coords = ' '.join(f'({row["tokens"]},{row[metric]["median"]*multiplier:.9f}) += (0,{(row[metric]["q75"]-row[metric]["median"])*multiplier:.9f}) -= (0,{(row[metric]["median"]-row[metric]["q25"])*multiplier:.9f})' for row in points)
            tex.append(r'\addplot+[color=' + color + r',mark=*,error bars/.cd,y dir=both,y explicit] coordinates {' + coords + '};')
            tex.append(r'\addlegendentry{' + mode + '}')
    tex.extend([r'\end{groupplot}', r'\end{tikzpicture}', r'\end{document}'])
    stem.with_suffix(".tex").write_text('\n'.join(tex) + '\n')


def run(args):
    if args.repeats < 1 or args.warmups < 0 or not args.lengths or min(args.lengths) < 1 or not math.isfinite(args.alpha):
        raise ValueError("positive repeats/lengths, nonnegative warmups and finite alpha required")
    model, tokenizer = load_model(args.model, args.device, args.dtype, args.local_files_only)
    runtime = WeightRuntime(model)
    module = model.get_submodule(args.module)
    rows, columns = module.weight.shape
    left, right = rank_one_vectors(rows, columns)
    source = {"version": 1, "instructions": [
        {"op": "RANK1", "target": args.module + ".weight", "left": "u", "right": "v", "alpha": args.alpha},
    ]}
    program = compile_program(source, model, {"u": left, "v": right})
    device = module.weight.device
    original_hash = tensor_digest(module.weight)
    encoded = encode_prompt(tokenizer, args.prompt, device)
    validation = validate_equivalence(model, encoded, runtime, module, program, left, right, args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(cached_file(args.model, "config.json", local_files_only=True))
    report = {"model": args.model, "revision": config_path.parent.name,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
              "created_utc": datetime.now(timezone.utc).isoformat(), "device": str(device), "dtype": str(module.weight.dtype),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "target_shape": list(module.weight.shape), "snapshot_bytes": module.weight.numel() * module.weight.element_size(),
              "one_fp64_work_buffer_bytes": module.weight.numel() * 8, "operand_fp64_bytes": (rows + columns) * 8,
              "prompt": args.prompt, "input_token_ids": encoded["input_ids"][0].cpu().tolist(),
              "program": source, "left_sha256": tensor_digest(left), "right_sha256": tensor_digest(right),
              "lengths": args.lengths, "repeats": args.repeats, "warmups_per_mode_per_length": args.warmups,
              "protocol": "batch=1, greedy fixed-length decoding including EOS, fresh cache per request, synchronize each token; order rotates each repetition",
              "memory_protocol": "10ms sampled process RSS and device allocator counters, absolute/start/end/delta; macOS unified-memory counters describe overlapping storage",
              "validation": validation, "trials": []}
    np.savez(args.output.with_suffix(".operands.npz"), u=left.numpy(), v=right.numpy())
    args.output.with_suffix(".program.json").write_text(json.dumps(source, indent=2) + '\n')
    for length in args.lengths:
        for _ in range(args.warmups):
            for mode in MODES:
                with intervention_scope(mode, runtime, module, program, left, right, args.alpha):
                    fixed_decode(model, encoded, length)
        for repeat in range(args.repeats):
            order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
            for mode in order:
                gc.collect()
                synchronize(device)
                with MemorySampler(device) as memory:
                    start = time.perf_counter()
                    with intervention_scope(mode, runtime, module, program, left, right, args.alpha):
                        synchronize(device)
                        entered = time.perf_counter()
                        decoded = fixed_decode(model, encoded, length)
                        body_end = time.perf_counter()
                    synchronize(device)
                    end = time.perf_counter()
                if tensor_digest(module.weight) != original_hash:
                    raise AssertionError("timed request changed base weights after scope exit")
                row = {"mode": mode, "tokens": length, "repeat": repeat,
                       "apply_seconds": entered - start, "restore_seconds": end - body_end,
                       "request_seconds": end - start,
                       "request_ttft_seconds": decoded.pop("first_token_at") - start,
                       **decoded, "memory": memory.result(), "restored_exact": True}
                report["trials"].append(row)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(json.dumps({"mode": mode, "tokens": length, "repeat": repeat,
                                  "request_ms": row["request_seconds"]*1000,
                                  "decode_ms_per_token": row["decode_seconds_per_token"]*1000}), flush=True)
    report["summary"] = aggregate(report["trials"])
    report["restored_exact"] = tensor_digest(module.weight) == original_hash
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with args.output.with_suffix(".csv").open('w', newline='') as handle:
        fields = ["mode", "tokens", "repeat", "apply_seconds", "restore_seconds", "request_seconds",
                  "request_ttft_seconds", "prefill_seconds", "decode_seconds_per_token"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in report["trials"])
    write_figures(report, args.output.with_suffix(".latency"))
    print(json.dumps({"output": str(args.output), "validation": validation,
                      "snapshot_bytes": report["snapshot_bytes"], "restored_exact": report["restored_exact"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--module", default="model.layers.9.mlp.down_proj")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prompt", default="Write a detailed tutorial on matrix multiplication, including its definition, dimensions, a worked example, and practical applications.")
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen15b-benchmark.json"))
    run(parser.parse_args())