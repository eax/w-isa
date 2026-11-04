"""Compile JSON instructions against actual parameter names and shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class CompileError(ValueError):
    """Invalid source, target or operand."""


@dataclass(frozen=True)
class Instruction:
    op: str
    target: str
    shape: tuple[int, ...]
    operands: tuple[torch.Tensor, ...] = ()
    scalar: float = 1.0
    selection: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...]


def load_operands(path: str | Path) -> dict[str, np.ndarray]:
    """Read numeric NPZ arrays without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise CompileError(f"{name} is outside the FP64 range") from exc
    if not math.isfinite(result):
        raise CompileError(f"{name} must be finite")
    return result


@torch.no_grad()
def compile_program(
    source: Mapping[str, Any],
    model: nn.Module,
    operands: Mapping[str, Any] | None = None,
) -> Program:
    """Bind all targets and validate all instructions before any mutation.

    Operand tensors are detached CPU FP64 copies. No device pointers or model
    references are embedded, so a program may run on a compatible model.
    """
    if not isinstance(source, Mapping):
        raise CompileError("program must be an object")
    if set(source) != {"version", "instructions"}:
        raise CompileError("program requires exactly version and instructions")
    if type(source["version"]) is not int or source["version"] != 1:
        raise CompileError("unsupported ISA version; expected integer 1")
    if not isinstance(source["instructions"], list):
        raise CompileError("instructions must be a list")

    parameters = dict(model.named_parameters(remove_duplicate=False))
    bank = {} if operands is None else operands
    specs = {
        "ADD": (("delta",), "alpha"),
        "SCALE": ((), "factor"),
        "RANK1": (("left", "right"), "alpha"),
        "PROJECT_OUT": (("direction",), None),
        "WRITE_KEY": (("key", "value"), None),
        "PRUNE": ((), "threshold"),
    }
    compiled = []
    for index, raw in enumerate(source["instructions"]):
        try:
            if not isinstance(raw, Mapping):
                raise CompileError("instruction must be an object")
            op = raw.get("op")
            if not isinstance(op, str) or op not in specs:
                raise CompileError(f"unknown opcode {op!r}")
            fields, scalar_field = specs[op]
            required = {"op", "target", *fields}
            if scalar_field in {"factor", "threshold"}:
                required.add(scalar_field)
            allowed = required | {"slice"} | ({scalar_field} if scalar_field else set())
            if not required <= set(raw) or set(raw) - allowed:
                raise CompileError(f"invalid fields for {op}: expected {sorted(allowed)}")
            target = raw["target"]
            if not isinstance(target, str) or target not in parameters:
                raise CompileError(f"unknown parameter {target!r}")
            weight = parameters[target]
            if not weight.is_floating_point() or weight.layout != torch.strided:
                raise CompileError(f"{target} must be a dense floating-point parameter")
            if weight.device.type == "meta":
                raise CompileError(f"{target} has no materialized weights")
            shape = tuple(weight.shape)
            selection = None
            selected_shape = shape
            if "slice" in raw:
                ranges = raw["slice"]
                if not isinstance(ranges, list) or len(ranges) != len(shape):
                    raise CompileError("slice requires one range or null per target dimension")
                bounds = []
                for axis, (entry, size) in enumerate(zip(ranges, shape)):
                    if entry is None:
                        bounds.append((0, size))
                    elif (isinstance(entry, list) and len(entry) == 2
                          and all(type(x) is int for x in entry)
                          and 0 <= entry[0] < entry[1] <= size):
                        bounds.append((entry[0], entry[1]))
                    else:
                        raise CompileError(f"slice axis {axis} requires 0 <= start < stop <= {size}")
                selection = tuple(bounds)
                selected_shape = tuple(stop - start for start, stop in selection)
            scalar = _number(raw.get(scalar_field, 1.0), scalar_field or "scalar")
            if op == "PRUNE" and scalar < 0:
                raise CompileError("threshold must be nonnegative")
            arrays = []
            for field in fields:
                name = raw[field]
                if not isinstance(name, str) or name not in bank:
                    raise CompileError(f"unknown operand {name!r}")
                try:
                    value = bank[name]
                    if isinstance(value, torch.Tensor):
                        tensor = value
                    else:
                        array = np.asarray(value)
                        if np.iscomplexobj(array):
                            raise CompileError(f"operand {name!r} must be real")
                        tensor = torch.as_tensor(array.copy(), dtype=torch.float64, device="cpu")
                    if tensor.is_complex() or tensor.layout != torch.strided:
                        raise CompileError(f"operand {name!r} must be a dense real tensor")
                    # MPS cannot combine a host transfer with a Double cast.
                    tensor = tensor.detach().cpu().to(dtype=torch.float64).clone()
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise CompileError(f"invalid operand {name!r}: {exc}") from exc
                if not torch.isfinite(tensor).all().item():
                    raise CompileError(f"operand {name!r} is not finite")
                arrays.append(tensor)
            if op == "ADD":
                expected = (selected_shape,)
            elif op in {"RANK1", "PROJECT_OUT", "WRITE_KEY"}:
                if len(selected_shape) != 2:
                    raise CompileError(f"{op} requires a matrix target, got {selected_shape}")
                rows, cols = selected_shape
                expected = {
                    "RANK1": ((rows,), (cols,)),
                    "PROJECT_OUT": ((cols,),),
                    "WRITE_KEY": ((cols,), (rows,)),
                }[op]
            else:
                expected = ()
            if tuple(tuple(x.shape) for x in arrays) != expected:
                raise CompileError(f"{op} operand shapes must be {expected}")
            if op in {"PROJECT_OUT", "WRITE_KEY"}:
                if arrays[0].numel() == 0 or arrays[0].abs().max().item() == 0:
                    raise CompileError(f"{op} requires a nonzero direction/key")
            compiled.append(Instruction(op, target, shape, tuple(arrays), scalar, selection))
        except CompileError as exc:
            raise CompileError(f"instruction {index}: {exc}") from exc
    return Program(tuple(compiled))