"""CPU FP64 algebra with transactional writes to CPU, MPS or CUDA parameters."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import weakref

import torch
from torch import nn

from .compiler import Instruction, Program


class ExecutionError(RuntimeError):
    """Execution failed; modified parameters have been restored."""


@dataclass(frozen=True)
class EditReport:
    index: int
    op: str
    target: str
    delta_norm: float
    changed_elements: int


@dataclass
class _Scope:
    backups: dict[int, tuple[nn.Parameter, torch.Tensor]]
    previous_undo: list[tuple[nn.Parameter, torch.Tensor]] | None


class _ModelState:
    def __init__(self):
        self.lock = threading.RLock()
        self.undo: list[tuple[nn.Parameter, torch.Tensor]] | None = None
        self.scopes: list[_Scope] = []


_states: weakref.WeakKeyDictionary[nn.Module, _ModelState] = weakref.WeakKeyDictionary()
_states_lock = threading.Lock()


def _synchronize_devices(devices):
    """Complete work on each touched accelerator, including indexed CUDA devices."""
    for device in sorted(set(devices), key=str):
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)


def _normalized_direction(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Scaling first avoids squaring a very small/large key in the denominator.
    scale = vector.abs().max()
    if scale.item() == 0 or not torch.isfinite(scale).item():
        raise ExecutionError("direction/key must be finite and nonzero")
    scaled = vector / scale
    return scaled / torch.linalg.vector_norm(scaled), scale


def _evaluate(weight: torch.Tensor, instruction: Instruction) -> torch.Tensor:
    args, alpha = instruction.operands, instruction.scalar
    if instruction.op == "ADD":
        return weight + alpha * args[0]
    if instruction.op == "SCALE":
        return weight * alpha
    if instruction.op == "RANK1":
        return weight + alpha * torch.outer(args[0], args[1])
    if instruction.op == "PROJECT_OUT":
        unit, _ = _normalized_direction(args[0])
        return weight - torch.outer(weight @ unit, unit)
    if instruction.op == "WRITE_KEY":
        key, value = args
        unit, scale = _normalized_direction(key)
        # W'k=y without explicitly forming k^T k or an n-by-n matrix.
        desired = (value / scale) / torch.linalg.vector_norm(key / scale)
        return weight + torch.outer(desired - weight @ unit, unit)
    if instruction.op == "PRUNE":
        return torch.where(weight.abs() < alpha, 0.0, weight)
    raise ExecutionError(f"unknown opcode {instruction.op!r}")


class WeightRuntime:
    """Execute in order, roll back failures, retain one successful undo point.

    Edits preserve Parameter identity and normal tied weights. Callers sharing
    the model between threads must hold ``lock`` during inference as well.
    Construction freezes parameters and selects eval mode; no training occurs.
    """

    def __init__(self, model: nn.Module):
        with _states_lock:
            if model not in _states:
                _states[model] = _ModelState()
            self._state = _states[model]
        self.lock = self._state.lock
        with self.lock:
            self.model = model.eval().requires_grad_(False)

    @property
    def _undo(self):
        return self._state.undo

    @_undo.setter
    def _undo(self, value):
        self._state.undo = value

    def _remember_in_scopes(self, backups):
        for scope in self._state.scopes:
            for identity, backup in backups.items():
                scope.backups.setdefault(identity, backup)

    @torch.no_grad()
    def _restore(self, backups):
        current_ids = {id(p) for p in self.model.parameters()}
        for target, original in backups:
            if id(target) not in current_ids or target.shape != original.shape:
                raise ExecutionError("model structure changed since the snapshot")
            if target.dtype != original.dtype:
                raise ExecutionError("weight dtype changed since the snapshot")
        _synchronize_devices(p.device for p, _ in backups)
        for target, original in backups:
            target.copy_(original)
        _synchronize_devices(p.device for p, _ in backups)

    @contextmanager
    def temporary(self, program: Program | None = None):
        """Hold the model lock; restore all runtime writes on every scope exit.

        Nested scopes and additional execute/undo calls enlist their touched
        parameters. The preexisting public undo point survives the scope.
        Instances wrapping the same model share lock, scopes and undo state.
        """
        with self.lock, torch.no_grad():
            scope = _Scope({}, self._undo)
            self._state.scopes.append(scope)
            try:
                reports = self.execute(program) if program is not None else ()
                yield reports
            finally:
                try:
                    self._restore(list(scope.backups.values()))
                finally:
                    self._state.scopes.pop()
                    self._undo = scope.previous_undo

    @torch.no_grad()
    def execute(self, program: Program) -> tuple[EditReport, ...]:
        with self.lock:
            parameters = dict(self.model.named_parameters(remove_duplicate=False))
            targets: dict[int, nn.Parameter] = {}
            # Rebind before writing: the module may have changed since compilation.
            for instruction in program.instructions:
                target = parameters.get(instruction.target)
                if target is None or tuple(target.shape) != instruction.shape:
                    raise ExecutionError(f"target changed: {instruction.target}")
                if (not target.is_floating_point() or target.layout != torch.strided
                        or target.device.type not in {"cpu", "mps", "cuda"}):
                    raise ExecutionError("runtime requires dense CPU/MPS/CUDA floating-point weights")
                targets[id(target)] = target

            _synchronize_devices(p.device for p in targets.values())
            backups = {identity: (target, target.detach().cpu().clone())
                       for identity, target in targets.items()}

            self._remember_in_scopes(backups)
            reports = []
            try:
                for index, instruction in enumerate(program.instructions):
                    target = parameters[instruction.target]
                    view = target if instruction.selection is None else target[
                        tuple(slice(start, stop) for start, stop in instruction.selection)
                    ]
                    before = view.detach().cpu().to(dtype=torch.float64).clone()
                    if not torch.isfinite(before).all().item():
                        raise ExecutionError(f"nonfinite target: {instruction.target}")
                    for operand in instruction.operands:
                        if not torch.isfinite(operand).all().item():
                            raise ExecutionError("compiled operand is no longer finite")
                    result = _evaluate(before, instruction)
                    if not torch.isfinite(result).all().item():
                        raise ExecutionError(f"FP64 overflow: {instruction.target}")
                    rounded = result.to(dtype=target.dtype)
                    if not torch.isfinite(rounded).all().item():
                        raise ExecutionError(f"weight dtype overflow: {instruction.target}")
                    difference = rounded.to(torch.float64) - before
                    reports.append(EditReport(
                        index, instruction.op, instruction.target,
                        torch.linalg.vector_norm(difference).item(),
                        int(torch.count_nonzero(difference).item()),
                    ))
                    view.copy_(rounded)
                _synchronize_devices(p.device for p, _ in backups.values())
            except BaseException as exc:
                for target, original in backups.values():
                    target.copy_(original)
                _synchronize_devices(p.device for p, _ in backups.values())
                if isinstance(exc, Exception):
                    raise ExecutionError(f"transaction rolled back: {exc}") from exc
                raise
            if backups:
                self._undo = list(backups.values())
            return tuple(reports)

    @torch.no_grad()
    def undo(self) -> bool:
        """Restore the last successful nonempty execution, bit for bit."""
        with self.lock:
            if self._undo is None:
                return False
            if self._state.scopes:
                self._remember_in_scopes({id(p): (p, p.detach().cpu().clone()) for p, _ in self._undo})
            self._restore(self._undo)
            self._undo = None
            return True

    def discard_undo(self) -> None:
        """Release the saved transaction without modifying weights."""
        with self.lock:
            self._undo = None