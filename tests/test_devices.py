"""Device selection, synchronization routing, and CUDA roundtrip integration."""

import asyncio
import contextlib
import io
import unittest
from unittest.mock import call, patch

import torch
from torch import nn

from wisa import ExecutionError, WeightRuntime, compile_program
from wisa.cli import main
from wisa.control import MatchActionController, Rule
from wisa.model import select_device
from wisa.profiling import ActivationSite
from wisa.runtime import _synchronize_devices
from test_profiling import Model, Tokenizer


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_prefers_cuda_and_uses_current_index(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=2), \
             patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(select_device(), torch.device("cuda:2"))

    def test_auto_mps_and_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            for available, expected in ((True, "mps"), (False, "cpu")):
                with patch("torch.backends.mps.is_available", return_value=available):
                    self.assertEqual(select_device(), torch.device(expected))

    def test_explicit_cuda_indices(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=1), \
             patch("torch.cuda.device_count", return_value=3):
            self.assertEqual(select_device("cuda"), torch.device("cuda:1"))
            self.assertEqual(select_device("cuda:0"), torch.device("cuda:0"))
            self.assertEqual(select_device("cuda:2"), torch.device("cuda:2"))
            for name in ("cuda:3", "cuda:128", "cuda:256"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    select_device(name)

    def test_explicit_device_errors_and_cpu_selection(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            for name in ("cuda", "cuda:0", "mps", "cuda:-1", "cuda:abc", "meta", "unknown"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    select_device(name)
            self.assertEqual(select_device("cpu"), torch.device("cpu"))

    def test_cpu_avoids_cuda_initialization(self):
        with patch("torch.cuda.is_available") as available, patch("torch.cuda.current_device") as current:
            self.assertEqual(select_device("cpu"), torch.device("cpu"))
            available.assert_not_called()
            current.assert_not_called()

    def test_cli_passes_indexed_cuda_to_loader(self):
        model = nn.Linear(2, 2, bias=False)
        with patch("sys.argv", ["wisa", "inspect", "--device", "cuda:2", "--local-files-only"]), \
             patch("wisa.cli.load_model", return_value=(model, None)) as loader, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        self.assertEqual(loader.call_args.args[1], "cuda:2")
        self.assertTrue(loader.call_args.args[3])
        self.assertIn("weight", output.getvalue())


class SynchronizationTests(unittest.TestCase):
    def test_each_cuda_device_and_mps_synchronized_once(self):
        devices = [torch.device(x) for x in ("cuda:1", "cpu", "cuda:0", "cuda:1", "mps", "mps")]
        with patch("torch.cuda.synchronize") as cuda, patch("torch.mps.synchronize") as mps:
            _synchronize_devices(iter(devices))
        self.assertEqual(cuda.call_args_list, [call(torch.device("cuda:0")), call(torch.device("cuda:1"))])
        mps.assert_called_once_with()

    def test_cpu_and_empty_targets_use_no_accelerator(self):
        with patch("torch.cuda.synchronize") as cuda, patch("torch.mps.synchronize") as mps:
            _synchronize_devices([])
            _synchronize_devices([torch.device("cpu")])
        cuda.assert_not_called()
        mps.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware tests")
class CUDARoundtripTests(unittest.TestCase):
    def matrix(self, device, dtype=torch.float32):
        model = nn.Linear(4, 3, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            model.weight.copy_(torch.arange(1, 13, device=device).reshape(3, 4))
        return model

    def test_all_instructions_match_cpu_and_restore_exactly(self):
        instructions = {"version": 1, "instructions": [
            {"op": "ADD", "target": "weight", "delta": "d", "alpha": 0.1},
            {"op": "SCALE", "target": "weight", "slice": [None, [1, 3]], "factor": 0},
            {"op": "RANK1", "target": "weight", "left": "u", "right": "v"},
            {"op": "PROJECT_OUT", "target": "weight", "direction": "v"},
            {"op": "WRITE_KEY", "target": "weight", "key": "v", "value": "u"},
            {"op": "PRUNE", "target": "weight", "threshold": 0.5},
        ]}
        operands = {"d": torch.ones(3, 4, device="cuda:0"), "u": [2, -1, 3], "v": [1, 2, 3, 4]}
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                cpu, gpu = self.matrix("cpu", dtype), self.matrix("cuda:0", dtype)
                original = gpu.weight.detach().cpu().clone()
                identity = id(gpu.weight)
                program = compile_program(instructions, gpu, operands)
                host, device = WeightRuntime(cpu), WeightRuntime(gpu)
                reports = device.execute(program)
                self.assertEqual(reports, host.execute(program))
                self.assertTrue(torch.equal(cpu.weight, gpu.weight.cpu()))
                self.assertEqual(gpu.weight.device, torch.device("cuda:0"))
                edited = gpu.weight.cpu().clone()
                self.assertTrue(device.undo())
                self.assertTrue(torch.equal(gpu.weight.cpu(), original))
                self.assertEqual(device.execute(program), reports)
                self.assertTrue(torch.equal(gpu.weight.cpu(), edited))
                self.assertEqual(id(gpu.weight), identity)
                self.assertIsNone(gpu.weight.grad)

    def test_overflow_rolls_back_partial_cuda_writes(self):
        model = self.matrix("cuda:0", torch.float16)
        runtime = WeightRuntime(model)
        original = model.weight.cpu().clone()
        program = compile_program({"version": 1, "instructions": [
            {"op": "SCALE", "target": "weight", "slice": [None, [0, 2]], "factor": 0},
            {"op": "SCALE", "target": "weight", "factor": 1e10},
        ]}, model)
        with self.assertRaises(ExecutionError):
            runtime.execute(program)
        self.assertTrue(torch.equal(model.weight.cpu(), original))

    def test_cuda_request_cancellation_and_stream_close(self):
        model = Model().to("cuda:0")
        runtime = WeightRuntime(model)
        program = compile_program({"version": 1, "instructions": [
            {"op": "SCALE", "target": "layer.weight", "factor": 0},
        ]}, model)
        rule = Rule("cuda", ActivationSite("layer"), torch.tensor([1., 0.]), 0.5, program)
        controller = MatchActionController(runtime, Tokenizer(), [rule])
        original = model.layer.weight.cpu().clone()
        with self.assertRaises(asyncio.CancelledError):
            with controller.session("0"):
                self.assertEqual(torch.count_nonzero(model.layer.weight).item(), 0)
                raise asyncio.CancelledError()
        self.assertTrue(torch.equal(model.layer.weight.cpu(), original))
        stream = controller.stream("0", lambda: iter(["one", "two"]))
        self.assertEqual(next(stream), "one")
        stream.close()
        self.assertTrue(torch.equal(model.layer.weight.cpu(), original))

    def test_side_stream_work_finishes_before_restoration(self):
        model = self.matrix("cuda:0")
        runtime = WeightRuntime(model)
        original = model.weight.cpu().clone()
        program = compile_program({"version": 1, "instructions": [
            {"op": "SCALE", "target": "weight", "factor": 0},
        ]}, model)
        stream = torch.cuda.Stream(device="cuda:0")
        with runtime.temporary(program):
            with torch.cuda.stream(stream):
                observed = model.weight.clone()
        self.assertEqual(torch.count_nonzero(observed).item(), 0)
        self.assertTrue(torch.equal(model.weight.cpu(), original))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Two-device CUDA test")
    def test_all_touched_cuda_devices_are_restored(self):
        model = nn.ModuleList([self.matrix("cuda:0"), self.matrix("cuda:1")])
        runtime = WeightRuntime(model)
        originals = [p.cpu().clone() for p in model.parameters()]
        program = compile_program({"version": 1, "instructions": [
            {"op": "SCALE", "target": "0.weight", "factor": 0},
            {"op": "SCALE", "target": "1.weight", "factor": 0},
        ]}, model)
        with runtime.temporary(program):
            self.assertTrue(all(torch.count_nonzero(p).item() == 0 for p in model.parameters()))
        self.assertTrue(all(torch.equal(p.cpu(), original) for p, original in zip(model.parameters(), originals)))


if __name__ == "__main__":
    unittest.main()