"""仅CPU：两次更新预算、AdamW状态保留及失败尝试计数。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train.target_probe_core import TwoUpdateAdamW


class TwoUpdateTests(unittest.TestCase):
    def create(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32))
        audit = []
        optimizer = TwoUpdateAdamW([parameter], lr=0.01, weight_decay=0.0, foreach=False,
                                   audit_callback=lambda state: audit.append(copy.deepcopy(state)))
        return parameter, optimizer, audit

    def backward(self, parameter, optimizer):
        optimizer.zero_grad(set_to_none=True)
        parameter.square().sum().backward()

    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(actual[name], value), name)
            else:
                self.assertEqual(actual[name], value)

    def test_two_updates_retain_same_adamw_state_objects_and_increment_step(self):
        parameter, optimizer, audit = self.create()
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (0, 0))
        self.assertEqual(len(optimizer.state), 0)
        initial = parameter.detach().clone()
        self.backward(parameter, optimizer)
        optimizer.step()
        first_value = parameter.detach().clone()
        self.assertFalse(torch.equal(initial, first_value))
        state = optimizer.state[parameter]
        tensors = {key: value for key, value in state.items()}
        first_moment = state["exp_avg"].clone()
        self.assertEqual(state["step"].item(), 1)
        self.backward(parameter, optimizer)
        optimizer.step()
        self.assertIs(optimizer.state[parameter], state)
        for key, value in tensors.items():
            self.assertIs(state[key], value)
        self.assertFalse(torch.equal(state["exp_avg"], first_moment))
        self.assertFalse(torch.equal(parameter, first_value))
        self.assertEqual(state["step"].item(), 2)
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (2, 2))

    def test_third_update_rejected_without_parameter_state_counter_or_callback_changes(self):
        parameter, optimizer, audit = self.create()
        for _ in range(2):
            self.backward(parameter, optimizer)
            optimizer.step()
        before_parameter = parameter.detach().clone()
        before_state = copy.deepcopy(optimizer.state[parameter])
        before_audit = copy.deepcopy(audit)
        self.backward(parameter, optimizer)
        with self.assertRaisesRegex(RuntimeError, "Two-update budget"):
            optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        self.assert_state_equal(optimizer.state[parameter], before_state)
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (2, 2))
        self.assertEqual(audit, before_audit)

    def test_failure_inside_second_parent_step_is_counted_and_cannot_retry(self):
        parameter, optimizer, audit = self.create()
        self.backward(parameter, optimizer)
        optimizer.step()
        self.backward(parameter, optimizer)
        with patch.object(torch.optim.AdamW, "step", side_effect=RuntimeError("injected second-step failure")):
            with self.assertRaisesRegex(RuntimeError, "injected second-step failure"):
                optimizer.step()
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (2, 1))
        self.assertEqual(audit[-1], {"step_calls": 2, "completed_steps": 1, "partial_update_possible": True})
        before_parameter = parameter.detach().clone()
        before_state = copy.deepcopy(optimizer.state[parameter])
        before_audit = copy.deepcopy(audit)
        with self.assertRaisesRegex(RuntimeError, "Previous optimizer attempt failed"):
            optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        self.assert_state_equal(optimizer.state[parameter], before_state)
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (2, 1))
        self.assertEqual(audit, before_audit)

    def test_failure_inside_first_parent_step_locks_remaining_budget_without_retry(self):
        parameter, optimizer, audit = self.create()
        self.backward(parameter, optimizer)
        with patch.object(torch.optim.AdamW, "step", side_effect=RuntimeError("injected first-step failure")) as parent_step:
            with self.assertRaisesRegex(RuntimeError, "injected first-step failure"):
                optimizer.step()
            self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (1, 0))
            self.assertEqual(audit, [{"step_calls": 1, "completed_steps": 0, "partial_update_possible": True}])
            before_parameter = parameter.detach().clone()
            before_state = copy.deepcopy(optimizer.state_dict())
            before_audit = copy.deepcopy(audit)
            with self.assertRaisesRegex(RuntimeError, "Previous optimizer attempt failed"):
                optimizer.step()
            self.assertEqual(parent_step.call_count, 1)
        self.assertTrue(torch.equal(parameter, before_parameter))
        self.assertEqual(optimizer.state_dict(), before_state)
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (1, 0))
        self.assertEqual(audit, before_audit)

    def test_callback_distinguishes_started_and_completed_updates(self):
        parameter, optimizer, audit = self.create()
        self.assertEqual(audit, [])
        for _ in range(2):
            self.backward(parameter, optimizer)
            optimizer.step()
        self.assertEqual(audit, [
            {"step_calls": 1, "completed_steps": 0, "partial_update_possible": True},
            {"step_calls": 1, "completed_steps": 1, "partial_update_possible": False},
            {"step_calls": 2, "completed_steps": 1, "partial_update_possible": True},
            {"step_calls": 2, "completed_steps": 2, "partial_update_possible": False},
        ])


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TwoUpdateTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "pass" if result.wasSuccessful() else "fail", "tests": result.testsRun,
        "seconds": time.perf_counter() - started, "failures": len(result.failures), "errors": len(result.errors),
        "failure_details": [message for _, message in result.failures + result.errors],
        "device": "cpu", "qwen_model_loaded": False, "gpu_used": False,
        "scope": "Synthetic CPU parameters only; no Qwen optimizer updates",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ("src/train/target_probe_core.py", "src/train/test_target_probe_core.py")}}
    (ROOT / "reports/target_probe_core_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)
