"""CPU vectors verify the durable 0→2→4 update budget across optimizer instances."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.train.sft_update_budget import JournaledAdamW, read_ledger, write_atomic

CHECKS = ROOT / "outputs/checks/sft_update_budget_cpu"
REPORT = ROOT / "reports/sft_update_budget_cpu_checks.json"
OPTIONS = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0, "foreach": False}


def initialize_ledger(path, completed=0, failed=False):
    # The launcher owns creation: x refuses to replace any prior run's ledger.
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"attempted_updates": completed, "completed_updates": completed,
                   "total_budget": 4, "partial_update_possible": False, "failed": failed}, stream)


def parameter():
    return torch.nn.Parameter(torch.tensor([0.5, -1.0, 2.0], dtype=torch.float32))


def update(optimizer, value):
    optimizer.zero_grad(set_to_none=True)
    (value.square().sum() + 0.1 * value.sum()).backward()
    optimizer.step()


def assert_equal_state(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_equal_state(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_equal_state(test, a, b)
    else:
        test.assertEqual(left, right)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        CHECKS.mkdir(parents=True, exist_ok=True)
        if not CHECKS.resolve().is_relative_to(ROOT.resolve()):
            raise AssertionError("Temporary tests must stay inside the workspace")
        self.folder = tempfile.TemporaryDirectory(prefix="ledger-", dir=CHECKS)
        self.addCleanup(self.folder.cleanup)
        self.ledger = Path(self.folder.name) / "ledger.json"

    def make(self, value, start=0, end=2):
        return JournaledAdamW([value], ledger_path=self.ledger, start=start,
                              segment_end=end, total_budget=4, **OPTIONS)

    def assert_refused_unchanged(self, optimizer, value, message):
        before = self.ledger.read_bytes()
        weights = value.detach().clone()
        state = copy.deepcopy(optimizer.state_dict())
        counts = optimizer.step_calls, optimizer.completed_updates
        with self.assertRaisesRegex(RuntimeError, message):
            optimizer.step()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertTrue(torch.equal(value, weights))
        assert_equal_state(self, state, optimizer.state_dict())
        self.assertEqual((optimizer.step_calls, optimizer.completed_updates), counts)

    def test_01_two_instances_restore_state_and_continue_global_count(self):
        initialize_ledger(self.ledger)
        first_parameter = parameter()
        first = self.make(first_parameter)
        reference_parameter = parameter()
        reference = torch.optim.AdamW([reference_parameter], **OPTIONS)
        for _ in range(2):
            update(first, first_parameter)
            update(reference, reference_parameter)
        self.assertEqual(read_ledger(self.ledger)["completed_updates"], 2)
        saved = copy.deepcopy(first.state_dict())
        resumed_parameter = torch.nn.Parameter(first_parameter.detach().clone())
        resumed = self.make(resumed_parameter, start=2, end=4)
        resumed.load_state_dict(saved)
        self.assertEqual(float(resumed.state[resumed_parameter]["step"]), 2.0)
        self.assertEqual((resumed.step_calls, resumed.completed_updates), (2, 2))
        for _ in range(2):
            update(resumed, resumed_parameter)
            update(reference, reference_parameter)
        self.assertEqual((resumed.step_calls, resumed.completed_updates), (4, 4))
        self.assertEqual(read_ledger(self.ledger), {"attempted_updates": 4, "completed_updates": 4,
            "total_budget": 4, "partial_update_possible": False, "failed": False})
        self.assertTrue(torch.equal(resumed_parameter, reference_parameter))
        assert_equal_state(self, reference.state_dict(), resumed.state_dict())

    def test_02_first_segment_refuses_third_update(self):
        initialize_ledger(self.ledger)
        value = parameter()
        optimizer = self.make(value)
        update(optimizer, value)
        update(optimizer, value)
        self.assert_refused_unchanged(optimizer, value, "budget exhausted")

    def test_03_second_segment_refuses_fifth_global_update(self):
        initialize_ledger(self.ledger, completed=2)
        value = parameter()
        optimizer = self.make(value, start=2, end=4)
        update(optimizer, value)
        update(optimizer, value)
        self.assert_refused_unchanged(optimizer, value, "budget exhausted")

    def test_04_partial_parent_update_is_recorded_and_cannot_retry(self):
        initialize_ledger(self.ledger)
        value = parameter()
        optimizer = self.make(value)
        original = value.detach().clone()

        def fail_after_mutation(*args, **kwargs):
            with torch.no_grad():
                value[0].add_(0.01)
            raise RuntimeError("Injected partial Adam update failure")

        with patch.object(torch.optim.AdamW, "step", side_effect=fail_after_mutation) as parent_step:
            with self.assertRaisesRegex(RuntimeError, "Injected partial"):
                optimizer.step()
            self.assertEqual(parent_step.call_count, 1)
            self.assertFalse(torch.equal(value, original))
            self.assertEqual((optimizer.step_calls, optimizer.completed_updates), (1, 0))
            ledger = read_ledger(self.ledger)
            self.assertEqual((ledger["attempted_updates"], ledger["completed_updates"]), (1, 0))
            self.assertTrue(ledger["partial_update_possible"])
            self.assert_refused_unchanged(optimizer, value, "previous failure")
            self.assertEqual(parent_step.call_count, 1)
        # A newly constructed optimizer cannot turn that attempted update into unused budget either.
        with self.assertRaisesRegex(RuntimeError, "previous failure"):
            self.make(parameter())

    def test_05_failed_flag_refuses_constructor_and_existing_optimizer(self):
        initialize_ledger(self.ledger, failed=True)
        before = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "previous failure"):
            self.make(parameter())
        self.assertEqual(before, self.ledger.read_bytes())
        record = read_ledger(self.ledger)
        record["failed"] = False
        write_atomic(self.ledger, record)
        value = parameter()
        optimizer = self.make(value)
        update(optimizer, value)
        record = read_ledger(self.ledger)
        record["failed"] = True
        write_atomic(self.ledger, record)
        self.assert_refused_unchanged(optimizer, value, "previous failure")

    def test_06_stale_checkpoint_cannot_overwrite_newer_ledger(self):
        initialize_ledger(self.ledger, completed=2)
        before = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Ledger mismatch"):
            self.make(parameter(), start=0, end=2)
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertEqual(read_ledger(self.ledger)["completed_updates"], 2)

    def test_07_exclusive_ledger_initialization_never_overwrites(self):
        initialize_ledger(self.ledger, completed=2)
        before = self.ledger.read_bytes()
        with self.assertRaises(FileExistsError):
            initialize_ledger(self.ledger, completed=0)
        self.assertEqual(self.ledger.read_bytes(), before)


def main():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BudgetTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if torch.cuda.is_initialized():
        raise AssertionError("CPU budget tests initialized CUDA")
    sources = {}
    for name in ("src/train/sft_update_budget.py", "src/train/test_sft_update_budget.py"):
        with (ROOT / name).open("rb") as stream:
            sources[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"status": "pass" if result.wasSuccessful() else "failed",
              "utc": datetime.now(timezone.utc).isoformat(), "device": "cpu", "tests": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "failure_details": [detail for _, detail in result.failures + result.errors],
              "source_sha256": sources, "seconds": time.perf_counter() - started,
              "gpu_used": False, "cuda_initialized": False, "real_model_loaded": False,
              "scope": "Synthetic CPU parameter vectors and temporary workspace ledger files only"}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "tests", "failures", "errors", "seconds")}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
