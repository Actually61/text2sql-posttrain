"""CPU测试：完整训练题选样、单次更新上限和AdamW懒初始化状态；不加载Qwen。"""
import copy
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.train.long_probe_core import (
    SingleUpdateAdamW,
    optimizer_state_inventory,
    select_probe_rows,
)


def row(index, length, split="train"):
    ids = [7] * (length - 3) + [11, 151645, 198]
    return {"source_index": index, "project_split": split, "db_id": "synthetic",
            "sequence_length": length, "input_ids": ids, "attention_mask": [1] * length,
            "labels": [-100] * (length - 3) + [11, 151645, -100],
            "prompt_length": length - 3, "eos_position": length - 2,
            "supervised_token_count": 2, "trailing_template_token_count": 1}


class LongProbeCoreTests(unittest.TestCase):
    def test_normal_phase_sync_failure_is_saved_and_raised(self):
        from src.train.qlora_long_probe import PhaseMemory
        import json
        with tempfile.TemporaryDirectory() as folder:
            memory = PhaseMemory(Path(folder), SimpleNamespace(stage="forward"))
            memory.current, memory.started = "forward", time.perf_counter()
            with patch.object(torch.cuda, "synchronize", side_effect=RuntimeError("injected sync failure")):
                with self.assertRaisesRegex(RuntimeError, "injected sync failure"):
                    memory.end()
            saved = json.loads((Path(folder) / "phase_memory.json").read_text())
            self.assertTrue(saved[0]["failed"])
            self.assertEqual(saved[0]["phase"], "forward")
            self.assertIn("injected sync failure", saved[0]["diagnostic_error"])

    def test_nearest_selection_tie_breaks_by_source_id_and_preserves_full_records(self):
        # 故意让较小ID对应较长一侧，防止把较短长度误当作tie-break。
        rows = [row(9, 1023), row(3, 1025), row(8, 2047), row(2, 2049),
                row(7, 3761), row(1, 3761), row(99, 200)]
        original = copy.deepcopy(rows)
        chosen = select_probe_rows(rows)
        self.assertEqual([r["source_index"] for r in chosen], [3, 2, 1])
        self.assertEqual([r["sequence_length"] for r in chosen], [1025, 2049, 3761])
        self.assertEqual(chosen, select_probe_rows(list(reversed(rows))))
        self.assertEqual(rows, original)
        by_id = {r["source_index"]: r for r in original}
        for selected in chosen:
            self.assertEqual(selected, by_id[selected["source_index"]])
            self.assertEqual(len(selected["input_ids"]), selected["sequence_length"])
            self.assertEqual(len(selected["labels"]), selected["sequence_length"])

    def test_selection_rejects_nontraining_duplicate_ids_or_wrong_population_max(self):
        good = [row(1, 1024), row(2, 2048), row(3, 3761)]
        bad_cases = [[], good + [row(1, 100)], good + [row(99, 100, "validation")],
                     [row(1, 1024), row(2, 2048), row(3, 3760)],
                     good + [row(4, 3762)]]
        for rows in bad_cases:
            with self.subTest(lengths=[r["sequence_length"] for r in rows]):
                with self.assertRaises(ValueError):
                    select_probe_rows(rows)

    def test_adamw_state_is_empty_until_first_update_and_inventory_counts_real_tensors(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32))
        optimizer = SingleUpdateAdamW([parameter], lr=1e-3, weight_decay=0.0, foreach=False)
        before = optimizer_state_inventory(optimizer)
        self.assertEqual(before["cuda_bytes"], 0)
        self.assertEqual(before["cpu_bytes"], 0)
        self.assertEqual(before["tensor_count"], 0)
        self.assertEqual(len(optimizer.state), 0)
        parameter.square().sum().backward()
        optimizer.step()
        after = optimizer_state_inventory(optimizer)
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (1, 1))
        self.assertEqual(after["cuda_bytes"], 0)
        expected_bytes = sum(t.numel() * t.element_size()
                             for state in optimizer.state.values()
                             for t in state.values() if isinstance(t, torch.Tensor))
        self.assertEqual(after["cpu_bytes"], expected_bytes)
        self.assertEqual(after["tensor_count"], 3)  # step标量、一阶与二阶矩。
        for key in ("exp_avg", "exp_avg_sq"):
            moment = after["moments"][key]
            self.assertEqual(moment["tensor_count"], 1)
            self.assertEqual(moment["bytes"], parameter.numel() * parameter.element_size())
            self.assertEqual(moment["dtypes"], ["torch.float32"])
            self.assertEqual(moment["devices"], ["cpu"])

    def test_second_update_is_rejected_without_parameter_state_or_counter_changes(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = SingleUpdateAdamW([parameter], lr=1e-3, weight_decay=0.0, foreach=False)
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before_parameter = parameter.detach().clone()
        before_state = copy.deepcopy(optimizer.state[parameter])
        parameter.square().sum().backward()
        with self.assertRaises(RuntimeError):
            optimizer.step()
        self.assertTrue(torch.equal(parameter, before_parameter))
        self.assertEqual(set(optimizer.state[parameter]), set(before_state))
        for key, value in before_state.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(optimizer.state[parameter][key], value))
            else:
                self.assertEqual(optimizer.state[parameter][key], value)
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (1, 1))

    def test_failed_optimizer_attempt_is_counted_and_cannot_be_automatically_retried(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = SingleUpdateAdamW([parameter], lr=1e-3, weight_decay=0.0)
        parameter.square().backward()
        with patch.object(torch.optim.AdamW, "step", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                optimizer.step()
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (1, 0))
        with self.assertRaises(RuntimeError):
            optimizer.step()
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
