"""CPU测试：选样/排程、有效token加权、无梯度loss模式；不加载Qwen或Spider。"""
from collections import Counter
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.train.stability_core import build_schedule, measure_subset_loss, select_short_subset, weighted_loss_summary


def feature(index, sql_tokens, rate):
    ids = [rate, 90] + [101] * sql_tokens + [151645, 198]
    return {"source_index": index, "project_split": "train", "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": [-100, -100] + ids[2:-1] + [-100],
            "prompt_length": 2, "eos_position": 2 + sql_tokens,
            "sequence_length": len(ids), "supervised_token_count": sql_tokens + 1,
            "trailing_template_token_count": 1}


class TokenLossModel(nn.Module):
    """按首个prompt token指定每个有效目标的损失，便于独立手算。"""
    def __init__(self, nan=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.nan = nan
        self.observations = []

    def forward(self, input_ids, attention_mask, labels, use_cache):
        valid = labels[:, 1:] != -100
        self.observations.append((self.training, torch.is_grad_enabled(), int(valid.sum())))
        if use_cache is not False:
            raise AssertionError("Training-subset loss must disable cache")
        if torch.any(labels[attention_mask == 0] != -100):
            raise AssertionError("Padding entered the loss")
        token_losses = input_ids[:, :1].float().expand_as(valid) * self.scale
        loss = token_losses[valid].mean()
        if self.nan:
            loss = loss * float("nan")
        return SimpleNamespace(loss=loss)


class StabilityCoreTests(unittest.TestCase):
    def test_weighted_loss_uses_tokens_not_equal_batch_weights(self):
        result = weighted_loss_summary([{"loss": 1.0, "supervised_tokens": 2},
                                        {"loss": 4.0, "supervised_tokens": 8}])
        self.assertEqual(result["total_negative_log_likelihood"], 34.0)
        self.assertEqual(result["supervised_tokens"], 10)
        self.assertEqual(result["weighted_loss"], 3.4)
        self.assertNotEqual(result["weighted_loss"], (1 + 4) / 2)

    def test_invalid_weighted_loss_inputs_are_rejected(self):
        bad = [[], [{"loss": 1.0, "supervised_tokens": 0}],
               [{"loss": float("nan"), "supervised_tokens": 2}],
               [{"loss": float("inf"), "supervised_tokens": 2}],
               [{"loss": -1.0, "supervised_tokens": 2}],
               [{"loss": 1.0, "supervised_tokens": True}]]
        for records in bad:
            with self.subTest(records=records), self.assertRaises(ValueError):
                weighted_loss_summary(records)

    def test_selection_is_deterministic_order_independent_and_nonmutating(self):
        rows = [{"source_index": i, "project_split": "train", "sequence_length": 256 if i < 36 else 513}
                for i in range(40)]
        original, rng = copy.deepcopy(rows), random.getstate()
        selected, eligible = select_short_subset(rows)
        reordered, _ = select_short_subset(list(reversed(rows)))
        self.assertEqual(eligible, 36)
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected, reordered)
        self.assertEqual(rows, original)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(len({r["source_index"] for r in selected}), 32)
        self.assertTrue(all(r["sequence_length"] <= 512 for r in selected))
        for bad in (rows[:10], rows + [rows[0]], [{**r, "project_split": "validation"} for r in rows]):
            with self.assertRaises(ValueError):
                select_short_subset(bad)

    def test_two_epoch_schedule_covers_each_example_exactly_twice(self):
        ids = list(range(100, 132))
        original = ids.copy()
        schedule = build_schedule(ids)
        self.assertEqual(schedule, build_schedule(list(reversed(ids))))
        self.assertEqual(ids, original)
        self.assertEqual([r["step"] for r in schedule], list(range(1, 33)))
        self.assertTrue(all(len(r["source_indices"]) == 2 for r in schedule))
        for epoch in (1, 2):
            observed = [i for r in schedule if r["epoch"] == epoch for i in r["source_indices"]]
            self.assertEqual(Counter(observed), Counter(ids))
        self.assertEqual(Counter(i for r in schedule for i in r["source_indices"]), Counter({i: 2 for i in ids}))
        with self.assertRaises(ValueError):
            build_schedule(ids, steps=33)

    def test_optimizer_refuses_33rd_update_without_changing_parameter_or_counters(self):
        from src.train.qlora_stability import BoundedAdamW
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = BoundedAdamW([parameter], lr=1e-3, weight_decay=0.0)
        for _ in range(32):
            optimizer.zero_grad(set_to_none=True)
            parameter.square().backward()
            optimizer.step()
        self.assertEqual(optimizer.step_calls, 32)
        self.assertEqual(optimizer.completed_steps, 32)
        self.assertLess(parameter.item(), 1.0)
        before = parameter.detach().clone()
        before_state_step = optimizer.state[parameter]["step"].clone()
        optimizer.zero_grad(set_to_none=True)
        parameter.square().backward()
        with self.assertRaises(RuntimeError):
            optimizer.step()
        self.assertTrue(torch.equal(parameter, before))
        self.assertTrue(torch.equal(optimizer.state[parameter]["step"], before_state_step))
        self.assertEqual((optimizer.step_calls, optimizer.completed_steps), (32, 32))

    def test_measure_loss_counts_shifted_targets_and_restores_training_mode(self):
        rows = [feature(0, 1, 1), feature(1, 2, 3), feature(2, 3, 5)]
        original = copy.deepcopy(rows)
        model = TokenLossModel()
        result = measure_subset_loss(model, rows, "cpu")
        self.assertAlmostEqual(result["weighted_loss"], 31 / 9, places=6)
        self.assertEqual(result["supervised_tokens"], 9)
        self.assertEqual(result["batch_count"], 2)
        self.assertEqual(model.observations, [(False, False, 5), (False, False, 4)])
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)
        self.assertEqual(rows, original)
        self.assertEqual(result["gradient_updates"], 0)

    def test_measure_loss_restores_mode_on_failure_and_preserves_eval_mode(self):
        rows = [feature(0, 1, 1)]
        broken = TokenLossModel(nan=True)
        with self.assertRaisesRegex(ValueError, "Invalid no-grad"):
            measure_subset_loss(broken, rows, "cpu")
        self.assertTrue(broken.training)
        self.assertIsNone(broken.scale.grad)
        model = TokenLossModel().eval()
        measure_subset_loss(model, rows, "cpu")
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
