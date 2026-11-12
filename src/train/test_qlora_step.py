"""CPU微型LoRA测试：检验真实autograd与更新保护，不加载Qwen、GPU或Spider。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.train.qlora_step import one_batch_update


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False)
        self.lora_A = nn.Linear(3, 1, bias=False)
        self.lora_B = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]))
            self.lora_A.weight.copy_(torch.tensor([[0.2, -0.1, 0.3]]))
            self.lora_B.weight.zero_()
        self.base.weight.requires_grad_(False)

    def forward(self, values):
        return self.base(values) + self.lora_B(self.lora_A(values))


class TinyModel(nn.Module):
    def __init__(self, mode="valid"):
        super().__init__()
        self.projection = TinyLoRA()
        self.mode = mode
        self.forward_calls = 0
        self.backward_calls = 0

    def forward(self, input_ids, attention_mask, labels, use_cache):
        self.forward_calls += 1
        if use_cache is not False:
            raise AssertionError("Training must disable cache")
        loss = (self.projection(input_ids) - labels).square().mean()
        if self.mode == "nan_loss":
            loss = loss * float("nan")
        elif self.mode == "zero_signal":
            loss = loss * 0
        def count_backward(gradient):
            self.backward_calls += 1
            return gradient
        loss.register_hook(count_backward)
        return SimpleNamespace(loss=loss)


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr=lr, weight_decay=0.0, foreach=False)
        self.step_calls = 0
        self.clear_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.clear_calls += 1
        return super().zero_grad(set_to_none=set_to_none)


class OneBatchUpdateTests(unittest.TestCase):
    def setUp(self):
        # 浮点小输入仅测试训练循环；真实token与assistant mask由既有SFT测试覆盖。
        self.batch = {"input_ids": torch.tensor([[1., 2., 3.], [2., 1., 1.]]),
                      "attention_mask": torch.ones((2, 3), dtype=torch.long),
                      "labels": torch.tensor([[1., -1.], [-1., 1.]])}

    def make(self, mode="valid", lr=1e-3):
        model = TinyModel(mode)
        optimizer = CountingAdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        return model, optimizer

    def snapshot(self, model):
        return {n: p.detach().clone() for n, p in model.named_parameters()}

    def assert_unchanged(self, model, previous):
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(previous[name], parameter), name)

    def test_exactly_one_update_and_default_lora_first_step(self):
        model, optimizer = self.make()
        previous = self.snapshot(model)
        # 放入旧梯度，确认循环前的zero_grad确实清除它们。
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        stages = []
        result = one_batch_update(model, self.batch, optimizer, stages.append)
        self.assertEqual((model.forward_calls, model.backward_calls, optimizer.step_calls), (1, 1, 1))
        self.assertEqual(optimizer.clear_calls, 2)
        self.assertEqual(stages, ["forward", "backward", "optimizer_step"])
        self.assertTrue(result["loss_finite"])
        self.assertGreater(result["gradients"]["global_l2_norm"], 0)
        gradients = {r["group"]: r for r in result["gradients"]["tensors"]}
        changes = {r["group"]: r for r in result["adapter_changes"]}
        self.assertEqual(gradients["A"]["norm"], 0)
        self.assertEqual(changes["A"]["changed_elements"], 0)
        self.assertGreater(gradients["B"]["nonzero_elements"], 0)
        self.assertGreater(changes["B"]["changed_elements"], 0)
        self.assertTrue(torch.equal(previous["projection.base.weight"], model.projection.base.weight))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_nonfinite_loss_stops_before_backward_or_step(self):
        model, optimizer = self.make("nan_loss")
        previous = self.snapshot(model)
        with self.assertRaisesRegex(ValueError, "finite scalar"):
            one_batch_update(model, self.batch, optimizer)
        self.assertEqual((model.forward_calls, model.backward_calls, optimizer.step_calls), (1, 0, 0))
        self.assert_unchanged(model, previous)

    def test_nonfinite_gradient_stops_before_step(self):
        model, optimizer = self.make()
        previous = self.snapshot(model)
        model.projection.lora_B.weight.register_hook(lambda gradient: gradient * float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite adapter gradient"):
            one_batch_update(model, self.batch, optimizer)
        self.assertEqual((model.backward_calls, optimizer.step_calls), (1, 0))
        self.assert_unchanged(model, previous)

    def test_zero_training_signal_does_not_update(self):
        model, optimizer = self.make("zero_signal")
        previous = self.snapshot(model)
        with self.assertRaisesRegex(ValueError, "No finite nonzero training signal"):
            one_batch_update(model, self.batch, optimizer)
        self.assertEqual(optimizer.step_calls, 0)
        self.assert_unchanged(model, previous)

    def test_optimizer_containing_frozen_base_is_rejected(self):
        model, _ = self.make()
        optimizer = CountingAdamW(list(model.parameters()))
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            one_batch_update(model, self.batch, optimizer)
        self.assertEqual((model.forward_calls, optimizer.step_calls), (0, 0))

    def test_unfrozen_base_is_rejected(self):
        model, _ = self.make()
        model.projection.base.weight.requires_grad_(True)
        optimizer = CountingAdamW(list(model.parameters()))
        with self.assertRaisesRegex(ValueError, "Only LoRA"):
            one_batch_update(model, self.batch, optimizer)
        self.assertEqual((model.forward_calls, optimizer.step_calls), (0, 0))

    def test_no_parameter_change_is_not_reported_as_success(self):
        model, optimizer = self.make(lr=0)
        previous = self.snapshot(model)
        with self.assertRaisesRegex(ValueError, "no adapter value changed"):
            one_batch_update(model, self.batch, optimizer)
        self.assertEqual(optimizer.step_calls, 1)
        self.assert_unchanged(model, previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
