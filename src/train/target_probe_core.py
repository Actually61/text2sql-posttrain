"""独立两步实验的调用上限，保留部分更新可能性的持久化计数。"""
import torch


class TwoUpdateAdamW(torch.optim.AdamW):
    def __init__(self, params, audit_callback=lambda status: None, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.completed_steps = 0
        self.audit_callback = audit_callback

    def step(self, closure=None):
        if self.step_calls != self.completed_steps:
            raise RuntimeError("Previous optimizer attempt failed; no retry")
        if self.step_calls >= 2:
            raise RuntimeError("Two-update budget exhausted; no retry")
        self.step_calls += 1
        self.audit_callback({"step_calls": self.step_calls, "completed_steps": self.completed_steps,
                             "partial_update_possible": True})
        value = super().step(closure)
        self.completed_steps += 1
        self.audit_callback({"step_calls": self.step_calls, "completed_steps": self.completed_steps,
                             "partial_update_possible": False})
        return value
