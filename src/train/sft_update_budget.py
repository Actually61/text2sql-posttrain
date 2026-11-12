"""跨进程更新账本：已经尝试但未完成的更新不能自动重放。"""
import json
from pathlib import Path
import torch


def read_ledger(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_atomic(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    pending.replace(path)


class JournaledAdamW(torch.optim.AdamW):
    def __init__(self, params, *, ledger_path, start, segment_end, total_budget, **kwargs):
        super().__init__(params, **kwargs)
        self.ledger_path = Path(ledger_path)
        self.completed_updates = self.step_calls = start
        self.segment_end, self.total_budget = segment_end, total_budget
        if not 0 <= start < segment_end <= total_budget:
            raise ValueError("Invalid segment/global update budget")
        self._check_ledger()

    def _check_ledger(self):
        record = read_ledger(self.ledger_path)
        if (record["attempted_updates"] != self.step_calls or record["completed_updates"] != self.completed_updates
                or record["attempted_updates"] != record["completed_updates"]
                or record.get("failed") or record.get("partial_update_possible")
                or record["total_budget"] != self.total_budget):
            raise RuntimeError("Ledger mismatch or previous failure; never reset/replay it")

    def step(self, closure=None):
        self._check_ledger()
        if self.step_calls >= self.segment_end or self.step_calls >= self.total_budget:
            raise RuntimeError("Authorized update budget exhausted")
        self.step_calls += 1
        status = {"attempted_updates": self.step_calls, "completed_updates": self.completed_updates,
                  "total_budget": self.total_budget, "partial_update_possible": True, "failed": False}
        write_atomic(self.ledger_path, status)
        result = super().step(closure)
        self.completed_updates += 1
        status.update(completed_updates=self.completed_updates, partial_update_possible=False)
        write_atomic(self.ledger_path, status)
        return result
