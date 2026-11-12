"""长样本选样、一次更新上限和优化器状态统计；不修改正式数据。"""
import torch


def select_probe_rows(rows):
    ids = [r["source_index"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or any(r["project_split"] != "train" for r in rows):
        raise ValueError("Require nonempty unique training examples")
    if max(r["sequence_length"] for r in rows) != 3761:
        raise ValueError("Frozen longest training length must be 3761")
    selected = [min(rows, key=lambda r: (abs(r["sequence_length"] - target), r["source_index"]))
                for target in (1024, 2048, 3761)]
    if len({r["source_index"] for r in selected}) != 3:
        raise ValueError("Three tiers must use distinct examples")
    return selected


class SingleUpdateAdamW(torch.optim.AdamW):
    """先记录调用，再执行更新；step中途失败不等于完全没有改过参数。"""
    def __init__(self, params, audit_callback=lambda status: None, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.completed_steps = 0
        self.audit_callback = audit_callback

    def step(self, closure=None):
        if self.step_calls >= 1:
            raise RuntimeError("One-update limit reached; no retry")
        self.step_calls += 1
        self.audit_callback({"step_calls": self.step_calls, "completed_steps": self.completed_steps,
                             "partial_update_possible": True})
        result = super().step(closure)
        self.completed_steps += 1
        self.audit_callback({"step_calls": self.step_calls, "completed_steps": self.completed_steps,
                             "partial_update_possible": False})
        return result


def optimizer_state_inventory(optimizer):
    """按真实dtype/device统计状态张量字节；CPU的step标量不算GPU占用。"""
    result = {"cuda_bytes": 0, "cpu_bytes": 0, "tensor_count": 0, "parameter_states": len(optimizer.state),
              "moments": {k: {"tensor_count": 0, "bytes": 0, "dtypes": [], "devices": []}
                          for k in ("exp_avg", "exp_avg_sq")}}
    for state in optimizer.state.values():
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            size = value.numel() * value.element_size()
            result["tensor_count"] += 1
            key = "cuda_bytes" if value.device.type == "cuda" else "cpu_bytes"
            result[key] += size
            if name in result["moments"]:
                item = result["moments"][name]
                item["tensor_count"] += 1
                item["bytes"] += size
                item["dtypes"].append(str(value.dtype))
                item["devices"].append(str(value.device))
    for item in result["moments"].values():
        item["dtypes"] = sorted(set(item["dtypes"]))
        item["devices"] = sorted(set(item["devices"]))
    return result
