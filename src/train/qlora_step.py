"""可逐行阅读的单batch更新；审计也可用于CPU小模型测试。"""
from contextlib import nullcontext
import math
import time

import torch


def adapter_parameters(model):
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not named or any(".lora_A." not in n and ".lora_B." not in n for n in named):
        raise ValueError("Only LoRA A/B parameters may be trainable")
    return named


def check_gradients(model):
    records = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise ValueError(f"Frozen base received a gradient: {name}")
            continue
        if p.grad is None or not torch.isfinite(p.grad).all().item():
            raise ValueError(f"Missing or nonfinite adapter gradient: {name}")
        norm = p.grad.detach().float().norm().item()
        if not math.isfinite(norm):
            raise ValueError(f"Gradient norm overflow: {name}")
        records.append({"name": name, "group": "A" if ".lora_A." in name else "B",
                        "norm": norm, "nonzero_elements": int(torch.count_nonzero(p.grad).item())})
    total = math.sqrt(sum(r["norm"] ** 2 for r in records))
    if not records or total == 0 or not math.isfinite(total):
        raise ValueError("No finite nonzero training signal")
    return {"all_finite": True, "global_l2_norm": total, "tensors": records}


def one_batch_update(model, batch, optimizer, stage_callback=lambda stage: None):
    """恰好一次forward/backward/step；不会重新计算loss或自动重试。"""
    trainable = adapter_parameters(model)
    expected = {id(p) for p in trainable.values()}
    actual_list = [p for group in optimizer.param_groups for p in group["params"]]
    if {id(p) for p in actual_list} != expected or len(actual_list) != len(expected):
        raise ValueError("Optimizer parameters must exactly equal the trainable adapters")
    before = {n: p.detach().float().cpu().clone() for n, p in trainable.items()}
    device = batch["input_ids"].device
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    timings = {}
    model.train()  # 开启训练行为；是否更新某个参数仍由requires_grad和optimizer决定。
    optimizer.zero_grad(set_to_none=True)  # 清除旧梯度，不清除权重。

    stage_callback("forward")
    sync()
    started = time.perf_counter()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with amp:
        loss = model(**batch, use_cache=False).loss  # labels已含-100，模型内部移位一次。
    sync()
    timings["forward_seconds"] = time.perf_counter() - started
    if loss.numel() != 1 or not torch.isfinite(loss).item() or not loss.requires_grad:
        raise ValueError("Loss must be a finite scalar connected to autograd")
    loss_value = loss.detach().float().item()

    stage_callback("backward")
    sync()
    started = time.perf_counter()
    loss.backward()  # 计算并累积梯度；到此还没更新参数。
    sync()
    timings["backward_seconds"] = time.perf_counter() - started
    gradients = check_gradients(model)  # 非有限或全零梯度直接停止，不执行step。

    stage_callback("optimizer_step")
    sync()
    started = time.perf_counter()
    optimizer.step()  # AdamW根据梯度更新LoRA参数，整个函数只有这一次step。
    sync()
    timings["optimizer_step_seconds"] = time.perf_counter() - started
    changes = []
    for name, p in trainable.items():
        after = p.detach().float().cpu()
        if not torch.isfinite(after).all().item():
            raise ValueError(f"Nonfinite parameter after update: {name}")
        delta = after - before[name]
        changes.append({"name": name, "group": "A" if ".lora_A." in name else "B",
                        "changed_elements": int(torch.count_nonzero(delta).item()),
                        "max_abs_delta": delta.abs().max().item(), "l2_delta": delta.norm().item()})
    if not any(r["changed_elements"] for r in changes):
        raise ValueError("Optimizer ran but no adapter value changed")
    optimizer.zero_grad(set_to_none=True)  # 为下一步清空梯度；本轮到此停止。
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Gradients were not cleared")
    return {"forward_calls": 1, "backward_calls": 1, "optimizer_steps": 1,
            "loss": loss_value, "loss_finite": True, "timings": timings,
            "gradients": gradients, "adapter_changes": changes,
            "gradients_cleared_after_step": True}
