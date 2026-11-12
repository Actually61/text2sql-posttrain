"""带观测/缓存释放边界的逐条梯度累积；不负责优化器更新。"""
from __future__ import annotations

import math

import torch

from src.train.target_loss import target_position_loss
from src.train.token_accumulation import supervised_token_count


def backward_cache_group(model, batches, *, device, loss_fn=target_position_loss,
                         observe=None, release=None):
    """沿用整组有效答案 token 归一化，仅在局部张量释放后调用 release。

    observe(stage, zero_based_micro_index, payload) 只能观测或抛异常停止。
    release(zero_based_micro_index) 只能释放未使用缓存，不清 .grad/Adam。
    回调异常直接传播，调用方不得用残余部分梯度继续 step。
    loss_fn 可以只包前向 autocast；backward 不处于本函数新建的 autocast 中。
    不传 release 时不释放缓存，方便 CPU 数值对照；真实试验必须注入实现。
    """
    batches = list(batches)
    if not batches:
        raise ValueError("Cannot accumulate an empty group")
    counts = [supervised_token_count(batch) for batch in batches]
    total_targets = sum(counts)
    target_device = torch.device(device)
    weighted_loss, micro_losses = 0.0, []

    def emit(stage, index, payload):
        if observe is not None:
            observe(stage, index, payload)

    for index, (batch, count) in enumerate(zip(batches, counts)):
        emit("micro_start", index, {})
        current = {key: value.to(target_device) for key, value in batch.items()}
        loss, metadata = loss_fn(model, current)
        if metadata["valid_target_count"] != count:
            raise ValueError("Loss target count differs from precomputed group denominator")
        if not torch.isfinite(loss).item() or not loss.requires_grad:
            raise ValueError("Loss must be finite and retain its autograd graph")
        weight = count / total_targets
        loss_number = float(loss.detach().item())
        emit("forward_end", index, {**metadata, "forward_loss": loss_number,
                                    "weight": weight, "total_target_tokens": total_targets})
        weighted_loss += loss_number * weight
        micro_losses.append(loss_number)
        (loss * weight).backward()
        del loss, current
        emit("backward_end", index, {})
        if release is not None:
            release(index)
        emit("micro_end", index, {})
    if not math.isfinite(weighted_loss):
        raise ValueError("Nonfinite accumulated loss")
    return {"group_loss": weighted_loss, "micro_losses": micro_losses,
            "micro_target_counts": counts, "total_target_tokens": total_targets,
            "microbatch_count": len(batches), "device": str(target_device),
            "normalization": "sum of token losses / total valid targets in actual group",
            "optimizer_steps_inside_helper": 0}
