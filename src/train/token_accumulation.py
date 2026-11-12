"""按整个累积组的有效答案 token 数归一化；不在 microbatch 之间更新参数。"""
from __future__ import annotations

import math

import torch

from src.train.target_loss import target_position_loss


def supervised_token_count(batch):
    """CPU 预检或组开始时计数；未移位 labels 的第一列不属于下一 token 目标。"""
    if set(batch) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Expected exactly the three prepared SFT fields")
    ids, mask, labels = (batch[key] for key in ("input_ids", "attention_mask", "labels"))
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in (ids, mask, labels)):
        raise ValueError("Prepared batch fields must be rank-two tensors")
    if ids.shape != mask.shape or ids.shape != labels.shape or ids.shape[0] < 1 or ids.shape[1] < 2:
        raise ValueError("Prepared batch shapes must match and be nonempty")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("Input IDs and labels must be torch.long")
    if any(value.device != ids.device for value in (mask, labels)):
        raise ValueError("Prepared batch fields must share a device")
    if not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("Attention mask must contain only zero or one")
    if bool(((labels != -100) & (mask == 0)).any().item()):
        raise ValueError("Padding labels must be ignored")
    count = int((labels[:, 1:] != -100).sum().item())
    if count == 0:
        raise ValueError("Each microbatch must have valid next-token targets")
    return count


def backward_token_normalized_group(model, batches, *, device=None, loss_fn=target_position_loss):
    """返回普通数值日志，梯度留在参数上供调用方组末 clip/step。

    batches 通常是 CPU tensor 字典列表，只有当前 microbatch 会搬到 device。
    调用方必须组前 zero_grad；本函数不清梯度、不裁剪、不更新参数。
    只在完成整个组后调用一次 clip_grad_norm_ / optimizer.step / scheduler.step。
    组中报错应由调用方停止，不能用残留的部分梯度直接更新。
    """
    batches = list(batches)
    if not batches:
        raise ValueError("Cannot accumulate an empty group")
    # 先对整组计数，包括不足额的尾组；不是固定除以 nominal accumulation steps。
    counts = [supervised_token_count(batch) for batch in batches]
    total_targets = sum(counts)
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    weighted_loss = 0.0
    micro_losses = []
    for batch, count in zip(batches, counts):
        current = {key: value.to(target_device) for key, value in batch.items()}
        loss, metadata = loss_fn(model, current)
        if metadata["valid_target_count"] != count:
            raise ValueError("Loss target count differs from precomputed group denominator")
        if not torch.isfinite(loss).item() or not loss.requires_grad:
            raise ValueError("Loss must be finite and retain its autograd graph")
        weight = count / total_targets
        loss_number = float(loss.detach().item())
        weighted_loss += loss_number * weight
        micro_losses.append(loss_number)
        (loss * weight).backward()
        del loss, current  # 下一条样本开始前，不保留本条的求导图。
    if not math.isfinite(weighted_loss):
        raise ValueError("Nonfinite accumulated loss")
    return {"group_loss": weighted_loss, "micro_losses": micro_losses,
            "micro_target_counts": counts, "total_target_tokens": total_targets,
            "microbatch_count": len(batches), "device": str(target_device),
            "normalization": "sum of token losses / total valid targets in actual group",
            "optimizer_steps_inside_helper": 0}
