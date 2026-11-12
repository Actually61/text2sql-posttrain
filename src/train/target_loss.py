"""完整因果上下文不变，只在有效下一token目标对应的位置计算词表logits。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def target_position_loss(model, batch):
    """返回带求导图的标量loss及只含普通数值的元数据。

    本函数针对已准备的三字段SFT batch，不接收已移位labels。
    batch>1时取各行有效预测位置的列并集；某些保留位置在另一行仍为-100。
    """
    required = {"input_ids", "attention_mask", "labels"}
    if set(batch) != required:
        raise ValueError("Expected exactly input_ids, attention_mask and unshifted labels")
    input_ids, attention_mask, labels = (batch[k] for k in ("input_ids", "attention_mask", "labels"))
    if any(not isinstance(t, torch.Tensor) or t.ndim != 2 for t in (input_ids, attention_mask, labels)):
        raise ValueError("All batch fields must be rank-two tensors")
    if input_ids.shape != labels.shape or input_ids.shape != attention_mask.shape:
        raise ValueError("Batch fields must have identical shapes")
    if input_ids.shape[0] < 1 or input_ids.shape[1] < 2:
        raise ValueError("Require a nonempty batch with at least two sequence positions")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError("Token IDs and labels must remain torch.long")
    if any(t.device != input_ids.device for t in (attention_mask, labels)):
        raise ValueError("All batch fields must be on the same device")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all().item()):
        raise ValueError("Attention mask must contain only zero or one")
    if bool(((labels != -100) & (attention_mask == 0)).any().item()):
        raise ValueError("Padding targets must already be ignored")

    # 与当前Transformers ForCausalLMLoss一致：位置t预测labels[t+1]，末尾补-100。
    shift_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
    valid = shift_labels != -100
    positions = valid.any(dim=0).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("No valid next-token targets; do not return NaN loss")
    targets = shift_labels.index_select(1, positions)
    vocab_size = model.config.vocab_size
    selected_valid = targets[targets != -100]
    if bool(((selected_valid < 0) | (selected_valid >= vocab_size)).any().item()):
        raise ValueError("Supervised token ID outside model vocabulary")

    # 不裁剪input_ids或attention_mask；只让Qwen在指定hidden-state位置执行lm_head。
    output = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=False, logits_to_keep=positions)
    logits = output.logits
    expected_shape = (input_ids.shape[0], positions.numel(), vocab_size)
    if tuple(logits.shape) != expected_shape:
        raise ValueError("Model did not honor logits_to_keep positions")
    # 不传labels给模型，避免它对已选列再次隐式移位；在此显式对齐目标一次。
    loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1),
                           ignore_index=-100, reduction="mean")
    valid_count = int(valid.sum().item())
    metadata = {
        "input_shape": list(input_ids.shape),
        "full_logit_shape": [input_ids.shape[0], input_ids.shape[1], vocab_size],
        "logits_shape": list(logits.shape),
        "kept_position_count": int(positions.numel()),
        "valid_target_count": valid_count,
        "first_predict_position": int(positions[0].item()),
        "last_predict_position": int(positions[-1].item()),
        "retained_ignored_cells": int(targets.numel()) - valid_count,
        "loss_dtype": str(loss.dtype),
        "input_was_truncated": False,
    }
    return loss, metadata
