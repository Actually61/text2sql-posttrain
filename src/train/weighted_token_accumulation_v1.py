"""Weighted assistant-token accumulation; caller owns zero_grad/clip/step."""
from __future__ import annotations

import math
import torch

from src.train.target_loss import target_position_loss
from src.train.token_accumulation import supervised_token_count


def backward_weighted_token_group(model, batches, weights, *, device=None,
                                  loss_fn=target_position_loss):
    """Accumulate sum(w_i * token_loss_sum_i) / sum(w_i * valid_count_i).

    Validate the whole group before backward. Do not divide by group size again.
    No optimizer updates, gradient clearing, clipping, or file writes occur here.
    """
    batches, weights = list(batches), list(weights)
    if not batches or len(batches) != len(weights):
        raise ValueError('Nonempty batches and one weight per microbatch required')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Weights must be finite positive numbers, not booleans')
    counts = [supervised_token_count(batch) for batch in batches]
    if any(batch['input_ids'].shape[0] != 1 for batch in batches):
        raise ValueError('One weight per example requires microbatch size one')
    denominator = math.fsum(w * n for w, n in zip(weights, counts))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Invalid weighted group denominator')
    coefficients = [w * n / denominator for w, n in zip(weights, counts)]
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    micro_losses = []
    for batch, count, coefficient in zip(batches, counts, coefficients):
        current = {key: value.to(target_device) for key, value in batch.items()}
        loss, metadata = loss_fn(model, current)
        if metadata['valid_target_count'] != count:
            raise ValueError('Loss target count differs from the group denominator')
        if loss.ndim != 0 or not loss.requires_grad or not torch.isfinite(loss).item():
            raise ValueError('Loss must be a finite differentiable scalar')
        micro_losses.append(float(loss.detach().item()))
        (loss * coefficient).backward()
        del loss, current
    return {
        'group_loss': math.fsum(loss * c for loss, c in zip(micro_losses, coefficients)),
        'micro_losses': micro_losses, 'micro_target_counts': counts,
        'microbatch_count': len(batches), 'weights': weights,
        'total_target_tokens': sum(counts), 'weighted_target_denominator': denominator,
        'micro_mean_loss_coefficients': coefficients,
        'optimizer_steps_inside_helper': 0,
        'normalization': 'weighted_valid_answer_tokens_per_actual_accumulation_group_v1',
    }
