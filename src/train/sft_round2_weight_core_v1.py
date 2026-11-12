"""Weighted answer-token accumulation with the verified cache-release boundaries.

This helper owns no optimizer, ledger, checkpoint, CUDA initialization, or retry.
The caller supplies frozen group metadata and handles one update after completion.
"""
from __future__ import annotations

import math

import torch

from src.train.target_loss import target_position_loss
from src.train.token_accumulation import supervised_token_count


def backward_weighted_cache_group(model, batches, *, rows, group, device,
                                  loss_fn=target_position_loss, observe=None, release=None):
    """Accumulate sum(w_i * answer_loss_sum_i) / sum(w_i * answer_token_count_i).

    `rows` supplies source_index and supervised_token_count for every microbatch.
    `group` binds their order, target counts, integer 1/2 weights and denominator.
    Validate the entire group before any forward or callback. Callback failures
    propagate immediately; a partially accumulated group must never be stepped.

    Hook order remains micro_start -> forward_end -> backward_end -> release ->
    micro_end. release runs only after local loss/input references are deleted;
    the caller's implementation must preserve accumulated gradients and Adam.
    """
    batches, rows = list(batches), list(rows)
    if not batches or len(rows) != len(batches) or not isinstance(group, dict):
        raise ValueError('Require a nonempty group with one metadata row per microbatch')
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Every training row must be a metadata dictionary')
    source_indices = [row.get('source_index') for row in rows]
    if (any(type(index) is not int or index < 0 for index in source_indices)
            or len(set(source_indices)) != len(source_indices)
            or group.get('source_indices') != source_indices
            or any(type(index) is not int for index in group.get('source_indices', []))):
        raise ValueError('Frozen group source IDs/order differ from the supplied rows')
    counts = [supervised_token_count(batch) for batch in batches]
    if any(batch['input_ids'].shape[0] != 1 for batch in batches):
        raise ValueError('One sample weight requires microbatch size one')
    row_counts = [row.get('supervised_token_count') for row in rows]
    saved_counts = group.get('target_counts')
    if (any(type(count) is not int for count in row_counts) or row_counts != counts
            or not isinstance(saved_counts, list)
            or any(type(count) is not int for count in saved_counts) or saved_counts != counts):
        raise ValueError('Actual answer-token counts differ from row/group metadata')
    weights = group.get('loss_weights')
    if (not isinstance(weights, list) or len(weights) != len(batches)
            or any(type(weight) is not int or weight not in (1, 2) for weight in weights)):
        raise ValueError('Frozen group must supply one integer 1/2 weight per example')
    weights = list(weights)
    total_targets = sum(counts)
    denominator = sum(weight * count for weight, count in zip(weights, counts))
    if (type(group.get('group_target_tokens')) is not int
            or group['group_target_tokens'] != total_targets
            or type(group.get('weighted_target_denominator')) is not int
            or group['weighted_target_denominator'] != denominator):
        raise ValueError('Frozen unweighted/weighted group denominator changed')
    coefficients = [weight * count / denominator for weight, count in zip(weights, counts)]
    saved_coefficients = group.get('micro_mean_loss_coefficients')
    if (not isinstance(saved_coefficients, list) or len(saved_coefficients) != len(coefficients)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
                   for value, expected in zip(saved_coefficients, coefficients))):
        raise ValueError('Frozen microbatch loss coefficients changed')
    target_device = torch.device(device)
    micro_losses = []

    def emit(stage, index, payload):
        if observe is not None:
            observe(stage, index, payload)

    for index, (batch, count, sample_weight, coefficient) in enumerate(
            zip(batches, counts, weights, coefficients)):
        emit('micro_start', index, {})
        current = {key: value.to(target_device) for key, value in batch.items()}
        loss, metadata = loss_fn(model, current)
        if metadata['valid_target_count'] != count:
            raise ValueError('Loss target count differs from the frozen group denominator')
        if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
                or not loss.requires_grad or not torch.isfinite(loss).item()):
            raise ValueError('Loss must be a finite differentiable scalar')
        loss_number = float(loss.detach().item())
        # `weight` retains the legacy event meaning: coefficient of micro-mean loss.
        # Source IDs are injected by the worker observer to avoid duplicate kwargs.
        emit('forward_end', index, {**metadata, 'forward_loss': loss_number,
            'weight': coefficient, 'sample_weight': sample_weight,
            'micro_mean_loss_coefficient': coefficient, 'total_target_tokens': total_targets,
            'weighted_target_denominator': denominator})
        micro_losses.append(loss_number)
        (loss * coefficient).backward()
        del loss, current
        emit('backward_end', index, {})
        if release is not None:
            release(index)
        emit('micro_end', index, {})
    group_loss = math.fsum(loss * coefficient for loss, coefficient in zip(micro_losses, coefficients))
    if not math.isfinite(group_loss):
        raise ValueError('Nonfinite accumulated weighted loss')
    return {
        'group_loss': group_loss, 'micro_losses': micro_losses,
        'source_indices': source_indices, 'weights': weights, 'loss_weights': weights,
        'micro_target_counts': counts, 'total_target_tokens': total_targets,
        'weighted_target_denominator': denominator,
        'micro_mean_loss_coefficients': coefficients,
        'microbatch_count': len(batches), 'device': str(target_device),
        'normalization': 'weighted_valid_answer_tokens_per_actual_accumulation_group_v1',
        'optimizer_steps_inside_helper': 0,
    }
