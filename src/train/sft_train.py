"""正式SFT的分段训练循环；本次由独立pilot控制器限定为2+2次更新。"""
from __future__ import annotations
import json
import torch

from src.data.sft_features import collate_sft_batch
from src.infer.baseline_pilot import save_json
from src.train.qlora_step import adapter_parameters, check_gradients
from src.train.long_probe_core import optimizer_state_inventory
from src.train.sft_round1_plan import learning_rate_for_update
from src.train.target_loss import target_position_loss
from src.train.token_accumulation import backward_token_normalized_group


def adam_steps(optimizer):
    return sorted({float(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def set_next_lr(optimizer, completed_updates, training):
    schedule = training["scheduler"]
    value = learning_rate_for_update(completed_updates + 1, schedule["total_updates"],
                                     schedule["warmup_updates"], training["optimizer"]["lr"])
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = value
    return value


def train_segment(model, optimizer, rows_by_id, groups, start, end, training,
                  directory, phases, check_memory, save_boundary):
    """Consume each scheduled example once; each actual group has one token-normalized update."""
    named = adapter_parameters(model)
    parameter_bytes = sum(p.numel() * p.element_size() for p in named.values())
    records = []
    model.train()
    for group_index in range(start, end):
        update = group_index + 1
        group = groups[group_index]
        rows = [rows_by_id[index] for index in group["source_indices"]]
        batches = [collate_sft_batch([row], 151643, max_length=4096) for row in rows]
        expected_targets = sum(row["supervised_token_count"] for row in rows)
        if expected_targets != group["group_target_tokens"]:
            raise ValueError("Frozen accumulation denominator changed")
        phases.begin(f"update_{update}_setup")
        optimizer.zero_grad(set_to_none=True)
        before = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
        before_state = optimizer_state_inventory(optimizer)
        if (before_state["cuda_bytes"] != (0 if group_index == 0 else 2 * parameter_bytes)
                or adam_steps(optimizer) != ([] if group_index == 0 else [float(group_index)])):
            raise ValueError("Persistent Adam state/cursor mismatch before update")
        lr = set_next_lr(optimizer, group_index, training)
        phases.end()
        check_memory()
        micro_records = []

        def forward_loss(current_model, batch):
            # Close the preceding microbatch only after its backward has finished.
            phases.end()
            check_memory()
            micro_index = len(micro_records)
            row = rows[micro_index]
            phases.begin(f"update_{update}_micro_{micro_index + 1}_L{row['sequence_length']}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metadata = target_position_loss(current_model, batch)
            if (metadata["input_shape"] != [1, row["sequence_length"]]
                    or metadata["valid_target_count"] != row["supervised_token_count"]):
                raise ValueError("Full input or assistant-only targets changed")
            record = {"update": update, "microbatch": micro_index + 1, "source_index": row["source_index"],
                      "forward_loss": float(loss.detach().item()), **metadata}
            micro_records.append(record)
            with (directory / "micro_forwards.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            return loss, metadata

        # This helper was tested against a direct CPU batch, including one AdamW update.
        group_result = backward_token_normalized_group(model, batches, device="cuda:0", loss_fn=forward_loss)
        phases.end()
        check_memory()
        if len(micro_records) != len(rows) or group_result["total_target_tokens"] != expected_targets:
            raise ValueError("Missing microbatch or wrong accumulation denominator")
        phases.begin(f"update_{update}_gradient_and_clip")
        gradients = check_gradients(model)
        save_json(directory / f"update_{update}_gradients.json", gradients)
        before_clip = torch.nn.utils.clip_grad_norm_(list(named.values()), training["max_gradient_norm"],
                                                    error_if_nonfinite=True)
        after_clip = check_gradients(model)["global_l2_norm"]
        if after_clip > training["max_gradient_norm"] + 1e-5:
            raise ValueError("Gradient clipping did not enforce the norm budget")
        phases.end()
        check_memory()

        phases.begin(f"update_{update}_optimizer_step")
        optimizer.step()
        phases.end()
        check_memory()
        phases.begin(f"update_{update}_audit")
        if optimizer.completed_updates != update:
            raise ValueError("Unexpected global optimizer update count")
        after_state = optimizer_state_inventory(optimizer)
        if (after_state["cuda_bytes"] != 2 * parameter_bytes or after_state["parameter_states"] != len(named)
                or adam_steps(optimizer) != [float(update)]):
            raise ValueError("Adam state reset, incomplete, or has the wrong step")
        changes = []
        for name, parameter in named.items():
            value = parameter.detach().cpu()
            if not torch.isfinite(value).all().item():
                raise ValueError("Adapter contains nonfinite values")
            changes.append({"name": name, "changed_elements": int(torch.count_nonzero(value - before[name]))})
        if not any(item["changed_elements"] for item in changes):
            raise ValueError("No adapter parameter changed")
        save_json(directory / f"update_{update}_adapter_changes.json", changes)
        next_lr = set_next_lr(optimizer, update, training)
        optimizer.zero_grad(set_to_none=True)
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Gradient cleanup failed")
        record = {"update": update, "group_index": group_index, "source_indices": group["source_indices"],
                  **group_result, "lr_used": lr, "next_update_lr": next_lr,
                  "gradient_norm_before_clip": float(before_clip.item()), "gradient_norm_after_clip": after_clip,
                  "gradients_all_finite": True, "optimizer_cuda_mib_before": before_state["cuda_bytes"] / 2**20,
                  "optimizer_cuda_mib_after": after_state["cuda_bytes"] / 2**20,
                  "a_changed_tensors": sum(c["changed_elements"] > 0 for c in changes if ".lora_A." in c["name"]),
                  "b_changed_tensors": sum(c["changed_elements"] > 0 for c in changes if ".lora_B." in c["name"])}
        records.append(record)
        save_json(directory / "history.json", records)
        del before, changes, batches
        phases.end()
        check_memory()
        save_boundary(update)
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)
    return records
