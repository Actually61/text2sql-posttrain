"""CPU 小 Qwen：直接 batch 对照 token 归一化累积的 loss、梯度和一次 AdamW 更新。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import numpy as np

from src.train.test_target_loss import digest, make_batch, make_model
from src.train.token_accumulation import backward_token_normalized_group
from src.train.target_loss import target_position_loss

REPORT = ROOT / "reports/token_accumulation_cpu_checks.json"
ATOL, RTOL = 1e-6, 1e-5  # 执行前固定，失败不得放宽。
OPTIMIZER_KWARGS = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
                    "weight_decay": 0.0, "foreach": False}


def microbatches(full_batch):
    """从同一直接 batch 取出每行真实序列；移除的只有对照用右 padding。"""
    batches = []
    for index in range(full_batch["input_ids"].shape[0]):
        length = int(full_batch["attention_mask"][index].sum().item())
        if not bool(full_batch["attention_mask"][index, :length].all().item()):
            raise AssertionError("This fixture requires right padding")
        batches.append({key: value[index:index + 1, :length].clone() for key, value in full_batch.items()})
    return batches


def named_trainable(model):
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def compare_tensors(reference, accumulated, field):
    maximum, worst, count = 0.0, None, 0
    for name in reference:
        left, right = reference[name], accumulated[name]
        if left is None or right is None or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f"Missing/nonfinite {field}: {name}")
        torch.testing.assert_close(right, left, atol=ATOL, rtol=RTOL,
                                   msg=lambda message: f"{field}/{name}: {message}")
        error = float((right - left).abs().max().item())
        if error > maximum:
            maximum, worst = error, name
        count += 1
    return {"max_absolute_difference": maximum, "worst_parameter": worst, "tensors_compared": count}


def compare_case(case):
    full_batch = make_batch(case["specifications"])
    if case.get("mask_hole"):
        full_batch["labels"][0, case["mask_hole"]] = -100
    original_batch = {key: value.clone() for key, value in full_batch.items()}
    prototype = make_model(case["seed"], tied=case.get("tied", False), lora=case.get("lora", False))
    prototype_optimizer = torch.optim.AdamW(list(named_trainable(prototype).values()), **OPTIMIZER_KWARGS)
    if case.get("existing_adam_state"):
        prototype_optimizer.zero_grad(set_to_none=True)
        prototype(**full_batch, use_cache=False).loss.backward()
        torch.nn.utils.clip_grad_norm_(list(named_trainable(prototype).values()), 1.0, error_if_nonfinite=True)
        prototype_optimizer.step()
        prototype_optimizer.zero_grad(set_to_none=True)
    reference, accumulated = copy.deepcopy(prototype), copy.deepcopy(prototype)
    ref_params, acc_params = named_trainable(reference), named_trainable(accumulated)
    initial_params = {name: parameter.detach().clone() for name, parameter in ref_params.items()}
    reference_optimizer = torch.optim.AdamW(list(ref_params.values()), **OPTIMIZER_KWARGS)
    accumulated_optimizer = torch.optim.AdamW(list(acc_params.values()), **OPTIMIZER_KWARGS)
    reference_optimizer.load_state_dict(copy.deepcopy(prototype_optimizer.state_dict()))
    accumulated_optimizer.load_state_dict(copy.deepcopy(prototype_optimizer.state_dict()))
    reference_optimizer.zero_grad(set_to_none=True)
    accumulated_optimizer.zero_grad(set_to_none=True)

    loss_reference = reference(**full_batch, use_cache=False).loss
    loss_reference.backward()
    # 组中任何 optimizer.step/zero_grad 调用都会使测试失败。
    with patch.object(accumulated_optimizer, "step", side_effect=AssertionError("No step inside group")), \
         patch.object(accumulated_optimizer, "zero_grad", side_effect=AssertionError("No zero_grad inside group")):
        group = backward_token_normalized_group(accumulated, microbatches(full_batch), device="cpu")
    torch.testing.assert_close(torch.tensor(group["group_loss"]), loss_reference.detach(), atol=ATOL, rtol=RTOL)
    for name, parameter in acc_params.items():
        torch.testing.assert_close(parameter.detach(), initial_params[name], atol=0, rtol=0)
    gradients = compare_tensors({name: p.grad for name, p in ref_params.items()},
                               {name: p.grad for name, p in acc_params.items()}, "gradient_before_clip")
    for model in (reference, accumulated):
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError("Frozen parameters received gradients")
    nonzero_a = sum(bool(torch.count_nonzero(p.grad).item()) for name, p in acc_params.items() if ".lora_A." in name)
    if case.get("lora") and nonzero_a == 0:
        raise AssertionError("LoRA case must exercise nonzero A gradients")

    norm_reference = torch.nn.utils.clip_grad_norm_(list(ref_params.values()), 1.0, error_if_nonfinite=True)
    norm_accumulated = torch.nn.utils.clip_grad_norm_(list(acc_params.values()), 1.0, error_if_nonfinite=True)
    torch.testing.assert_close(norm_accumulated, norm_reference, atol=ATOL, rtol=RTOL)
    clipped = compare_tensors({name: p.grad for name, p in ref_params.items()},
                             {name: p.grad for name, p in acc_params.items()}, "gradient_after_clip")
    reference_optimizer.step()
    accumulated_optimizer.step()
    updates = compare_tensors({name: p.detach() for name, p in ref_params.items()},
                             {name: p.detach() for name, p in acc_params.items()}, "parameters_after_step")
    changed = sum(int(torch.count_nonzero(p.detach() - initial_params[name])) for name, p in acc_params.items())
    if changed == 0:
        raise AssertionError("No parameter changed after AdamW step")
    # 一次更新后的 Adam moments 和 step 也必须对应，而不只是参数碰巧接近。
    for ref_parameter, acc_parameter in zip(ref_params.values(), acc_params.values()):
        ref_state, acc_state = reference_optimizer.state[ref_parameter], accumulated_optimizer.state[acc_parameter]
        for key in ref_state:
            torch.testing.assert_close(acc_state[key], ref_state[key], atol=ATOL, rtol=RTOL)
    for key in original_batch:
        torch.testing.assert_close(full_batch[key], original_batch[key], atol=0, rtol=0)
    state_steps = sorted({float(state["step"].item()) for state in accumulated_optimizer.state.values()})
    if state_steps != ([2.0] if case.get("existing_adam_state") else [1.0]):
        raise AssertionError("Unexpected Adam step count")
    return {"name": case["name"], "seed": case["seed"], "specifications": case["specifications"],
            "peft_lora": case.get("lora", False), "tied_embeddings": case.get("tied", False),
            "priming_updates_per_prototype": int(case.get("existing_adam_state", False)),
            "compared_updates_per_path": 1, "reference_loss": float(loss_reference.detach().item()),
            "loss_absolute_difference": abs(group["group_loss"] - float(loss_reference.detach().item())),
            "accumulation": group, "gradient_comparison": gradients, "clipped_gradient_comparison": clipped,
            "parameter_comparison": updates, "changed_parameter_elements": changed,
            "gradient_norm_before_clip": float(norm_reference.item()), "clip_max_norm": 1.0,
            "clipping_was_active": bool(norm_reference.item() > 1.0), "nonzero_lora_a_gradient_tensors": nonzero_a,
            "optimizer_state_steps_after": state_steps, "optimizer_state_comparison": "pass",
            "input_unchanged": True, "group_has_no_updates_or_gradient_clears": True}


def negative_controls():
    """错误实现必须被已选对照识别，避免等答案长度导致错误平均也能通过。"""
    full = make_batch([(3, 0, True), (13, 10, False)])
    good = make_model(177, tied=True, lora=True)
    bad = copy.deepcopy(good)
    reference_loss = good(**full, use_cache=False).loss
    reference_loss.backward()
    bad_loss = 0.0
    for batch in microbatches(full):
        loss, _ = target_position_loss(bad, batch)
        (loss / 2).backward()
        bad_loss += float(loss.detach().item()) / 2
    loss_error = abs(bad_loss - float(reference_loss.detach().item()))
    grad_error = max(float((left.grad - right.grad).abs().max().item())
                     for left, right in zip(named_trainable(good).values(), named_trainable(bad).values()))
    if loss_error <= 1e-5 or grad_error <= 1e-5:
        raise AssertionError("Incorrect equal-microbatch averaging was not detected")

    tail = make_batch([(3, 1, True), (7, 1, True), (11, 1, False), (4, 1, True)])
    tail_good = make_model(179)
    tail_bad = copy.deepcopy(tail_good)
    tail_loss = tail_good(**tail, use_cache=False).loss
    tail_loss.backward()
    wrong_tail_loss = 0.0
    for batch in microbatches(tail):
        loss, _ = target_position_loss(tail_bad, batch)
        (loss / 8).backward()
        wrong_tail_loss += float(loss.detach().item()) / 8
    torch.testing.assert_close(torch.tensor(wrong_tail_loss), tail_loss.detach() / 2, atol=ATOL, rtol=RTOL)
    for left, right in zip(tail_good.parameters(), tail_bad.parameters()):
        torch.testing.assert_close(right.grad, left.grad / 2, atol=ATOL, rtol=RTOL)
    return [{"name": "incorrect_equal_microbatch_average", "target_counts": [1, 11],
             "loss_absolute_error": loss_error, "max_gradient_absolute_error": grad_error, "detected": True},
            {"name": "tail_four_incorrectly_divided_by_eight", "gradient_scale_observed": 0.5,
             "loss_scale_observed": wrong_tail_loss / float(tail_loss.detach().item()), "detected": True}]


def invalid_groups():
    model = make_model(333)
    valid = make_batch([(3, 1, True)])
    ignored = {key: value.clone() for key, value in valid.items()}
    ignored["labels"].fill_(-100)
    cases = [("empty_group", []), ("all_targets_ignored", [ignored]),
             ("invalid_later_microbatch_rejected_before_backward", [valid, ignored])]
    results = []
    for name, batches in cases:
        try:
            backward_token_normalized_group(model, batches)
        except ValueError as error:
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError("Input validation should precede all backward calls")
            results.append({"name": name, "rejected": True, "error": str(error)})
        else:
            raise AssertionError(f"Invalid accumulation group accepted: {name}")
    return results


def checkpoint_resume_case():
    """只验证 CPU 内存 checkpoint 语义，不冒充真实 GPU 磁盘恢复已测试。"""
    groups = [microbatches(make_batch([(3, 2, True), (7, 5, False)])),
              microbatches(make_batch([(2, 0, True), (11, 3, False), (5, 7, True), (16, 1, True)]))]
    initial = make_model(401, tied=True, lora=True)
    uninterrupted = copy.deepcopy(initial)
    interrupted = copy.deepcopy(initial)
    opt_full = torch.optim.AdamW(list(named_trainable(uninterrupted).values()), **OPTIMIZER_KWARGS)
    opt_part = torch.optim.AdamW(list(named_trainable(interrupted).values()), **OPTIMIZER_KWARGS)

    def update(model, optimizer, index):
        # 本例处于正式计划的 24 步线性 warmup；恢复时按 next_group_index 设置下一步 LR。
        lr = 1e-4 * (index + 1) / 24
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        result = backward_token_normalized_group(model, groups[index], device="cpu")
        torch.nn.utils.clip_grad_norm_(list(named_trainable(model).values()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return result

    update(uninterrupted, opt_full, 0)
    full_second = update(uninterrupted, opt_full, 1)
    update(interrupted, opt_part, 0)
    checkpoint = {"model": interrupted.state_dict(), "optimizer": opt_part.state_dict(),
                  "next_group_index": 1, "completed_optimizer_updates": 1,
                  "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                  "numpy_rng": np.random.get_state()}
    memory_file = io.BytesIO()
    torch.save(checkpoint, memory_file)
    serialized_bytes = memory_file.tell()
    memory_file.seek(0)
    saved = torch.load(memory_file, map_location="cpu", weights_only=False)
    resumed = make_model(999, tied=True, lora=True)  # 故意从不同随机状态新建，必须靠 checkpoint 恢复。
    opt_resumed = torch.optim.AdamW(list(named_trainable(resumed).values()), **OPTIMIZER_KWARGS)
    resumed.load_state_dict(saved["model"], strict=True)
    opt_resumed.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    if not torch.equal(torch.get_rng_state(), checkpoint["torch_rng"]):
        raise AssertionError("Torch RNG state not restored")
    if random.getstate() != checkpoint["python_rng"]:
        raise AssertionError("Python RNG state not restored")
    restored_numpy = np.random.get_state()
    if (restored_numpy[0] != checkpoint["numpy_rng"][0]
            or not np.array_equal(restored_numpy[1], checkpoint["numpy_rng"][1])
            or restored_numpy[2:] != checkpoint["numpy_rng"][2:]):
        raise AssertionError("NumPy RNG state not restored")
    if saved["next_group_index"] != saved["completed_optimizer_updates"] or saved["next_group_index"] != 1:
        raise AssertionError("Resume cursor not at a complete optimizer boundary")
    resumed_second = update(resumed, opt_resumed, saved["next_group_index"])
    parameters = compare_tensors({name: p.detach() for name, p in named_trainable(uninterrupted).items()},
                                 {name: p.detach() for name, p in named_trainable(resumed).items()}, "resume_parameters")
    for ref_p, restored_p in zip(named_trainable(uninterrupted).values(), named_trainable(resumed).values()):
        for key, reference_value in opt_full.state[ref_p].items():
            torch.testing.assert_close(opt_resumed.state[restored_p][key], reference_value, atol=ATOL, rtol=RTOL)
    if opt_full.param_groups[0]["lr"] != opt_resumed.param_groups[0]["lr"]:
        raise AssertionError("Resumed learning rate differs")
    torch.testing.assert_close(torch.tensor(full_second["group_loss"]), torch.tensor(resumed_second["group_loss"]), atol=ATOL, rtol=RTOL)
    return {"name": "cpu_bytesio_checkpoint_resume", "status": "pass",
            "checkpoint_completed_updates": 1, "restored_next_group_index": 1, "final_completed_updates": 2,
            "parameter_comparison": parameters, "optimizer_state_comparison": "pass",
            "rng_restored": ["torch_cpu", "python", "numpy"], "second_update_lr": opt_resumed.param_groups[0]["lr"],
            "serialized_bytes": serialized_bytes, "scope": "CPU tiny model and BytesIO only; real GPU/disk resume untested"}


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    cases = [
        {"name": "unequal_target_lengths_dense", "seed": 21, "specifications": [(3, 1, True), (12, 8, False)]},
        {"name": "eos_only_and_right_padding_tied", "seed": 23, "tied": True,
         "specifications": [(2, 0, True), (8, 3, False), (5, 1, True)]},
        {"name": "full_eight_lora_nonzero_b", "seed": 29, "tied": True, "lora": True,
         "specifications": [(2, 0, True), (5, 2, True), (10, 5, False), (17, 8, True),
                            (4, 1, False), (8, 3, True), (12, 6, True), (22, 10, False)]},
        {"name": "tail_four_lora_nonzero_b", "seed": 31, "lora": True,
         "specifications": [(3, 1, True), (11, 5, False), (18, 2, True), (6, 8, True)]},
        {"name": "persistent_adam_state_lora", "seed": 37, "tied": True, "lora": True, "existing_adam_state": True,
         "specifications": [(4, 2, True), (9, 7, False), (20, 1, True), (7, 4, True)]},
        {"name": "noncontiguous_targets_dense", "seed": 41, "mask_hole": 6,
         "specifications": [(4, 5, True), (13, 2, False)]},
    ]
    previous_attempts = []
    if REPORT.exists():
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        previous_attempts = previous.get("previous_attempts", []) + [
            {key: previous[key] for key in ("started_utc", "status", "error_type", "error", "source_sha256") if key in previous}]
    source_files = ("src/train/token_accumulation.py", "src/train/test_token_accumulation.py",
                    "src/train/target_loss.py", "src/train/test_target_loss.py",
                    ".venv/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py",
                    ".venv/Lib/site-packages/transformers/loss/loss_utils.py")
    report = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "dtype": "torch.float32", "gpu_used": False,
              "real_model_weights_loaded": False, "formal_training_started": False,
              "final_examples_read": False, "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
              "tolerance_policy": "Fixed before execution; no widening on failure",
              "packages": {name: version(name) for name in ("torch", "transformers", "peft")},
              "source_sha256": {name: digest(ROOT / name) for name in source_files},
              "optimizer": OPTIMIZER_KWARGS, "case_results": [], "negative_controls": [], "invalid_groups": [],
              "previous_attempts": previous_attempts}
    try:
        for case in cases:
            row = compare_case(case)
            report["case_results"].append(row)
            print(f"PASS {case['name']}: loss={row['loss_absolute_difference']:.3g}, grad={row['gradient_comparison']['max_absolute_difference']:.3g}, parameter={row['parameter_comparison']['max_absolute_difference']:.3g}", flush=True)
        report["negative_controls"] = negative_controls()
        report["invalid_groups"] = invalid_groups()
        report["checkpoint_resume"] = checkpoint_resume_case()
        if not any(row["clipping_was_active"] for row in report["case_results"]):
            raise AssertionError("At least one case must exercise actual gradient clipping")
        report.update(status="pass", successful_equivalence_cases=len(cases),
                      successful_negative_controls=len(report["negative_controls"]),
                      successful_invalid_group_checks=len(report["invalid_groups"]),
                      max_loss_absolute_difference=max(row["loss_absolute_difference"] for row in report["case_results"]),
                      max_gradient_absolute_difference=max(row["gradient_comparison"]["max_absolute_difference"] for row in report["case_results"]),
                      max_parameter_absolute_difference=max(row["parameter_comparison"]["max_absolute_difference"] for row in report["case_results"]))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "successful_equivalence_cases", "successful_negative_controls", "successful_invalid_group_checks", "max_loss_absolute_difference", "max_gradient_absolute_difference", "max_parameter_absolute_difference", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
