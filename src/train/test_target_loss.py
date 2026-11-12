"""CPU小Qwen严格对照：完整词表loss与目标位置loss及所有可训练参数梯度。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from src.train.target_loss import target_position_loss

REPORT = ROOT / "reports/target_loss_cpu_equivalence.json"
ATOL, RTOL = 1e-6, 1e-5  # 执行前固定；失败时不得自动调宽容差。
EOS, PAD, VOCAB = 2, 0, 67


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_batch(specifications):
    """每个规格是(prompt长度, SQL长度, 是否保留模板尾换行)。"""
    records = []
    for index, (prompt_length, sql_length, tail) in enumerate(specifications):
        prompt = [3 + (index * 7 + i) % (VOCAB - 3) for i in range(prompt_length)]
        sql = [3 + (index * 11 + i + 5) % (VOCAB - 3) for i in range(sql_length)]
        ids = prompt + sql + [EOS] + ([4] if tail else [])
        labels = [-100] * prompt_length + sql + [EOS] + ([-100] if tail else [])
        records.append((ids, labels))
    longest = max(len(ids) for ids, labels in records)
    return {
        "input_ids": torch.tensor([ids + [PAD] * (longest - len(ids)) for ids, labels in records], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (longest - len(ids)) for ids, labels in records], dtype=torch.long),
        "labels": torch.tensor([labels + [-100] * (longest - len(ids)) for ids, labels in records], dtype=torch.long),
    }


def make_model(seed, tied=False, lora=False):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=PAD, eos_token_id=EOS,
        attention_dropout=0.0, use_cache=False, tie_word_embeddings=tied)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).float().cpu()
    if tied and model.lm_head.weight is not model.model.embed_tokens.weight:
        raise AssertionError("Requested tied embeddings were not tied")
    if lora:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        # CPU诊断中给B非零初值，确保A梯度也真正参与等价性比较；不影响真实GPU新LoRA初始化。
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(mean=0.0, std=0.03)
    model.train()
    return model


def compare_case(case):
    batch = make_batch(case["specifications"])
    if case.get("mask_hole"):
        batch["labels"][0, case["mask_hole"]] = -100
    model_reference = make_model(case["seed"], case.get("tied", False), case.get("lora", False))
    model_target = copy.deepcopy(model_reference)
    before_batch = {key: value.clone() for key, value in batch.items()}
    # 确认两条计算路径从完全相同参数开始，不进行optimizer.step。
    for name, value in model_reference.state_dict().items():
        torch.testing.assert_close(value, model_target.state_dict()[name], atol=0, rtol=0)
    reference = model_reference(**batch, use_cache=False)
    full_shape = list(reference.logits.shape)
    full_loss = reference.loss
    target_loss, metadata = target_position_loss(model_target, batch)
    torch.testing.assert_close(target_loss, full_loss, atol=ATOL, rtol=RTOL)
    full_loss.backward()
    target_loss.backward()
    reference_params, target_params = dict(model_reference.named_parameters()), dict(model_target.named_parameters())
    max_grad_error, worst_name, compared, nonzero_a = 0.0, None, 0, 0
    for name, parameter in reference_params.items():
        other = target_params[name]
        if not parameter.requires_grad:
            if parameter.grad is not None or other.grad is not None:
                raise AssertionError("Frozen base unexpectedly received gradients")
            continue
        if parameter.grad is None or other.grad is None:
            raise AssertionError(f"Missing gradient for {name}")
        if not torch.isfinite(parameter.grad).all() or not torch.isfinite(other.grad).all():
            raise AssertionError(f"Nonfinite gradient for {name}")
        torch.testing.assert_close(other.grad, parameter.grad, atol=ATOL, rtol=RTOL,
                                   msg=lambda message: f"{name}: {message}")
        error = float((other.grad - parameter.grad).abs().max().item())
        if error > max_grad_error:
            max_grad_error, worst_name = error, name
        compared += 1
        if ".lora_A." in name and bool(torch.count_nonzero(parameter.grad).item()):
            nonzero_a += 1
    if case.get("lora") and nonzero_a == 0:
        raise AssertionError("LoRA comparison must exercise nonzero A gradients")
    for key, original in before_batch.items():
        torch.testing.assert_close(batch[key], original, atol=0, rtol=0)
    expected_targets = int((batch["labels"][:, 1:] != -100).sum().item())
    if metadata["valid_target_count"] != expected_targets or metadata["input_shape"] != list(batch["input_ids"].shape):
        raise AssertionError("Target count or full input shape changed")
    return {"name": case["name"], "seed": case["seed"], "specifications": case["specifications"],
        "tied_embeddings": case.get("tied", False), "peft_lora": case.get("lora", False),
        "standard_loss": float(full_loss.detach().item()), "target_loss": float(target_loss.detach().item()),
        "loss_absolute_difference": abs(float((target_loss - full_loss).detach().item())),
        "max_gradient_absolute_difference": max_grad_error, "worst_gradient_parameter": worst_name,
        "gradient_tensors_compared": compared, "nonzero_lora_a_gradients": nonzero_a,
        "standard_logits_shape": full_shape, "optimized": metadata,
        "all_gradient_comparisons_passed": True, "batch_unchanged": True, "optimizer_updates": 0}


def negative_cases():
    model = make_model(99)
    original = make_batch([(3, 2, True)])
    cases = []
    def reject(name, batch):
        try:
            target_position_loss(model, batch)
        except ValueError as error:
            cases.append({"name": name, "status": "rejected", "error": str(error)})
        else:
            raise AssertionError(f"Invalid case was not rejected: {name}")
    masked = {key: value.clone() for key, value in original.items()}
    masked["labels"].fill_(-100)
    reject("all_targets_ignored", masked)
    wrong_shape = {**original, "labels": original["labels"][:, :-1]}
    reject("mismatched_shape", wrong_shape)
    reject("float_labels", {**original, "labels": original["labels"].float()})
    padded = make_batch([(3, 1, True), (7, 2, True)])
    padded["labels"][0, -1] = EOS
    reject("supervision_on_padding", padded)
    invalid_token = {key: value.clone() for key, value in original.items()}
    invalid_token["labels"][0, 3] = VOCAB
    reject("out_of_vocabulary_target", invalid_token)
    reject("extra_field_rejected", {**original, "position_ids": original["input_ids"]})
    return cases


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    cases = [
        {"name": "short_sql_eos_tail", "seed": 0, "specifications": [(3, 2, True)]},
        {"name": "eos_at_last_position_tied", "seed": 7, "specifications": [(7, 4, False)], "tied": True},
        {"name": "batch2_different_masks_right_padding_tied", "seed": 42,
         "specifications": [(2, 2, True), (8, 4, True)], "tied": True},
        {"name": "noncontiguous_target_columns", "seed": 11, "specifications": [(5, 5, True)], "mask_hole": 7},
        {"name": "eos_only_answer", "seed": 13, "specifications": [(6, 0, True)]},
        {"name": "longer_prompt_peft_lora_nonzero_b", "seed": 17,
         "specifications": [(29, 5, True)], "tied": True, "lora": True},
        {"name": "batch2_peft_lora_nonzero_b", "seed": 23,
         "specifications": [(3, 2, True), (12, 5, False)], "lora": True},
    ]
    report = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu", "dtype": "torch.float32", "attention_implementation": "sdpa",
        "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
        "tolerance_policy": "Fixed before execution; no widening on failure",
        "packages": {name: version(name) for name in ("torch", "transformers", "peft")},
        "case_results": [], "negative_cases": [], "real_model_weights_loaded": False,
        "gpu_used": False, "optimizer_updates": 0,
        "source_sha256": {name: digest(ROOT / name) for name in (
            "src/train/target_loss.py", "src/train/test_target_loss.py",
            ".venv/Lib/site-packages/transformers/models/qwen2/modeling_qwen2.py",
            ".venv/Lib/site-packages/transformers/loss/loss_utils.py")}}
    try:
        for case in cases:
            result = compare_case(case)
            report["case_results"].append(result)
            print(f"PASS {case['name']}: loss_abs={result['loss_absolute_difference']:.3g}, grad_abs={result['max_gradient_absolute_difference']:.3g}, gradients={result['gradient_tensors_compared']}", flush=True)
        report["negative_cases"] = negative_cases()
        report.update(status="pass", successful_equivalence_cases=len(cases),
                      successful_negative_cases=len(report["negative_cases"]),
                      max_loss_absolute_difference=max(r["loss_absolute_difference"] for r in report["case_results"]),
                      max_gradient_absolute_difference=max(r["max_gradient_absolute_difference"] for r in report["case_results"]))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "successful_equivalence_cases", "successful_negative_cases", "max_loss_absolute_difference", "max_gradient_absolute_difference", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
