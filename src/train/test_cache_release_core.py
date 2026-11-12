"""CPU 小 LoRA 两次更新数值对照及组内停止；不验证真实 CUDA 缓存效果。"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 必须早于任何 torch/transformers 导入。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from src.train.cache_release_core import backward_cache_group
from src.train.token_accumulation import backward_token_normalized_group
from src.train.test_target_loss import make_batch, make_model
from src.train.test_token_accumulation import microbatches

REPORT = ROOT / "reports/cache_release_core_cpu_checks.json"
ATOL, RTOL = 1e-6, 1e-5
ADAM = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
        "weight_decay": 0.0, "foreach": False}
GROUP_SPECS = [
    [(2, 0, True), (5, 2, True), (10, 5, False), (17, 8, True),
     (4, 1, False), (8, 3, True), (12, 6, True), (22, 10, False)],
    [(19, 5, True), (3, 8, False), (7, 0, True), (21, 11, True),
     (6, 2, True), (8, 7, False), (15, 1, True), (4, 4, False)],
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def named(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def compare(left, right, *, atol=ATOL, rtol=RTOL):
    """递归比较 Adam 状态与参数张量，返回最大绝对误差。"""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(right, left, atol=atol, rtol=rtol)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError("Nonfinite tensor in comparison")
        return float((right - left).abs().max().item()) if left.numel() else 0.0
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("Dictionary keys differ")
        return max((compare(value, right[key], atol=atol, rtol=rtol) for key, value in left.items()), default=0.0)
    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("Sequence mismatch")
        return max((compare(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right)), default=0.0)
    if left != right:
        raise AssertionError(f"Scalar mismatch: {left} != {right}")
    return 0.0


def equivalence(seed, tied):
    reference = make_model(seed, tied=tied, lora=True)
    candidate = copy.deepcopy(reference)
    reference_optimizer = torch.optim.AdamW(list(named(reference).values()), **ADAM)
    optimizer = torch.optim.AdamW(list(named(candidate).values()), **ADAM)
    events, updates, release_count = [], [], 0
    for group_index, specs in enumerate(GROUP_SPECS):
        full = make_batch(specs)
        batches = microbatches(full)
        originals = copy.deepcopy(batches)
        reference_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        initial = {name: p.detach().clone() for name, p in named(candidate).items()}
        state_before = copy.deepcopy(optimizer.state_dict())
        expected_steps = [] if group_index == 0 else [1.0]
        if sorted({float(s["step"]) for s in optimizer.state.values()}) != expected_steps:
            raise AssertionError("Second group must retain first update Adam state")

        def observe(stage, index, payload):
            events.append((group_index, index, stage))
            if stage == "forward_end":
                if not isinstance(payload["forward_loss"], float):
                    raise AssertionError("Loss log must be a Python float")
                if payload["input_shape"] != list(batches[index]["input_ids"].shape):
                    raise AssertionError("Input was padded or truncated unexpectedly")
            if stage in ("backward_end", "micro_end"):
                if any(p.grad is None for p in named(candidate).values()):
                    raise AssertionError("Accumulated gradients were cleared")
            compare(state_before, optimizer.state_dict(), atol=0, rtol=0)
            compare(initial, {name: p.detach() for name, p in named(candidate).items()}, atol=0, rtol=0)

        def release(index):
            nonlocal release_count
            events.append((group_index, index, "release"))
            gradients = {name: p.grad.detach().clone() for name, p in named(candidate).items()}
            grad_ids = {name: id(p.grad) for name, p in named(candidate).items()}
            adam = copy.deepcopy(optimizer.state_dict())
            # CPU 模拟释放普通无用对象；绝不调用 CUDA empty_cache 伪装 GPU 实测。
            gc.collect()
            compare(gradients, {name: p.grad for name, p in named(candidate).items()}, atol=0, rtol=0)
            compare(adam, optimizer.state_dict(), atol=0, rtol=0)
            if grad_ids != {name: id(p.grad) for name, p in named(candidate).items()}:
                raise AssertionError("Release callback replaced a gradient")
            release_count += 1

        old = backward_token_normalized_group(reference, batches, device="cpu")
        with patch.object(optimizer, "step", side_effect=AssertionError("No group-internal step")), \
             patch.object(optimizer, "zero_grad", side_effect=AssertionError("No group-internal zero_grad")):
            new = backward_cache_group(candidate, batches, device="cpu", observe=observe, release=release)
        torch.testing.assert_close(torch.tensor(new["group_loss"]), torch.tensor(old["group_loss"]), atol=ATOL, rtol=RTOL)
        grad_diff = compare({n: p.grad for n, p in named(reference).items()},
                            {n: p.grad for n, p in named(candidate).items()})
        for model in (reference, candidate):
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise AssertionError("Frozen base received gradient")
        old_norm = torch.nn.utils.clip_grad_norm_(list(named(reference).values()), 1.0, error_if_nonfinite=True)
        new_norm = torch.nn.utils.clip_grad_norm_(list(named(candidate).values()), 1.0, error_if_nonfinite=True)
        torch.testing.assert_close(new_norm, old_norm, atol=ATOL, rtol=RTOL)
        clipped_diff = compare({n: p.grad for n, p in named(reference).items()},
                               {n: p.grad for n, p in named(candidate).items()})
        reference_optimizer.step()
        optimizer.step()
        param_diff = compare({n: p.detach() for n, p in named(reference).items()},
                             {n: p.detach() for n, p in named(candidate).items()})
        adam_diff = compare(reference_optimizer.state_dict(), optimizer.state_dict())
        compare(originals, batches, atol=0, rtol=0)
        changed = sum(int(torch.count_nonzero(p.detach() - initial[n])) for n, p in named(candidate).items())
        if not changed:
            raise AssertionError("No actual CPU update happened")
        updates.append({"update": group_index + 1, "old_loss": old["group_loss"], "new_loss": new["group_loss"],
                        "loss_absolute_difference": abs(new["group_loss"] - old["group_loss"]),
                        "gradient_max_difference": grad_diff, "clipped_gradient_max_difference": clipped_diff,
                        "parameter_max_difference": param_diff, "adam_max_difference": adam_diff,
                        "changed_elements": changed, "target_counts": new["micro_target_counts"],
                        "retained_adam_steps_before": expected_steps})
    expected_events = [(g, i, stage) for g in range(2) for i in range(8)
                       for stage in ("micro_start", "forward_end", "backward_end", "release", "micro_end")]
    if events != expected_events or release_count != 16:
        raise AssertionError("Release must follow backward and precede the next sample")
    if sorted({float(s["step"]) for s in optimizer.state.values()}) != [2.0]:
        raise AssertionError("Expected exactly two CPU updates")
    return {"seed": seed, "tied_embeddings": tied, "status": "pass", "updates": updates,
            "release_callbacks": release_count, "event_order_checked": True,
            "release_preserved_gradient_values_and_identities": True, "release_preserved_adam_values": True,
            "no_step_or_zero_grad_inside_group": True, "inputs_unchanged": True}


def stop_checks():
    reports = []
    class RequestedStop(RuntimeError):
        pass
    for stop_stage in ("micro_start", "forward_end", "backward_end", "release", "micro_end"):
        model = make_model(333, lora=True)
        optimizer = torch.optim.AdamW(list(named(model).values()), **ADAM)
        batches = microbatches(make_batch(GROUP_SPECS[0]))
        backward_token_normalized_group(model, batches, device="cpu")
        optimizer.step()  # CPU fixture priming: stopped group must preserve nonempty Adam state.
        optimizer.zero_grad(set_to_none=True)
        before = {name: p.detach().clone() for name, p in named(model).items()}
        adam = copy.deepcopy(optimizer.state_dict())
        called_after_stop, step_calls = [], []
        def observe(stage, index, payload):
            called_after_stop.append((index, stage))
            if index == 3 and stage == stop_stage:
                raise RequestedStop(stop_stage)
        def release(index):
            called_after_stop.append((index, "release"))
            if index == 3 and stop_stage == "release":
                raise RequestedStop(stop_stage)
        with patch.object(optimizer, "step", side_effect=lambda: step_calls.append(True)):
            try:
                backward_cache_group(model, batches, device="cpu", observe=observe, release=release)
                optimizer.step()  # This outer step must be unreachable after any callback failure.
            except RequestedStop as error:
                if str(error) != stop_stage:
                    raise
            else:
                raise AssertionError("Stop was swallowed")
        if step_calls or called_after_stop[-1] != (3, stop_stage):
            raise AssertionError("Work continued after stop")
        compare(before, {name: p.detach() for name, p in named(model).items()}, atol=0, rtol=0)
        compare(adam, optimizer.state_dict(), atol=0, rtol=0)
        optimizer.zero_grad(set_to_none=True)  # Explicitly discard incomplete accumulated gradients.
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Partial gradient discard failed")
        reports.append({"stage": stop_stage, "micro_index": 3, "status": "pass",
                        "stopped_group_optimizer_updates": 0, "adam_unchanged": True,
                        "parameters_unchanged": True, "partial_gradients_discarded": True})
    return reports


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt = ROOT / "outputs/checks/cache_release_core_cpu" / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    sources = ["src/train/cache_release_core.py", "src/train/test_cache_release_core.py",
               "src/train/token_accumulation.py", "src/train/target_loss.py",
               "src/train/test_target_loss.py", "src/train/test_token_accumulation.py"]
    report = {"status": "started", "device": "cpu", "real_model_loaded": False, "cuda_initialized": False,
              "final_examples_read": False, "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
              "source_sha256": {name: digest(ROOT / name) for name in sources},
              "attempt_directory": attempt.relative_to(ROOT).as_posix(), "equivalence_cases": [], "stop_cases": [],
              "limitation": "CPU simulated release checks callback semantics only; no actual CUDA cache behavior measured"}
    for name in sources[:2]:
        shutil.copyfile(ROOT / name, attempt / Path(name).name)
    try:
        for seed, tied in ((71, False), (73, True)):
            result = equivalence(seed, tied)
            report["equivalence_cases"].append(result)
            print(f"PASS two-consecutive-eight-micro LoRA seed={seed} tied={tied}", flush=True)
        report["stop_cases"] = stop_checks()
        if torch.cuda.is_initialized():
            raise AssertionError("CPU-only test initialized CUDA")
        report.update(status="pass", cuda_initialized=False)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      cuda_initialized=torch.cuda.is_initialized())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        (attempt / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if report["status"] == "pass":
            if REPORT.exists():
                raise FileExistsError("Passing report already exists; do not silently overwrite")
            REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "equivalence_cases": len(report["equivalence_cases"]),
                      "stop_cases": len(report["stop_cases"]), "seconds": report["seconds"],
                      "cuda_initialized": report["cuda_initialized"]}), flush=True)


if __name__ == "__main__":
    main()
