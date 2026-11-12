"""固定32条短训练题、最多32次更新；prepare/run/verify分离，失败不自动重跑。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

STARTED = time.perf_counter()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.data.sft_features import collate_sft_batch
from src.infer.baseline_pilot import GPUSampler, read_json, save_json, sha256
from src.train.qlora_smoke import check_hashes, frozen_parameter_hash
from src.train.qlora_step import adapter_parameters, one_batch_update
from src.train.stability_core import select_short_subset, build_schedule, measure_subset_loss, weighted_loss_summary

CONFIG = ROOT / "configs/qlora_stability.json"


def publish_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Refuse to overwrite frozen selection: {path}")
    if not path.exists():
        path.write_bytes(data)


def prepare():
    config = read_json(CONFIG)
    if (config["subset_count"], config["batch_size"], config["max_steps"], config["max_sequence_length"]) != (32, 2, 32, 512):
        raise ValueError("This experiment is restricted to 32 short examples, batch2,32 steps")
    old = read_json(ROOT / "outputs/runs/qlora_single_batch_v2/preflight.json")
    protected = dict(old["protected_sha256"])
    for run_name in ("qlora_single_batch_v1", "qlora_single_batch_v2"):
        run = ROOT / "outputs/runs" / run_name
        for path in run.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name, expected in read_json(ROOT / "outputs/runs/qlora_single_batch_v2/artifact_sha256.json").items():
        if sha256(ROOT / "outputs/runs/qlora_single_batch_v2" / name) != expected:
            raise ValueError("Previous successful smoke artifacts changed")
    for name, expected in read_json(ROOT / "reports/qlora_smoke_v1_preservation.json")["sha256"].items():
        if sha256(ROOT / "outputs/runs/qlora_single_batch_v1" / name) != expected:
            raise ValueError("First smoke artifacts changed")
    for name in ("configs/qlora_stability.json", "src/train/qlora_stability.py", "src/train/stability_core.py",
                 "src/train/test_stability_core.py", "reports/qlora_smoke_result.json",
                 "reports/qlora_smoke_verification.json"):
        protected[name] = sha256(ROOT / name)
    # Pin the actually installed loss implementation used for token-weighted measurement.
    loss_source = ".venv/Lib/site-packages/transformers/loss/loss_utils.py"
    protected[loss_source] = sha256(ROOT / loss_source)
    check_hashes(protected)
    for name, pinned in old["packages"].items():
        if version(name) != pinned:
            raise ValueError(f"Frozen dependency version changed: {name}")
    source_manifest = ROOT / config["sft_manifest"]
    if sha256(source_manifest) != config["sft_manifest_sha256"]:
        raise ValueError("SFT manifest changed")
    with (source_manifest.parent / "train_tokens.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 6364:
        raise ValueError("Training population changed")
    selected, eligible = select_short_subset(rows, seed=config["seed"])
    ids = [r["source_index"] for r in selected]
    for r in selected:
        collate_sft_batch([r], 151643, max_length=512)
    schedule = build_schedule(ids, seed=config["seed"])
    subset = ROOT / config["subset_dir"]
    publish_json(subset / "features.json", selected)
    publish_json(subset / "schedule.json", schedule)
    lengths = [r["sequence_length"] for r in selected]
    manifest = {"config": config, "source_examples": 6364, "eligible_examples": eligible,
        "selection_seed": config["seed"], "source_indices": ids,
        "selected_examples": [{k: r[k] for k in ("source_index", "db_id", "sequence_length", "supervised_token_count")} for r in selected],
        "database_count": len({r["db_id"] for r in selected}),
        "length_min": min(lengths), "length_max": max(lengths), "length_median": statistics.median(lengths),
        "supervised_tokens_per_complete_pass": sum(r["supervised_token_count"] for r in selected),
        "protected_sha256": protected, "packages": old["packages"],
        "subset_sha256": {n: sha256(subset / n) for n in ("features.json", "schedule.json")},
        "truncated_examples": 0, "formal_training_examples_removed": 0,
        "validation_examples_loaded": False, "final_examples_read": False}
    publish_json(subset / "manifest.json", manifest)
    save_json(ROOT / "reports/qlora_stability_selection.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("source_examples", "eligible_examples", "source_indices", "database_count", "length_min", "length_max", "supervised_tokens_per_complete_pass")}), flush=True)
    return config, selected, schedule, manifest


def gpu_resource_check(config):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 is required")
    raw = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
    free = float(raw.split(",")[-1])
    result = {"gpu": raw, "free_mib": free, "minimum_free_mib": config["minimum_free_mib"],
              "estimated_process_peak_gib": config["estimated_process_peak_gib"],
              "estimated_runtime_minutes": config["estimated_runtime_minutes"], "status": "pass" if free >= config["minimum_free_mib"] else "insufficient_free_memory"}
    save_json(ROOT / "reports/qlora_stability_resources.json", result)
    if free < config["minimum_free_mib"]:
        raise RuntimeError(f"Only {free} MiB free; no model loaded, close GPU applications before starting")
    return result


def fresh_model(smoke):
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(ROOT / smoke["model_dir"], local_files_only=True,
        trust_remote_code=False, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    bias_count = 0
    quant_layers = []
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            quant_layers.append(module)
            if module.bias is not None:
                old = module.bias.detach()
                new = old.to(module.compute_dtype)
                if not torch.equal(old, new.to(old.dtype)):
                    raise ValueError("Bias preparation would lose numerical values")
                module.bias.data = new
                bias_count += 1
    if bias_count != 84 or len(quant_layers) != 196 or not model.is_loaded_in_4bit:
        raise ValueError("Unexpected Qwen quantized structure")
    if any(m.weight.quant_state.quant_type != "nf4" or not m.weight.quant_state.nested
           or m.compute_dtype != torch.bfloat16 for m in quant_layers):
        raise ValueError("Quantization protocol changed")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        revision=smoke["revision"], init_lora_weights=True, **smoke["lora"]))
    named = adapter_parameters(model)
    if sum(p.numel() for p in named.values()) != 9232384 or any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("Unexpected trainable count or CPU offload")
    if any(torch.count_nonzero(p).item() for n, p in named.items() if ".lora_B." in n):
        raise ValueError("Fresh LoRA B must start at zero; do not load a previous adapter")
    return model


class BoundedAdamW(torch.optim.AdamW):
    """32次step硬上限，同时记录尝试数与已返回的更新数。"""
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.completed_steps = 0

    def step(self, closure=None):
        if self.step_calls >= 32:
            raise RuntimeError("The 32-update hard limit has been reached")
        self.step_calls += 1
        result = super().step(closure)
        self.completed_steps += 1
        return result


def save_adapter_checked(model, destination):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(destination / "adapter_model.safetensors"))
    live = get_peft_model_state_dict(model)
    if len(saved) != 392 or set(saved) != set(live) or any("lora_" not in n for n in saved):
        raise ValueError("Adapter checkpoint contains unexpected tensors")
    if any(not torch.equal(p, live[n].detach().cpu()) for n, p in saved.items()):
        raise ValueError("Saved adapter differs from updated in-memory values")


def run(config, rows, schedule, selection):
    import numpy as np
    resources = gpu_resource_check(config)
    directory = ROOT / config["run_dir"]
    if directory.exists():
        raise FileExistsError("Run exists: use --verify, no implicit retry or extra updates")
    directory.mkdir(parents=True)
    sampler = GPUSampler(0.2)
    sampler.start()
    started = time.perf_counter()
    smoke = read_json(ROOT / config["source_smoke_config"])
    save_json(directory / "status.json", {"status": "started", "utc": datetime.now(timezone.utc).isoformat(), "completed_updates": 0})
    save_json(directory / "selection.json", selection)
    save_json(directory / "resources.json", resources)
    for filename in ("configs/qlora_stability.json", "configs/qlora_smoke.json", "src/train/qlora_stability.py",
                     "src/train/stability_core.py", "src/train/qlora_step.py", "src/data/sft_features.py", "src/train/test_stability_core.py"):
        dest = directory / "snapshot" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, dest)
    phase_memory, history = [], []
    model, optimizer = None, None
    current_phase = None
    def finish_phase():
        if current_phase is not None:
            torch.cuda.synchronize()
            record = {"phase": current_phase, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                      "current_allocated_mib": torch.cuda.memory_allocated() / 2**20}
            phase_memory.append(record)
            return record
    def start_phase(name):
        nonlocal current_phase
        finish_phase()
        current_phase = name
        sampler.stage = name
        torch.cuda.reset_peak_memory_stats()
    try:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        start_phase("load_and_prepare")
        phase_started = time.perf_counter()
        model = fresh_model(smoke)
        load_seconds = time.perf_counter() - phase_started
        base_before = frozen_parameter_hash(model)
        initial = {n: p.detach().cpu().clone() for n, p in adapter_parameters(model).items()}
        save_json(directory / "initial_state.json", {"base": base_before, "fresh_lora_b_all_zero": True,
            "adapter_source": "newly initialized from frozen original base, not smoke adapter", "trainable_parameters": 9232384})

        start_phase("before_training_loss")
        before = measure_subset_loss(model, rows, "cuda:0")
        save_json(directory / "loss_before.json", before)
        print(f"BEFORE: token-weighted training-subset loss={before['weighted_loss']:.6f}, targets={before['supervised_tokens']}", flush=True)
        # 优化器只创建一次，两个epoch之间保留AdamW的一阶/二阶统计。
        optimizer = BoundedAdamW(list(adapter_parameters(model).values()), lr=smoke["optimizer"]["lr"],
            weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        by_id = {r["source_index"]: r for r in rows}
        train_started = time.perf_counter()
        for item in schedule:
            if time.perf_counter() - train_started > config["training_loop_budget_seconds"]:
                raise TimeoutError("Training-loop budget exceeded; do not add steps or restart silently")
            start_phase(f"train_step_{item['step']:02d}")
            group = [by_id[i] for i in item["source_indices"]]
            batch = collate_sft_batch(group, 151643, max_length=512)
            targets = int((batch["labels"][:, 1:] != -100).sum())
            shape = list(batch["input_ids"].shape)
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            step_started = time.perf_counter()
            detail = one_batch_update(model, batch, optimizer)
            elapsed = time.perf_counter() - step_started
            del batch
            if optimizer.completed_steps != item["step"]:
                raise ValueError("Unexpected number of actual optimizer updates")
            save_json(directory / "steps" / f"step_{item['step']:02d}.json", detail)
            record = {**item, "loss": detail["loss"], "supervised_tokens": targets, "batch_shape": shape,
                "gradient_l2_norm": detail["gradients"]["global_l2_norm"], "gradients_all_finite": detail["gradients"]["all_finite"],
                "step_seconds_including_checks": elapsed, "timings": detail["timings"],
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "a_changed_tensors": sum(r["changed_elements"] > 0 for r in detail["adapter_changes"] if r["group"] == "A"),
                "b_changed_tensors": sum(r["changed_elements"] > 0 for r in detail["adapter_changes"] if r["group"] == "B")}
            history.append(record)
            with (directory / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            save_json(directory / "status.json", {"status": "training", "completed_updates": optimizer.completed_steps})
            print(f"STEP {item['step']:02d}/32 epoch={item['epoch']} loss={record['loss']:.6f} grad={record['gradient_l2_norm']:.4f} peak={record['peak_allocated_mib']:.0f}MiB", flush=True)
        train_seconds = time.perf_counter() - train_started

        start_phase("save_adapter")
        save_started = time.perf_counter()
        save_adapter_checked(model, directory / "adapter")  # 后续检查失败也保全这32步的真实产物。
        save_seconds = time.perf_counter() - save_started
        start_phase("after_training_loss")
        after = measure_subset_loss(model, rows, "cuda:0")
        save_json(directory / "loss_after.json", after)
        if ([r["source_indices"] for r in before["batches"]] != [r["source_indices"] for r in after["batches"]]
                or before["supervised_tokens"] != after["supervised_tokens"]):
            raise ValueError("Before/after loss used different examples or denominators")
        start_phase("final_integrity_checks")
        base_after = frozen_parameter_hash(model)
        save_json(directory / "base_parameter_hashes.json", {"before": base_before, "after": base_after})
        if base_before != base_after:
            raise ValueError("Frozen base parameter bytes changed")
        changes = []
        for name, p in adapter_parameters(model).items():
            delta = p.detach().cpu() - initial[name]
            changes.append({"name": name, "changed_elements": int(torch.count_nonzero(delta)),
                            "max_abs_delta": float(delta.abs().max())})
        save_json(directory / "adapter_changes_from_initial.json", changes)
        check_hashes(selection["protected_sha256"])
        for name, digest in selection["subset_sha256"].items():
            if sha256(ROOT / config["subset_dir"] / name) != digest:
                raise ValueError("Frozen short subset changed")
        finish_phase()
        current_phase = None
        sampler.close()
        save_json(directory / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors})
        save_json(directory / "phase_memory.json", phase_memory)
        training_only = [r for r in phase_memory if r["phase"].startswith("train_step_")]
        report = {"status": "pass", "run_dir": config["run_dir"], "selected_examples": 32,
            "completed_optimizer_steps": optimizer.completed_steps, "optimizer_step_calls": optimizer.step_calls,
            "epochs": 2, "batch_size": 2, "max_actual_length": selection["length_max"],
            "fresh_adapter": True, "trainable_parameters": 9232384,
            "training_subset_loss_before": before["weighted_loss"], "training_subset_loss_after": after["weighted_loss"],
            "loss_absolute_change_after_minus_before": after["weighted_loss"] - before["weighted_loss"],
            "supervised_tokens_per_loss_measurement": before["supervised_tokens"],
            "training_supervised_token_exposures": sum(r["supervised_tokens"] for r in history),
            "gradient_norm_min": min(r["gradient_l2_norm"] for r in history),
            "gradient_norm_max": max(r["gradient_l2_norm"] for r in history), "all_losses_and_gradients_finite": True,
            "all_updates_changed_adapter_values": True, "frozen_base_hashes_equal": True,
            "frozen_sources_and_prior_smoke_files_unchanged": True, "adapter_saved_and_readback_equal": True,
            "epoch_training_loss_online": {str(e): weighted_loss_summary([r for r in history if r["epoch"] == e])["weighted_loss"] for e in (1, 2)},
            "timings": {"load_and_prepare_seconds": load_seconds, "before_loss_seconds": before["seconds"],
                "training_loop_seconds": train_seconds, "after_loss_seconds": after["seconds"], "save_readback_seconds": save_seconds,
                "run_seconds_excluding_preflight": time.perf_counter() - started,
                "script_seconds_through_checks": time.perf_counter() - STARTED},
            "memory": {"torch_overall_peak_allocated_mib": max(r["peak_allocated_mib"] for r in phase_memory),
                "torch_overall_peak_reserved_mib": max(r["peak_reserved_mib"] for r in phase_memory),
                "torch_training_peak_allocated_mib": max(r["peak_allocated_mib"] for r in training_only),
                "whole_gpu_sampled_peak_mib": max(r["used_mib"] for r in sampler.samples),
                "whole_gpu_before_load_mib": sampler.samples[0]["used_mib"], "sample_count": len(sampler.samples), "sample_errors": sampler.errors},
            "adapter_bytes": (directory / "adapter/adapter_model.safetensors").stat().st_size,
            "no_truncation_or_formal_data_deletion": True, "validation_examples_loaded": False,
            "final_examples_read": False, "generation_or_ex_evaluation": False, "formal_full_training": False}
        save_json(directory / "summary.json", report)
        save_json(ROOT / "reports/qlora_stability_result.json", report)
        hashes = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*")
                  if p.is_file() and p.name not in ("artifact_sha256.json", "status.json")}
        save_json(directory / "artifact_sha256.json", hashes)
        save_json(directory / "status.json", {"status": "pass", "completed_updates": optimizer.completed_steps})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        checkpoint_error = None
        if model is not None and optimizer is not None and optimizer.completed_steps > 0:
            try:
                save_adapter_checked(model, directory / "partial_adapter")
            except Exception as save_error:
                checkpoint_error = repr(save_error)
        phase_error = None
        try:
            finish_phase()
        except Exception as diagnostic_error:
            phase_error = repr(diagnostic_error)
        sampler.close()
        save_json(directory / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors})
        save_json(directory / "phase_memory.json", phase_memory)
        failure = {"status": "failed", "error": repr(exc), "completed_optimizer_steps": optimizer.completed_steps if optimizer else 0,
                   "step_calls": optimizer.step_calls if optimizer else 0, "partial_adapter_save_error": checkpoint_error,
                   "phase_diagnostic_error": phase_error, "automatic_retry": False}
        save_json(directory / "failure.json", failure)
        save_json(directory / "status.json", failure)
        raise


def verify():
    config = read_json(CONFIG)
    directory = ROOT / config["run_dir"]
    selection = read_json(directory / "selection.json")
    check_hashes(selection["protected_sha256"])
    for name, expected in selection["subset_sha256"].items():
        if sha256(ROOT / config["subset_dir"] / name) != expected:
            raise ValueError(f"Frozen selected subset changed: {name}")
    for name, expected in read_json(directory / "artifact_sha256.json").items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    schedule = read_json(ROOT / config["subset_dir"] / "schedule.json")
    if len(history) != 32 or any(any(row[k] != planned[k] for k in ("step", "epoch", "source_indices")) for row, planned in zip(history, schedule)):
        raise ValueError("Actual updates differ from frozen32-step schedule")
    report = read_json(directory / "summary.json")
    if report != read_json(ROOT / "reports/qlora_stability_result.json") or report["completed_optimizer_steps"] != 32:
        raise ValueError("Summary mismatch")
    for stage in ("before", "after"):
        loss = read_json(directory / f"loss_{stage}.json")
        if weighted_loss_summary(loss["batches"])["weighted_loss"] != loss["weighted_loss"]:
            raise ValueError("Weighted loss cannot be reconstructed")
    from safetensors import safe_open
    with safe_open(str(directory / "adapter/adapter_model.safetensors"), framework="pt", device="cpu") as saved:
        if len(list(saved.keys())) != 392 or any("lora_" not in name for name in saved.keys()):
            raise ValueError("Unexpected saved adapter content")
    result = {"status": "pass", "completed_updates": 32, "source_and_run_hashes_verified": True,
        "loss_weighting_reconstructed": True, "schedule_matches_all32_updates": True,
        "model_load_or_training": False, "validation_examples_loaded": False, "final_examples_read": False}
    save_json(ROOT / "reports/qlora_stability_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--prepare", action="store_true")
    choices.add_argument("--run", action="store_true")
    choices.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify:
        verify()
    else:
        config, rows, schedule, selection = prepare()
        if arguments.run:
            run(config, rows, schedule, selection)
        else:
            gpu_resource_check(config)
            print("STABILITY_PREPARE: PASS; selection frozen; model not loaded")
