"""固定两条训练记录的一次QLoRA更新。--check预检；--run只运行一次；--verify只读验收。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

SCRIPT_STARTED = time.perf_counter()
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.sft_features import collate_sft_batch
from src.infer.baseline_pilot import GPUSampler, read_json, save_json, sha256
from src.train.qlora_step import adapter_parameters, one_batch_update

CONFIG = ROOT / "configs/qlora_smoke.json"
RUN = ROOT / "outputs/runs/qlora_single_batch_v1"
MANIFEST = ROOT / "data/processed/spider_sft_tokens_v1/manifest.json"
SUMMARY = ROOT / "reports/qlora_smoke_result.json"


def check_hashes(expected):
    for name, digest in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError(f"Frozen file changed: {name}")


def preflight():
    import torch
    import peft
    from transformers import AutoTokenizer
    config = read_json(CONFIG)
    if (config["source_indices"] != [0, 1] or config["optimizer_steps"] != 1
            or config["max_batch_length"] != 512 or config["use_cache"]):
        raise ValueError("This smoke script is limited to one short fixed batch")
    if sha256(MANIFEST) != config["sft_manifest_sha256"]:
        raise ValueError("SFT dataset manifest changed")
    manifest = read_json(MANIFEST)
    if manifest["stats"]["count"] != 6364 or manifest["model_revision"] != config["revision"]:
        raise ValueError("Unexpected dataset/model version")
    protected = dict(manifest["source_sha256"])
    protected.update({str((MANIFEST.parent / n).relative_to(ROOT)).replace("\\", "/"): h
                      for n, h in manifest["output_sha256"].items()})
    protected[MANIFEST.relative_to(ROOT).as_posix()] = sha256(MANIFEST)
    protected.update(read_json(ROOT / "configs/baseline_full.json")["protected_pilot_sha256"])
    # Preserve historical predictions without reading validation labels/databases.
    full_pred = "outputs/preds/20261026T080956_648523Z_baseline_full/predictions.jsonl"
    protected[full_pred] = "0933b85da2b7e9628bd80063c0d85403c2432cad259d6b0a5adcef234c10bcf6"
    model_manifest = read_json(ROOT / "reports/model_manifest.json")
    for weight in (r for r in model_manifest["files"] if r["is_weight"]):
        protected[config["model_dir"] + "/" + weight["name"]] = weight["sha256"]
    for name in ("configs/qlora_smoke.json", "src/train/qlora_smoke.py", "src/train/qlora_step.py",
                 "src/infer/baseline_pilot.py", "src/train/test_qlora_step.py", "reports/sft_data_acceptance.json"):
        protected[name] = sha256(ROOT / name)
    check_hashes(protected)
    packages = {}
    for line in (ROOT / "requirements.lock.txt").read_text().splitlines():
        if "==" not in line:
            continue
        name, pinned = line.split("==")
        packages[name] = version(name)
        if packages[name] != pinned:
            raise ValueError(f"Previously frozen dependency changed: {name}")
    packages["peft"] = peft.__version__
    if packages["peft"] != "0.21.2":
        raise ValueError("Expected peft==0.21.2")
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / config["model_dir"], local_files_only=True)
    selected = {}
    with (MANIFEST.parent / "train_tokens.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["source_index"] in config["source_indices"]:
                if row["source_index"] in selected:
                    raise ValueError("Duplicate selected training id")
                selected[row["source_index"]] = row
            if len(selected) == 2:
                break
    rows = [selected[index] for index in config["source_indices"]]
    batch = collate_sft_batch(rows, tokenizer.pad_token_id, max_length=config["max_batch_length"])
    if list(batch["input_ids"].shape) != [2, 306] or int((batch["labels"] != -100).sum()) != 29:
        raise ValueError("Fixed batch shape or supervision changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 support is required")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
    free_mib = float(gpu.splitlines()[0].split(",")[-2])
    if free_mib < config["minimum_free_mib"]:
        raise RuntimeError(f"Only {free_mib} MiB free; stop before loading the model")
    report = {"status": "pass", "config": config, "packages": packages,
              "pip_check": pip_check.stdout.strip(), "gpu": gpu,
              "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
              "dataset_manifest_sha256": sha256(MANIFEST), "protected_sha256": protected,
              "batch": {"source_indices": config["source_indices"], "db_ids": [r["db_id"] for r in rows],
                        "shape": list(batch["input_ids"].shape), "sequence_lengths": [r["sequence_length"] for r in rows],
                        "supervised_tokens": int((batch["labels"] != -100).sum()),
                        "padding_tokens": int((batch["attention_mask"] == 0).sum()),
                        "gold_sql": [tokenizer.decode(r["input_ids"][r["prompt_length"]:r["eos_position"]],
                                      clean_up_tokenization_spaces=False) for r in rows]},
              "model_weights_loaded": False, "validation_examples_loaded": False, "final_examples_read": False}
    return config, batch, report


def frozen_parameter_hash(model):
    """加载/类型转换完成后，对所有冻结参数的名字、形状、dtype和原始字节取摘要。"""
    import torch
    digest = hashlib.sha256()
    count = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(json.dumps([name, list(p.shape), str(p.dtype)]).encode())
            raw = p.detach().cpu().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast("B"))
            count += 1
    return {"sha256": digest.hexdigest(), "frozen_parameter_tensors": count}


def run(config, batch, checked):
    import numpy as np
    import torch
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    if RUN.exists():
        raise FileExistsError("Smoke run already exists; use --verify, not a second optimizer update")
    started = time.perf_counter()
    RUN.mkdir(parents=True, exist_ok=False)
    sampler = GPUSampler(0.2)
    sampler.start()
    save_json(RUN / "status.json", {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat()})
    save_json(RUN / "preflight.json", checked)
    save_json(RUN / "batch.json", {k: v.tolist() for k, v in batch.items()})
    for name in ("configs/qlora_smoke.json", "src/train/qlora_smoke.py", "src/train/qlora_step.py",
                 "src/data/sft_features.py", "src/train/test_qlora_step.py", "configs/model.yaml", "configs/prompt.yaml", "requirements.lock.txt"):
        dest = RUN / "snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    (RUN / "environment.lock.txt").write_text("\n".join(f"{n}=={v}" for n, v in sorted(checked["packages"].items())) + "\n")
    try:
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        sampler.stage = "load_and_attach"
        load_started = time.perf_counter()
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(ROOT / config["model_dir"],
            local_files_only=True, trust_remote_code=False, quantization_config=quantization,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=True,
                          revision=config["revision"], **config["lora"])
        model = get_peft_model(model, lora)
        trainable = adapter_parameters(model)
        train_count, total_count = model.get_nb_trainable_parameters()
        if train_count != config["expected_trainable_parameters"]:
            raise ValueError(f"Unexpected trainable parameter count {train_count}")
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("Unexpected CPU/disk offload")
        quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        if len(quantized) != 196 or not model.is_loaded_in_4bit:
            raise ValueError("Expected 196 quantized Qwen linear layers")
        if any(m.weight.quant_state.quant_type != "nf4" or not m.weight.quant_state.nested
               or m.compute_dtype != torch.bfloat16 for m in quantized):
            raise ValueError("Quantization settings differ from frozen NF4/double/bf16")
        parameter_report = {"trainable": train_count, "total_logical_including_lora": total_count,
            "trainable_percent": 100 * train_count / total_count, "quantized_linear_layers": len(quantized),
            "adapter_tensors": len(trainable), "all_base_requires_grad_false": True,
            "adapter_dtypes": sorted({str(p.dtype) for p in trainable.values()}),
            "gradient_checkpointing": model.is_gradient_checkpointing,
            "trainable_names_and_shapes": {n: list(p.shape) for n, p in trainable.items()}}
        save_json(RUN / "parameters.json", parameter_report)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        print(f"MODEL_READY: trainable={train_count:,}, total={total_count:,}, batch=[2,306]", flush=True)
        base_before = frozen_parameter_hash(model)
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=config["optimizer"]["lr"],
            weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        step_started = time.perf_counter()
        result = one_batch_update(model, batch, optimizer, lambda stage: setattr(sampler, "stage", stage))
        step_with_checks_seconds = time.perf_counter() - step_started
        save_json(RUN / "step.json", result)  # 更新后的证据先落盘，后续保存失败也不能偷偷重跑。
        save_json(RUN / "status.json", {"status": "optimizer_step_completed", "optimizer_steps": 1})
        base_after = frozen_parameter_hash(model)
        if base_before != base_after:
            raise ValueError("Frozen in-memory base parameter bytes changed")
        sampler.stage = "save_adapter"
        save_started = time.perf_counter()
        model.save_pretrained(RUN / "adapter", safe_serialization=True, save_embedding_layers=False)
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict
        saved = load_file(str(RUN / "adapter/adapter_model.safetensors"))
        live = get_peft_model_state_dict(model)
        if set(saved) != set(live) or any("lora_" not in n for n in saved):
            raise ValueError("Saved adapter contains unexpected base parameters")
        if any(not torch.equal(t, live[n].detach().cpu()) for n, t in saved.items()):
            raise ValueError("Saved adapter differs from updated in-memory values")
        save_seconds = time.perf_counter() - save_started
        check_hashes(checked["protected_sha256"])
        sampler.close()
        memory = {"torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "whole_gpu_sampled_peak_mib": max(r["used_mib"] for r in sampler.samples),
                  "whole_gpu_before_load_mib": sampler.samples[0]["used_mib"],
                  "sample_count": len(sampler.samples), "sample_errors": sampler.errors,
                  "note": "torch allocator peaks cover load+step+save; nvidia-smi samples include other apps and may miss brief peaks"}
        group_summary = {}
        for group in ("A", "B"):
            gs = [r for r in result["gradients"]["tensors"] if r["group"] == group]
            ds = [r for r in result["adapter_changes"] if r["group"] == group]
            group_summary[group] = {"tensor_count": len(gs), "nonzero_gradient_tensors": sum(r["norm"] > 0 for r in gs),
                "changed_tensors": sum(r["changed_elements"] > 0 for r in ds),
                "changed_elements": sum(r["changed_elements"] for r in ds),
                "max_abs_delta": max(r["max_abs_delta"] for r in ds)}
        report = {"status": "pass", "run_dir": RUN.relative_to(ROOT).as_posix(), "batch": checked["batch"],
            "model_revision": config["revision"], "parameters": {k: v for k, v in parameter_report.items() if k != "trainable_names_and_shapes"},
            "loss": result["loss"], "loss_finite": result["loss_finite"], "forward_calls": 1, "backward_calls": 1, "optimizer_steps": 1,
            "gradient_l2_norm": result["gradients"]["global_l2_norm"], "gradient_all_finite": True,
            "adapter_groups": group_summary, "gradients_cleared_after_step": True,
            "base_parameters_before": base_before, "base_parameters_after": base_after,
            "frozen_files_unchanged": True, "saved_adapter_verified_against_memory": True,
            "memory": memory, "timings": {**result["timings"], "load_and_attach_seconds": load_seconds,
                "step_including_diagnostics_seconds": step_with_checks_seconds, "save_and_readback_seconds": save_seconds,
                "run_seconds_excluding_preflight_and_imports": time.perf_counter() - started,
                "script_seconds_through_integrity_check": time.perf_counter() - SCRIPT_STARTED},
            "adapter_file_bytes": (RUN / "adapter/adapter_model.safetensors").stat().st_size,
            "validation_examples_loaded": False, "final_examples_read": False, "long_sample_training": False,
            "generation_or_evaluation_performed": False, "formal_training": False}
        save_json(RUN / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors})
        save_json(RUN / "summary.json", report)
        save_json(SUMMARY, report)
        artifacts = {p.relative_to(RUN).as_posix(): sha256(p) for p in RUN.rglob("*")
                     if p.is_file() and p.name not in ("status.json", "artifact_sha256.json")}
        save_json(RUN / "artifact_sha256.json", artifacts)
        save_json(RUN / "status.json", {"status": "pass", "optimizer_steps": 1})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        sampler.close()
        save_json(RUN / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                  "retry_policy": "No automatic retry; inspect step.json/status.json before any further action"})
        save_json(RUN / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors})
        raise


def verify():
    from safetensors import safe_open
    report = read_json(RUN / "summary.json")
    checked = read_json(RUN / "preflight.json")
    check_hashes(checked["protected_sha256"])
    for name, digest in read_json(RUN / "artifact_sha256.json").items():
        if sha256(RUN / name) != digest:
            raise ValueError(f"Smoke artifact changed: {name}")
    if report != read_json(SUMMARY) or report["optimizer_steps"] != 1 or report["status"] != "pass":
        raise ValueError("Smoke summary mismatch")
    with safe_open(str(RUN / "adapter/adapter_model.safetensors"), framework="pt", device="cpu") as f:
        if len(list(f.keys())) != 392 or any("lora_" not in n for n in f.keys()):
            raise ValueError("Adapter file should contain only 392 LoRA tensors")
    save_json(ROOT / "reports/qlora_smoke_verification.json", {"status": "pass", "all_source_and_artifact_hashes_verified": True,
        "adapter_tensors": 392, "optimizer_steps": 1, "new_training_or_model_load": False,
        "validation_examples_loaded": False, "final_examples_read": False})
    print("QLORA_SMOKE_VERIFY: PASS; no model loaded or optimizer step repeated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        config, batch, checked = preflight()
        save_json(ROOT / "reports/qlora_smoke_preflight.json", checked)
        print("QLORA_PREFLIGHT: PASS; 2 training records, 29 supervised tokens, no model weights loaded", flush=True)
        if args.run:
            run(config, batch, checked)
