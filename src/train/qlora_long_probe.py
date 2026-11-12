"""每档独立进程、完整长样本、最多一次更新；任何失败停止后续档。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from src.data.sft_features import collate_sft_batch
from src.infer.baseline_pilot import GPUSampler, read_json, save_json, sha256
from src.train.qlora_smoke import check_hashes, frozen_parameter_hash
from src.train.qlora_stability import fresh_model, publish_json, save_adapter_checked
from src.train.qlora_step import adapter_parameters, check_gradients
from src.train.long_probe_core import select_probe_rows, SingleUpdateAdamW, optimizer_state_inventory

CONFIG = ROOT / "configs/qlora_long_probe.json"
SOURCES = ["configs/qlora_long_probe.json", "src/train/qlora_long_probe.py",
           "src/train/long_probe_core.py", "src/train/test_long_probe_core.py"]


def prepare():
    config = read_json(CONFIG)
    if (config["tiers"], config["batch_size"], config["max_updates_per_tier"]) != (["1024", "2048", "3761"], 1, 1):
        raise ValueError("Only the authorized three tiers, batch1, one update are allowed")
    previous = read_json(ROOT / "data/processed/spider_stability_32_v1/manifest.json")
    protected = dict(previous["protected_sha256"])
    check_hashes(protected)
    for name, pinned in previous["packages"].items():
        if version(name) != pinned:
            raise ValueError(f"Frozen dependency changed: {name}")
    old_run = ROOT / "outputs/runs/qlora_stability_32x32_v1"
    for name, digest in read_json(old_run / "artifact_sha256.json").items():
        if sha256(old_run / name) != digest:
            raise ValueError(f"Previous stability artifact changed: {name}")
    for parent in (old_run, ROOT / "data/processed/spider_stability_32_v1"):
        for path in parent.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    for name in SOURCES + ["reports/qlora_stability_result.json", "reports/qlora_stability_verification.json"]:
        protected[name] = sha256(ROOT / name)
    with (ROOT / "data/processed/spider_sft_tokens_v1/train_tokens.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 6364:
        raise ValueError("Frozen training population changed")
    selected = select_probe_rows(rows)
    for row in selected:
        batch = collate_sft_batch([row], 151643, max_length=3761)
        if list(batch["input_ids"].shape) != [1, row["sequence_length"]] or (batch["attention_mask"] == 0).any():
            raise ValueError("Probe must use one complete unpadded example")
        if int((batch["labels"][:, 1:] != -100).sum()) != row["supervised_token_count"]:
            raise ValueError("Shifted target count changed")
    subset = ROOT / config["subset_dir"]
    publish_json(subset / "features.json", selected)
    manifest = {"config": config, "source_examples": len(rows), "packages": previous["packages"],
        "selected": [{"tier": t, **{k: r[k] for k in ("source_index", "db_id", "sequence_length", "prompt_length", "supervised_token_count")}}
                     for t, r in zip(config["tiers"], selected)],
        "features_sha256": sha256(subset / "features.json"), "protected_sha256": protected,
        "truncated_examples": 0, "formal_examples_removed": 0,
        "validation_examples_loaded": False, "final_examples_read": False}
    publish_json(subset / "manifest.json", manifest)
    save_json(ROOT / "reports/qlora_long_probe_selection.json", manifest)
    print(json.dumps(manifest["selected"], ensure_ascii=False), flush=True)


def load_checked():
    config = read_json(CONFIG)
    subset = ROOT / config["subset_dir"]
    manifest = read_json(subset / "manifest.json")
    check_hashes(manifest["protected_sha256"])
    if config != manifest["config"] or sha256(subset / "features.json") != manifest["features_sha256"]:
        raise ValueError("Frozen probe config/features changed")
    for name, pinned in manifest["packages"].items():
        if version(name) != pinned:
            raise ValueError(f"Frozen dependency changed: {name}")
    return config, manifest, read_json(subset / "features.json")


def gpu_snapshot():
    raw = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
    fields = [x.strip() for x in raw.split(",")]
    return {"name": fields[0], "total_mib": float(fields[1]), "used_mib": float(fields[2]),
            "free_mib": float(fields[3]), "driver": fields[4]}


class PhaseMemory:
    """阶段峰值是进程总量，包含该阶段仍存活的模型/梯度等；不是独占增量。"""
    def __init__(self, directory, sampler):
        self.directory, self.sampler = directory, sampler
        self.rows, self.current = [], None

    def begin(self, name):
        if self.current is not None:
            raise RuntimeError("Previous phase was not closed")
        torch.cuda.synchronize()
        self.current = name
        self.sampler.stage = name
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        save_json(self.directory / "current_phase.json", {"phase": name, "utc": datetime.now(timezone.utc).isoformat()})

    def end(self, failed=False):
        if self.current is None:
            return
        row = {"phase": self.current, "failed": failed}
        diagnostic_failure = None
        try:
            if not failed:
                torch.cuda.synchronize()
            row.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                       end_allocated_mib=torch.cuda.memory_allocated() / 2**20,
                       end_reserved_mib=torch.cuda.memory_reserved() / 2**20)
        except Exception as error:
            row["diagnostic_error"] = str(error)
            row["failed"] = True
            diagnostic_failure = error
        row["seconds"] = time.perf_counter() - self.started
        self.rows.append(row)
        self.current = None
        save_json(self.directory / "phase_memory.json", self.rows)
        print(json.dumps(row), flush=True)
        if diagnostic_failure is not None and not failed:
            raise diagnostic_failure


def worker(tier):
    config, manifest, rows = load_checked()
    directory = ROOT / config["run_root"] / tier
    if os.environ.get("QLORA_PROBE_CLAIM") != str(directory.resolve()) or not (directory / "launch_claim.json").exists():
        raise RuntimeError("Worker must be launched once by --run-tier")
    if (directory / "worker_started.json").exists():
        raise FileExistsError("Worker already started; no retry")
    save_json(directory / "worker_started.json", {"utc": datetime.now(timezone.utc).isoformat()})
    index = config["tiers"].index(tier)
    row = rows[index]
    resources = gpu_snapshot()
    save_json(directory / "resources.json", resources)
    if resources["free_mib"] < config["minimum_free_mib"]:
        save_json(directory / "result.json", {"status": "resource_blocked", "completed_steps": 0, "step_calls": 0, "resources": resources})
        return 2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 required")
    sampler = GPUSampler(0.2)
    sampler.start()
    phases = PhaseMemory(directory, sampler)
    optimizer, model = None, None
    started = time.perf_counter()
    try:
        import numpy as np
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        phases.begin("load_and_prepare")
        model = fresh_model(read_json(ROOT / config["source_config"]))
        phases.end()
        phases.begin("setup_and_base_hash")
        base_before = frozen_parameter_hash(model)
        named = adapter_parameters(model)
        initial = {n: p.detach().cpu().clone() for n, p in named.items()}
        save_json(directory / "initial_state.json", {"base": base_before, "fresh_lora_b_zero": True,
            "trainable_parameters": sum(p.numel() for p in named.values()),
            "adapter_parameter_bytes": sum(p.numel() * p.element_size() for p in named.values()),
            "adapter_dtypes": sorted({str(p.dtype) for p in named.values()})})
        optimizer = SingleUpdateAdamW(list(named.values()),
            audit_callback=lambda status: save_json(directory / "update_status.json", status),
            lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)
        before_state = optimizer_state_inventory(optimizer)
        save_json(directory / "optimizer_before.json", before_state)
        if before_state["tensor_count"] != 0:
            raise ValueError("Fresh AdamW must not have previous state")
        batch = collate_sft_batch([row], 151643, max_length=3761)
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        model.train()
        optimizer.zero_grad(set_to_none=True)
        phases.end()

        phases.begin("forward")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss).item() or not loss.requires_grad:
            raise ValueError("Nonfinite or disconnected loss")
        loss_value = float(loss.detach().item())
        save_json(directory / "loss.json", {"loss": loss_value, "supervised_tokens": row["supervised_token_count"]})
        phases.end()

        phases.begin("backward")
        loss.backward()
        phases.end()
        phases.begin("gradient_check")
        gradients = check_gradients(model)
        save_json(directory / "gradients.json", gradients)
        del loss
        phases.end()

        phases.begin("optimizer_step")
        optimizer.step()  # 本档唯一一次更新；这里还会首次创建AdamW状态。
        phases.end()
        phases.begin("state_inventory_and_zero_grad")
        after_state = optimizer_state_inventory(optimizer)
        save_json(directory / "optimizer_after.json", after_state)
        parameter_bytes = sum(p.numel() * p.element_size() for p in named.values())
        if after_state["parameter_states"] != 392 or after_state["cuda_bytes"] != 2 * parameter_bytes:
            raise ValueError("Unexpected AdamW moment state size")
        for moment in after_state["moments"].values():
            if moment["tensor_count"] != 392 or moment["bytes"] != parameter_bytes:
                raise ValueError("Incomplete optimizer moment state")
        optimizer.zero_grad(set_to_none=True)
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Gradients were not cleared after the update")
        del batch
        phases.end()

        phases.begin("save_and_audit")
        save_adapter_checked(model, directory / "adapter")
        if any(not torch.isfinite(p).all().item() for p in named.values()):
            raise ValueError("Nonfinite adapter parameter after update")
        changes = [{"name": n, "changed_elements": int(torch.count_nonzero(p.detach().cpu() - initial[n]))}
                   for n, p in named.items()]
        if not any(r["changed_elements"] for r in changes):
            raise ValueError("No adapter parameter changed")
        save_json(directory / "adapter_changes.json", changes)
        base_after = frozen_parameter_hash(model)
        save_json(directory / "base_parameter_hashes.json", {"before": base_before, "after": base_after})
        if base_after != base_before:
            raise ValueError("Frozen base parameters changed")
        check_hashes(manifest["protected_sha256"])
        phases.end()
        result = {"status": "pass", "tier": tier, "selected": manifest["selected"][index],
            "batch_size": 1, "step_calls": optimizer.step_calls, "completed_steps": optimizer.completed_steps,
            "partial_update_possible": False, "loss": loss_value, "gradient_l2_norm": gradients["global_l2_norm"],
            "gradients_all_finite": True, "adapter_saved_readback_equal": True, "frozen_base_unchanged": True,
            "optimizer_cuda_mib": after_state["cuda_bytes"] / 2**20, "optimizer_cpu_bytes": after_state["cpu_bytes"],
            "model_scope_seconds": time.perf_counter() - started}
        code = 0
    except Exception as error:
        failed_phase = phases.current or (phases.rows[-1]["phase"] if phases.rows else "before_first_phase")
        phases.end(failed=True)
        calls = optimizer.step_calls if optimizer is not None else 0
        completed = optimizer.completed_steps if optimizer is not None else 0
        is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        result = {"status": "oom" if is_oom else "error", "tier": tier, "selected": manifest["selected"][index],
            "failed_phase": failed_phase, "error_type": type(error).__name__, "error": str(error),
            "step_calls": calls, "completed_steps": completed, "partial_update_possible": calls > completed,
            "model_scope_seconds": time.perf_counter() - started}
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        code = 2
    finally:
        sampler.close()
        save_json(directory / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors})
    result.update(torch_peak_allocated_mib=max((r.get("peak_allocated_mib", 0) for r in phases.rows), default=0),
                  torch_peak_reserved_mib=max((r.get("peak_reserved_mib", 0) for r in phases.rows), default=0),
                  whole_gpu_sampled_peak_mib=max((r["used_mib"] for r in sampler.samples), default=None),
                  validation_examples_loaded=False, final_examples_read=False, truncated_examples=0)
    save_json(directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


def launch(tier):
    config, manifest, rows = load_checked()
    root = ROOT / config["run_root"]
    for previous in config["tiers"][:config["tiers"].index(tier)]:
        previous_result = root / previous / "result.json"
        if not previous_result.exists() or read_json(previous_result)["status"] != "pass":
            raise RuntimeError("Earlier tier has not passed; later tiers are blocked")
    directory = root / tier
    if directory.exists():
        raise FileExistsError("Tier already claimed; no automatic rerun")
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "launch_claim.json", {"tier": tier, "utc": datetime.now(timezone.utc).isoformat()})
    save_json(directory / "selection.json", manifest)
    for name in SOURCES + ["src/train/qlora_stability.py", "src/train/qlora_step.py", config["source_config"]]:
        dest = directory / "snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    env = dict(os.environ, QLORA_PROBE_CLAIM=str(directory.resolve()), PYTHONUTF8="1")
    started = time.perf_counter()
    try:
        with (directory / "console.log").open("w", encoding="utf-8") as output:
            process = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", tier],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=config["process_timeout_seconds"],
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        # subprocess.run会终止并等待直接子进程；本worker不会再派生训练进程。
        exit_code = 124
    if exit_code == 124 or not (directory / "result.json").exists():
        status_file = directory / "update_status.json"
        update = read_json(status_file) if status_file.exists() else {"step_calls": 0, "completed_steps": 0, "partial_update_possible": False}
        phase_file = directory / "current_phase.json"
        save_json(directory / "result.json", {"status": "timeout" if exit_code == 124 else "process_error",
            "tier": tier, **update, "last_phase": read_json(phase_file) if phase_file.exists() else None})
    result = read_json(directory / "result.json")
    if exit_code != 0 and result["status"] == "pass":
        result["status"] = "process_exit_error"
        save_json(directory / "result.json", result)
    save_json(directory / "process_timing.json", {"process_wall_seconds": time.perf_counter() - started,
        "exit_code": exit_code, "scope": "Worker process startup through exit; includes model, step, save and checks"})
    try:
        check_hashes(manifest["protected_sha256"])
    except Exception as error:
        result.update(status="integrity_error", integrity_error=str(error))
        save_json(directory / "result.json", result)
    save_json(directory / "artifact_sha256.json", {p.relative_to(directory).as_posix(): sha256(p)
        for p in directory.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 2


def verify():
    config, manifest, rows = load_checked()
    results, stopped = [], False
    for tier in config["tiers"]:
        directory = ROOT / config["run_root"] / tier
        if not directory.exists():
            results.append({"tier": tier, "status": "not_run"})
            stopped = True
            continue
        if stopped:
            raise ValueError("A later tier was run after a gap or failure")
        for name, digest in read_json(directory / "artifact_sha256.json").items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Probe artifact changed: {tier}/{name}")
        result = read_json(directory / "result.json")
        if not 0 <= result["completed_steps"] <= result["step_calls"] <= 1:
            raise ValueError("Update budget exceeded")
        if result["status"] == "pass" and (result["completed_steps"] != 1 or result["step_calls"] != 1):
            raise ValueError("Passed tier requires exactly one update")
        stopped = result["status"] != "pass"
        results.append({**result, "process": read_json(directory / "process_timing.json")})
    report = {"audit_status": "pass", "tiers": results, "protected_files_unchanged": True,
        "total_completed_updates": sum(r.get("completed_steps", 0) for r in results),
        "no_automatic_retry": True, "verification_loaded_model": False, "validation_examples_loaded": False,
        "final_examples_read": False, "formal_data_truncated_or_removed": False}
    save_json(ROOT / "reports/qlora_long_probe_result.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run-tier", choices=("1024", "2048", "3761"))
    group.add_argument("--worker", choices=("1024", "2048", "3761"), help=argparse.SUPPRESS)
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run_tier:
        sys.exit(launch(args.run_tier))
    elif args.worker:
        sys.exit(worker(args.worker))
    else:
        verify()
