"""CPU等价门槛通过后，完整3761-token题、答案位置logits、最多两步训练。"""
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
from src.train.qlora_long_probe import PhaseMemory, gpu_snapshot
from src.train.qlora_step import adapter_parameters, check_gradients
from src.train.long_probe_core import optimizer_state_inventory
from src.train.target_loss import target_position_loss
from src.train.target_probe_core import TwoUpdateAdamW
from src.train.windows_gpu_memory import WindowsGPUMemorySampler, telemetry_preflight

CONFIG = ROOT / "configs/qlora_target_probe.json"
SOURCES = ["configs/qlora_target_probe.json", "src/train/qlora_target_probe.py",
    "src/train/target_probe_core.py", "src/train/test_target_probe_core.py",
    "src/train/target_loss.py", "src/train/test_target_loss.py", "src/train/windows_gpu_memory.py"]


def prepare():
    config = read_json(CONFIG)
    if (config["source_index"], config["sequence_length"], config["batch_size"], config["max_updates"]) != (3698, 3761, 1, 2):
        raise ValueError("This experiment is fixed to the complete longest example and two updates")
    previous = read_json(ROOT / "data/processed/spider_long_probe_v2/manifest.json")
    protected = dict(previous["protected_sha256"])
    check_hashes(protected)
    cpu = read_json(ROOT / config["cpu_equivalence_report"])
    if cpu["status"] != "pass" or cpu["device"] != "cpu":
        raise ValueError("CPU loss/gradient equivalence must pass before GPU work")
    check_hashes(cpu["source_sha256"])
    for package, pinned in previous["packages"].items():
        if version(package) != pinned:
            raise ValueError(f"Frozen package changed: {package}")
    for tier in ("1024", "2048", "3761"):
        old = ROOT / "outputs/runs/qlora_long_probe_v2" / tier
        for name, digest in read_json(old / "artifact_sha256.json").items():
            if sha256(old / name) != digest:
                raise ValueError("Historical long-probe artifact changed")
        for path in old.rglob("*"):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()] = sha256(path)
    protected.update(cpu["source_sha256"])
    for name in SOURCES + [config["source_features"], config["cpu_equivalence_report"],
            "data/processed/spider_long_probe_v2/manifest.json", "reports/qlora_long_probe_result.json",
            "reports/target_probe_core_checks.json"]:
        protected[name] = sha256(ROOT / name)
    guard_checks = read_json(ROOT / "reports/target_probe_core_checks.json")
    if guard_checks["status"] != "pass":
        raise ValueError("Two-update CPU guard tests have not passed")
    check_hashes(guard_checks["source_sha256"])
    if sha256(ROOT / config["source_features"]) != previous["features_sha256"]:
        raise ValueError("Frozen full examples changed")
    row = next(r for r in read_json(ROOT / config["source_features"]) if r["source_index"] == 3698)
    batch = collate_sft_batch([row], 151643, max_length=3761)
    if (row["project_split"] != "train" or row["sequence_length"] != 3761
            or list(batch["input_ids"].shape) != [1, 3761]
            or int((batch["labels"][:, 1:] != -100).sum()) != 135
            or bool((batch["attention_mask"] == 0).any())):
        raise ValueError("Expected complete longest training example without padding")
    provider = telemetry_preflight()
    save_json(ROOT / "reports/target_probe_telemetry_preflight.json", provider)
    if provider["status"] != "available":
        raise RuntimeError("Dedicated/shared process counters unavailable; no GPU training")
    report = {"status": "pass", "config": config, "protected_sha256": protected,
        "packages": previous["packages"], "selected": {k: row[k] for k in
            ("source_index", "db_id", "sequence_length", "prompt_length", "supervised_token_count", "eos_position")},
        "cpu_equivalence_passed": True, "cpu_tolerance": {k: cpu[k] for k in ("absolute_tolerance", "relative_tolerance")},
        "input_truncated": False, "model_loaded": False, "validation_examples_loaded": False, "final_examples_read": False}
    publish_json(ROOT / config["preflight"], report)
    print(json.dumps({"status": "pass", "selected": report["selected"], "telemetry": provider}), flush=True)


def checked_inputs():
    config = read_json(CONFIG)
    preflight = read_json(ROOT / config["preflight"])
    if preflight["status"] != "pass" or config != preflight["config"]:
        raise ValueError("Prepared configuration changed")
    check_hashes(preflight["protected_sha256"])
    for name, pinned in preflight["packages"].items():
        if version(name) != pinned:
            raise ValueError(f"Frozen package changed: {name}")
    row = next(r for r in read_json(ROOT / config["source_features"]) if r["source_index"] == config["source_index"])
    return config, preflight, row


class SampleStages:
    """仅转发阶段标签，让PyTorch、整卡和PID采样使用相同阶段名。"""
    def __init__(self, whole, process):
        self.whole, self.process, self._stage = whole, process, "before_load"

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, value):
        self._stage = value
        self.whole.stage = value
        self.process.stage = value


def state_steps(optimizer):
    # 本实验AdamW的step标量在CPU上；这里只读取计数，不创建更新。
    return sorted({float(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def require_pid_sample(sampler):
    sample = sampler.sample()
    if sample["status"] != "ok":
        raise RuntimeError(f"Invalid PID dedicated/shared memory sample: {sample['status']}")
    return sample


def worker():
    config, preflight, row = checked_inputs()
    directory = ROOT / config["run_dir"]
    if os.environ.get("QLORA_TARGET_CLAIM") != str(directory.resolve()) or not (directory / "launch_claim.json").exists():
        raise RuntimeError("Worker must be launched through --run")
    if (directory / "worker_started.json").exists():
        raise FileExistsError("Worker already started; no retry")
    save_json(directory / "worker_started.json", {"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat()})
    resources = gpu_snapshot()
    save_json(directory / "resources.json", resources)
    if resources["free_mib"] < config["minimum_free_mib"]:
        save_json(directory / "result.json", {"status": "resource_blocked", "step_calls": 0, "completed_steps": 0, "resources": resources})
        return 2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA bf16 required")
    whole = GPUSampler(0.2)
    process = WindowsGPUMemorySampler(os.getpid(), interval=0.1)
    whole.start()
    process.start()
    phases = PhaseMemory(directory, SampleStages(whole, process))
    optimizer, model, history = None, None, []
    started = time.perf_counter()
    cleanup_errors = []
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
        process.sample()
        phases.begin("setup_and_base_hash")
        base_before = frozen_parameter_hash(model)
        named = adapter_parameters(model)
        parameter_bytes = sum(p.numel() * p.element_size() for p in named.values())
        optimizer = TwoUpdateAdamW(list(named.values()),
            audit_callback=lambda status: save_json(directory / "update_status.json", status),
            lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)
        save_json(directory / "initial_state.json", {"base": base_before, "fresh_lora_b_zero": True,
            "trainable_parameters": sum(p.numel() for p in named.values()), "adapter_parameter_bytes": parameter_bytes,
            "optimizer_created_once": True})
        batch = collate_sft_batch([row], 151643, max_length=3761)
        batch = {k: v.to("cuda:0") for k, v in batch.items()}
        model.train()
        phases.end()
        require_pid_sample(process)

        for step in (1, 2):
            phases.begin(f"step_{step}_before")
            optimizer.zero_grad(set_to_none=True)
            initial = {n: p.detach().cpu().clone() for n, p in named.items()}
            before_state = optimizer_state_inventory(optimizer)
            before_steps = state_steps(optimizer)
            expected_bytes = 0 if step == 1 else 2 * parameter_bytes
            if before_state["cuda_bytes"] != expected_bytes or before_steps != ([] if step == 1 else [1.0]):
                raise ValueError("Expected empty initial state or persistent first-step AdamW state")
            save_json(directory / f"step_{step}_optimizer_before.json", {**before_state, "state_steps": before_steps})
            phases.end()

            phases.begin(f"step_{step}_forward")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metadata = target_position_loss(model, batch)
            if not torch.isfinite(loss).item() or not loss.requires_grad:
                raise ValueError("Nonfinite or detached optimized loss")
            if metadata["input_shape"] != [1, 3761] or metadata["logits_shape"] != [1, 135, 151936]:
                raise ValueError("Unexpected complete context or target-only logits shape")
            if (metadata["valid_target_count"], metadata["first_predict_position"], metadata["last_predict_position"]) != (135, 3624, 3758):
                raise ValueError("Prediction positions no longer align with original SQL/EOS targets")
            loss_value = float(loss.detach().item())
            save_json(directory / f"step_{step}_forward.json", {"loss": loss_value, **metadata})
            phases.end()
            require_pid_sample(process)

            phases.begin(f"step_{step}_backward")
            loss.backward()
            phases.end()
            require_pid_sample(process)
            phases.begin(f"step_{step}_gradient_check")
            gradients = check_gradients(model)
            save_json(directory / f"step_{step}_gradients.json", gradients)
            del loss
            phases.end()

            phases.begin(f"step_{step}_optimizer_step")
            optimizer.step()
            phases.end()
            require_pid_sample(process)
            phases.begin(f"step_{step}_save_and_check")
            if optimizer.completed_steps != step:
                raise ValueError("Unexpected number of completed updates")
            after_state = optimizer_state_inventory(optimizer)
            after_steps = state_steps(optimizer)
            if (after_state["parameter_states"] != 392 or after_state["cuda_bytes"] != 2 * parameter_bytes
                    or after_steps != [float(step)]):
                raise ValueError("AdamW state was reset, incomplete or not advanced")
            save_json(directory / f"step_{step}_optimizer_after.json", {**after_state, "state_steps": after_steps})
            changes = []
            for name, parameter in named.items():
                current = parameter.detach().cpu()
                if not torch.isfinite(current).all().item():
                    raise ValueError("Nonfinite adapter after optimizer step")
                changes.append({"name": name, "changed_elements": int(torch.count_nonzero(current - initial[name]))})
            if not any(item["changed_elements"] for item in changes):
                raise ValueError("No adapter parameter changed")
            save_json(directory / f"step_{step}_adapter_changes.json", changes)
            optimizer.zero_grad(set_to_none=True)
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError("Gradient cleanup failed")
            save_adapter_checked(model, directory / f"adapter_step_{step}")
            record = {"step": step, "source_index": row["source_index"], "loss": loss_value,
                "gradient_l2_norm": gradients["global_l2_norm"], "gradients_all_finite": gradients["all_finite"],
                "state_cuda_mib_before_forward": before_state["cuda_bytes"] / 2**20,
                "state_cuda_mib_after_update": after_state["cuda_bytes"] / 2**20,
                "a_changed_tensors": sum(c["changed_elements"] > 0 for c in changes if ".lora_A." in c["name"]),
                "b_changed_tensors": sum(c["changed_elements"] > 0 for c in changes if ".lora_B." in c["name"]),
                "logits": metadata, "adapter_saved_readback_equal": True}
            history.append(record)
            save_json(directory / "history.json", history)
            del initial, changes
            phases.end()
            require_pid_sample(process)
            if process.report()["status"] != "available" or process.report()["errors"]:
                raise RuntimeError("Dedicated/shared counter collection failed; stop remaining updates")
            print(json.dumps(record, ensure_ascii=False), flush=True)

        phases.begin("final_audit")
        base_after = frozen_parameter_hash(model)
        save_json(directory / "base_parameter_hashes.json", {"before": base_before, "after": base_after})
        if base_before != base_after:
            raise ValueError("Frozen base changed")
        check_hashes(preflight["protected_sha256"])
        phases.end()
        result = {"status": "pass", "step_calls": optimizer.step_calls, "completed_steps": optimizer.completed_steps,
            "partial_update_possible": False, "selected": preflight["selected"], "history": history,
            "frozen_base_unchanged": True, "optimizer_reused": True, "model_scope_seconds": time.perf_counter() - started}
        code = 0
    except Exception as error:
        failed_phase = phases.current or (phases.rows[-1]["phase"] if phases.rows else "before_first_phase")
        phases.end(failed=True)
        calls = optimizer.step_calls if optimizer is not None else 0
        completed = optimizer.completed_steps if optimizer is not None else 0
        result = {"status": "oom" if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower() else "error",
            "step_calls": calls, "completed_steps": completed, "partial_update_possible": calls > completed,
            "failed_phase": failed_phase, "error": str(error), "error_type": type(error).__name__, "history": history}
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        code = 2
    finally:
        for sampler in (process, whole):
            try:
                sampler.close()
            except Exception as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        save_json(directory / "gpu_whole_samples.json", {"samples": whole.samples, "errors": whole.errors})
        save_json(directory / "gpu_process_memory.json", process.report())
    process_report = process.report()
    if result["status"] == "pass" and (cleanup_errors or process_report["errors"]):
        result.update(status="telemetry_error", cleanup_errors=cleanup_errors)
        code = 2
    result.update(torch_peak_allocated_mib=max((p.get("peak_allocated_mib", 0) for p in phases.rows), default=0),
        torch_peak_reserved_mib=max((p.get("peak_reserved_mib", 0) for p in phases.rows), default=0),
        whole_gpu_sampled_peak_mib=max((s["used_mib"] for s in whole.samples), default=None),
        process_memory={k: v for k, v in process_report.items() if k not in ("samples", "errors")},
        no_truncation_or_formal_data_deletion=True, validation_examples_loaded=False, final_examples_read=False,
        generation_or_ex_evaluation=False, formal_full_training=False,
        second_step_is_limited_persistent_state_check=True, long_term_steady_state_validated=False)
    save_json(directory / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


def launch():
    config, preflight, row = checked_inputs()
    directory = ROOT / config["run_dir"]
    if directory.exists():
        raise FileExistsError("Experiment already claimed; no rerun")
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "launch_claim.json", {"utc": datetime.now(timezone.utc).isoformat(), "max_updates": 2})
    save_json(directory / "preflight.json", preflight)
    for name in SOURCES + [config["source_config"], config["cpu_equivalence_report"], "src/train/qlora_stability.py",
            "src/train/qlora_long_probe.py", "src/train/long_probe_core.py", "src/train/qlora_step.py", "src/data/sft_features.py"]:
        destination = directory / "snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    started = time.perf_counter()
    try:
        with (directory / "console.log").open("w", encoding="utf-8") as output:
            proc = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker"],
                cwd=ROOT, env=dict(os.environ, QLORA_TARGET_CLAIM=str(directory.resolve()), PYTHONUTF8="1"),
                stdout=output, stderr=subprocess.STDOUT, timeout=config["process_timeout_seconds"],
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    worker_elapsed = time.perf_counter() - started
    if exit_code == 124 or not (directory / "result.json").exists():
        update_path = directory / "update_status.json"
        counts = read_json(update_path) if update_path.exists() else {"step_calls": 0, "completed_steps": 0, "partial_update_possible": False}
        save_json(directory / "result.json", {"status": "timeout" if exit_code == 124 else "process_error", **counts})
    result = read_json(directory / "result.json")
    if result["status"] == "pass" and exit_code != 0:
        result["status"] = "process_exit_error"
    try:
        check_hashes(preflight["protected_sha256"])
    except Exception as error:
        result.update(status="integrity_error", integrity_error=str(error))
    save_json(directory / "result.json", result)
    save_json(directory / "process_timing.json", {"worker_wall_seconds": worker_elapsed,
        "exit_code": exit_code, "scope": "Worker startup through exit; includes imports, checks, loading, two updates, logging, saving"})
    save_json(directory / "artifact_sha256.json", {p.relative_to(directory).as_posix(): sha256(p)
        for p in directory.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 2


def verify():
    config, preflight, row = checked_inputs()
    directory = ROOT / config["run_dir"]
    for name, digest in read_json(directory / "artifact_sha256.json").items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Run artifact changed: {name}")
    result = read_json(directory / "result.json")
    if not 0 <= result["completed_steps"] <= result["step_calls"] <= 2:
        raise ValueError("Update cap violated")
    if result["status"] == "pass":
        if result["completed_steps"] != 2 or [r["step"] for r in result["history"]] != [1, 2]:
            raise ValueError("Missing completed two-step evidence")
        if [r["state_cuda_mib_before_forward"] for r in result["history"]] != [0.0, 70.4375]:
            raise ValueError("Second forward did not retain first-step optimizer state")
        samples = read_json(directory / "gpu_process_memory.json")
        if samples["status"] != "available" or samples["matched_sample_count"] == 0:
            raise ValueError("No valid per-process dedicated/shared measurements")
    save_json(ROOT / "reports/qlora_target_probe_result.json", {**result,
        "process_timing": read_json(directory / "process_timing.json"), "artifact_audit": "pass",
        "verification_loaded_model": False})
    print(json.dumps({"artifact_audit": "pass", "status": result["status"], "completed_steps": result["completed_steps"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        sys.exit(launch())
    elif args.worker:
        sys.exit(worker())
    else:
        verify()
