"""固定32题，最多4次真实更新：独立进程2步、保存退出、独立进程恢复2步。"""
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
# Dependency imports may probe CUDA. CPU controller/verification processes must
# not hold a GPU context beside the worker. Restore the original visibility only
# in the explicitly launched training children below.
ORIGINAL_CUDA_VISIBILITY = os.environ.get("CUDA_VISIBLE_DEVICES")
if "--worker" not in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
from src.data.sft_features import collate_sft_batch
from src.infer.baseline_pilot import GPUSampler, save_json
from src.train.qlora_smoke import frozen_parameter_hash
from src.train.qlora_stability import fresh_model
from src.train.qlora_long_probe import PhaseMemory, gpu_snapshot
from src.train.qlora_target_probe import SampleStages, require_pid_sample
from src.train.qlora_step import adapter_parameters
from src.train.sft_round1_plan import read, digest, write, verify_hashes, merge_hashes, learning_rate_for_update
from src.train.sft_pilot_data import select_pilot
from src.train.sft_checkpoint import save_checkpoint, load_checkpoint, inspect_checkpoint
from src.train.sft_update_budget import JournaledAdamW, read_ledger, write_atomic
from src.train.sft_train import train_segment, adam_steps
from src.train.long_probe_core import optimizer_state_inventory
from src.train.windows_gpu_memory import WindowsGPUMemorySampler, telemetry_preflight

CONFIG = ROOT / "configs/sft_resume_pilot.json"
SOURCES = ["configs/sft_resume_pilot.json", "src/train/sft_resume_pilot.py", "src/train/sft_train.py",
           "src/train/sft_checkpoint.py", "src/train/test_sft_checkpoint.py", "src/train/sft_update_budget.py",
           "src/train/test_sft_update_budget.py", "src/train/sft_pilot_data.py", "src/train/test_sft_pilot_data.py"]


def prepare():
    config = read(CONFIG)
    if (config["mode"], config["selected_examples"], config["batch_size"], config["accumulation_steps"],
            config["global_update_budget"], config["segment_ranges"]) != ("bounded_resume_pilot", 32, 1, 8, 4, [[0, 2], [2, 4]]):
        raise ValueError("Only the authorized 32-example, 2+2 update pilot is supported here")
    prior = read(ROOT / config["formal_preflight"])
    formal = read(ROOT / config["formal_config"])
    if prior["status"] != "pass" or formal != prior["config"]:
        raise ValueError("First-round CPU preflight/config is not frozen")
    protected = dict(prior["protected_sha256"])
    verify_hashes(protected)
    packages = dict(prior["packages"])
    for name, pinned in packages.items():
        if version(name) != pinned:
            raise ValueError(f"Frozen package changed: {name}")
    for name in config["cpu_reports"]:
        report = read(ROOT / name)
        if report["status"] != "pass":
            raise ValueError(f"CPU prerequisite failed: {name}")
        merge_hashes(protected, report["source_sha256"])
        merge_hashes(protected, {name: digest(ROOT / name)})
    for name in SOURCES + [config["formal_preflight"], config["formal_config"]]:
        merge_hashes(protected, {name: digest(ROOT / name)})
    verify_hashes(protected)
    with (ROOT / formal["data"]["train"]).open(encoding="utf-8") as stream:
        all_rows = [json.loads(line) for line in stream]
    selection = select_pilot(all_rows, config["seed"])
    if selection != select_pilot(all_rows, config["seed"]):
        raise ValueError("Selection is not reproducible")
    by_id = {row["source_index"]: row for row in all_rows}
    features = [by_id[index] for index in selection["selected_source_indices"]]
    for row in features:
        batch = collate_sft_batch([row], 151643, 4096)
        if list(batch["input_ids"].shape) != [1, row["sequence_length"]]:
            raise ValueError("Selected full input changed")
    directory = ROOT / config["prepared_dir"]
    write(directory / "selection.json", selection, immutable=True)
    write(directory / "features.json", features, immutable=True)
    for path in (directory / "selection.json", directory / "features.json"):
        merge_hashes(protected, {path.relative_to(ROOT).as_posix(): digest(path)})
    telemetry = telemetry_preflight()
    if telemetry["status"] != "available":
        raise RuntimeError("Dedicated/shared memory counters unavailable")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU preparation unexpectedly initialized CUDA")
    report = {"status": "pass", "config": config, "formal_config": formal, "packages": packages,
              "protected_sha256": protected, "selection": selection,
              "schedule_sha256": digest(directory / "selection.json"), "run_config_sha256": digest(CONFIG),
              "telemetry_preflight": telemetry, "model_loaded": False, "cuda_initialized": False,
              "validation_examples_loaded": False, "final_examples_read": False,
              "previous_prepare_attempts": ["reports/sft_resume_pilot_prepare_attempts/attempt_1/report.json"]}
    write(ROOT / config["preflight_report"], report, immutable=True)
    print(json.dumps({"status": "pass", "selected": len(features), "input_tokens": selection["input_tokens"],
                      "target_tokens": selection["target_tokens"], "required_examples": selection["required_examples"]}), flush=True)


def checked_inputs():
    config = read(CONFIG)
    preflight = read(ROOT / config["preflight_report"])
    if preflight["status"] != "pass" or config != preflight["config"]:
        raise ValueError("Pilot configuration changed after preparation")
    verify_hashes(preflight["protected_sha256"])
    for name, pinned in preflight["packages"].items():
        if version(name) != pinned:
            raise ValueError(f"Frozen package changed: {name}")
    rows = read(ROOT / config["prepared_dir"] / "features.json")
    return config, preflight, {row["source_index"]: row for row in rows}


def checkpoint_metadata(preflight, completed, base_hash):
    training = preflight["formal_config"]["training"]
    schedule = training["scheduler"]
    return {"completed_updates": completed, "next_group_index": completed,
            "next_update_lr": learning_rate_for_update(completed + 1, schedule["total_updates"],
                                                       schedule["warmup_updates"], training["optimizer"]["lr"]),
            "schedule_sha256": preflight["schedule_sha256"], "run_config_sha256": preflight["run_config_sha256"],
            "base_hash": base_hash, "protected_source_sha256": preflight["protected_sha256"]}


def worker(segment):
    config, preflight, rows = checked_inputs()
    root = ROOT / config["run_dir"]
    directory = root / f"segment_{segment}"
    if segment not in (1, 2) or os.environ.get("SFT_PILOT_CLAIM") != str(root.resolve()):
        raise RuntimeError("Worker must be launched by the single-use pilot controller")
    with (directory / "worker_started.json").open("x", encoding="utf-8") as output:
        json.dump({"pid": os.getpid(), "segment": segment, "utc": datetime.now(timezone.utc).isoformat()}, output)
    start, end = config["segment_ranges"][segment - 1]
    ledger_path = root / "progress.json"
    ledger = read_ledger(ledger_path)
    if ledger["attempted_updates"] != start or ledger["completed_updates"] != start or ledger.get("failed"):
        raise ValueError("Segment cannot rewind or replay the global ledger")
    resources = gpu_snapshot()
    save_json(directory / "resources.json", resources)
    if resources["free_mib"] < config["minimum_free_mib"]:
        raise RuntimeError("Insufficient free memory before starting this segment")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    whole = GPUSampler(0.2)
    process = WindowsGPUMemorySampler(os.getpid(), interval=0.1)
    whole.start()
    process.start()
    phases = PhaseMemory(directory, SampleStages(whole, process))
    started = time.perf_counter()
    cleanup_errors = []
    optimizer, model, history = None, None, []
    try:
        import numpy as np
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        phases.begin("load_and_prepare")
        formal = preflight["formal_config"]
        smoke = read(ROOT / formal["model"]["inherit_loading_and_lora_from"])
        model = fresh_model(smoke)
        named = adapter_parameters(model)
        base_before = frozen_parameter_hash(model)
        optimizer_kwargs = {k: v for k, v in formal["training"]["optimizer"].items() if k != "name"}
        optimizer = JournaledAdamW(list(named.values()), ledger_path=ledger_path, start=start,
                                  segment_end=end, total_budget=4, **optimizer_kwargs)
        phases.end()

        def check_memory():
            require_pid_sample(process)
            if process.report()["errors"] or whole.errors:
                raise RuntimeError("GPU memory telemetry failed; stop without retry")

        check_memory()
        if start:
            phases.begin("restore_complete_checkpoint")
            restored = load_checkpoint(root / "checkpoints" / "update_000002", model, optimizer,
                                       checkpoint_metadata(preflight, start, base_before))
            if (adam_steps(optimizer) != [2.0]
                    or optimizer_state_inventory(optimizer)["cuda_bytes"] != 73859072
                    or not restored["verification"]["cuda_rng_restored"]):
                raise ValueError("CUDA Adam/RNG state was not fully restored")
            save_json(directory / "restore_verification.json", restored)
            phases.end()
        else:
            if optimizer.state:
                raise ValueError("First segment must start with a fresh optimizer")
        save_json(directory / "initial_state.json", {"base_hash": base_before, "completed_updates": start,
                  "adam_steps": adam_steps(optimizer), "optimizer": optimizer_state_inventory(optimizer),
                  "trainable_parameters": sum(p.numel() for p in named.values())})
        check_memory()

        def save_boundary(update):
            if update != end:
                return
            phases.begin(f"save_complete_checkpoint_{update}")
            saved = save_checkpoint(root / "checkpoints" / f"update_{update:06d}", model, optimizer,
                                    checkpoint_metadata(preflight, update, base_before))
            if not saved["cuda_rng_saved"]:
                raise ValueError("CUDA RNG was not captured")
            save_json(directory / "checkpoint_save.json", saved)
            phases.end()
            check_memory()

        history = train_segment(model, optimizer, rows, preflight["selection"]["groups"], start, end,
                                formal["training"], directory, phases, check_memory, save_boundary)
        phases.begin("final_audit")
        base_after = frozen_parameter_hash(model)
        if base_before != base_after:
            raise ValueError("Frozen base changed")
        save_json(directory / "base_hashes.json", {"before": base_before, "after": base_after})
        verify_hashes(preflight["protected_sha256"])
        phases.end()
        check_memory()
        result = {"status": "pass", "segment": segment, "start": start, "end": end,
                  "local_completed_updates": end - start, "global_completed_updates": optimizer.completed_updates,
                  "history": history, "base_unchanged": True, "model_scope_seconds": time.perf_counter() - started}
        code = 0
    except Exception as error:
        failed_phase = phases.current or "between_phases"
        phases.end(failed=True)
        ledger = read_ledger(ledger_path)
        ledger.update(failed=True, error=str(error), segment=segment)
        write_atomic(ledger_path, ledger)
        result = {"status": "oom" if isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower() else "failed",
                  "segment": segment, "failed_phase": failed_phase, "error": str(error), "ledger": ledger,
                  "history": read(directory / "history.json") if (directory / "history.json").exists() else []}
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        code = 2
    finally:
        for sampler in (process, whole):
            try:
                sampler.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        save_json(directory / "gpu_process_memory.json", process.report())
        save_json(directory / "gpu_whole_samples.json", {"samples": whole.samples, "errors": whole.errors})
    pid_memory = process.report()
    if result["status"] == "pass" and (cleanup_errors or pid_memory["errors"] or whole.errors):
        result.update(status="telemetry_error", cleanup_errors=cleanup_errors)
        ledger = read_ledger(ledger_path)
        ledger.update(failed=True, error="Telemetry cleanup failure")
        write_atomic(ledger_path, ledger)
        code = 2
    result.update(torch_peak_allocated_mib=max((r.get("peak_allocated_mib", 0) for r in phases.rows), default=0),
                  torch_peak_reserved_mib=max((r.get("peak_reserved_mib", 0) for r in phases.rows), default=0),
                  whole_gpu_peak_mib=max((s["used_mib"] for s in whole.samples), default=None),
                  process_memory={k: v for k, v in pid_memory.items() if k not in ("samples", "errors")},
                  validation_examples_loaded=False, final_examples_read=False, full_training=False)
    save_json(directory / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "segment", "torch_peak_allocated_mib", "torch_peak_reserved_mib")}), flush=True)
    return code


def launch():
    config, preflight, rows = checked_inputs()
    resources = gpu_snapshot()
    if resources["free_mib"] < config["minimum_free_mib"] or shutil.disk_usage(ROOT).free < config["minimum_disk_gib"] * 2**30:
        raise RuntimeError("Resource gate failed before claiming the run; no model or update started")
    root = ROOT / config["run_dir"]
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "claim.json", {"utc": datetime.now(timezone.utc).isoformat(), "global_update_budget": 4})
    write_atomic(root / "progress.json", {"attempted_updates": 0, "completed_updates": 0, "total_budget": 4,
                                         "partial_update_possible": False, "failed": False})
    save_json(root / "preflight.json", preflight)
    for name in SOURCES + [config["formal_config"], "src/train/token_accumulation.py", "src/train/target_loss.py"]:
        target = root / "snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    started = time.perf_counter()
    results = []
    status, failure = "pass", None
    try:
        for segment in (1, 2):
            start, end = config["segment_ranges"][segment - 1]
            ledger = read_ledger(root / "progress.json")
            if (ledger["attempted_updates"] != start or ledger["completed_updates"] != start
                    or ledger["failed"] or ledger["partial_update_possible"]):
                raise RuntimeError("Global ledger does not allow this segment")
            if start:
                checkpoint = inspect_checkpoint(root / "checkpoints" / "update_000002")
                if checkpoint["metadata"]["completed_updates"] != 2:
                    raise ValueError("Cannot resume an incomplete checkpoint")
            verify_hashes(preflight["protected_sha256"])
            remaining = config["total_worker_budget_seconds"] - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("Global run time budget exhausted")
            directory = root / f"segment_{segment}"
            directory.mkdir(exist_ok=False)
            segment_start = time.perf_counter()
            worker_env = dict(os.environ, SFT_PILOT_CLAIM=str(root.resolve()), PYTHONUTF8="1")
            if ORIGINAL_CUDA_VISIBILITY is None:
                worker_env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                worker_env["CUDA_VISIBLE_DEVICES"] = ORIGINAL_CUDA_VISIBILITY
            with (directory / "console.log").open("w", encoding="utf-8") as output:
                result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker", str(segment)],
                    cwd=ROOT, env=worker_env,
                    stdout=output, stderr=subprocess.STDOUT, timeout=remaining,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            seconds = time.perf_counter() - segment_start
            save_json(directory / "process_timing.json", {"worker_wall_seconds": seconds, "exit_code": result.returncode})
            if not (directory / "result.json").exists():
                raise RuntimeError(f"Segment {segment} ended without a result; inspect console.log")
            report = read(directory / "result.json")
            results.append({"segment": segment, "status": report["status"], "worker_wall_seconds": seconds})
            if result.returncode != 0 or report["status"] != "pass" or report["global_completed_updates"] != end:
                raise RuntimeError(f"Segment {segment} failed; no retry or later segment")
            print(json.dumps(results[-1]), flush=True)
    except Exception as error:
        status, failure = "failed", f"{type(error).__name__}: {error}"
        ledger = read_ledger(root / "progress.json")
        ledger.update(failed=True, controller_error=failure)
        write_atomic(root / "progress.json", ledger)
        (root / "controller_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    try:
        verify_hashes(preflight["protected_sha256"])
    except Exception as error:
        status, failure = "integrity_error", str(error)
    summary = {"status": status, "error": failure, "ledger": read_ledger(root / "progress.json"),
               "segments": results, "controller_seconds": time.perf_counter() - started,
               "full_training": False, "evaluation_performed": False, "final_examples_read": False}
    save_json(root / "result.json", summary)
    save_json(root / "artifact_sha256.json", {p.relative_to(root).as_posix(): digest(p)
              for p in root.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"})
    print(json.dumps(summary), flush=True)
    return 0 if status == "pass" else 2


def verify():
    config, preflight, rows = checked_inputs()
    root = ROOT / config["run_dir"]
    for name, expected in read(root / "artifact_sha256.json").items():
        if digest(root / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    summary = read(root / "result.json")
    ledger = summary["ledger"]
    if not 0 <= ledger["completed_updates"] <= ledger["attempted_updates"] <= 4:
        raise ValueError("Global update limit violated")
    if summary["status"] == "pass":
        segments = [read(root / f"segment_{i}" / "result.json") for i in (1, 2)]
        pids = [read(root / f"segment_{i}" / "worker_started.json")["pid"] for i in (1, 2)]
        history = [record for segment in segments for record in segment["history"]]
        seen = [index for record in history for index in record["source_indices"]]
        if (ledger["completed_updates"] != 4 or ledger["failed"] or len(set(pids)) != 2
                or [r["update"] for r in history] != [1, 2, 3, 4]
                or seen != preflight["selection"]["selected_source_indices"] or len(set(seen)) != 32):
            raise ValueError("Missing process separation or exact 32-example/four-update evidence")
        restored = read(root / "segment_2" / "restore_verification.json")
        for key in ("adapter_exact", "optimizer_exact", "rng_exact", "parameter_order_exact", "cuda_rng_restored"):
            if restored["verification"][key] is not True:
                raise ValueError(f"Resume check failed: {key}")
        checkpoints = {str(i): inspect_checkpoint(root / "checkpoints" / f"update_{i:06d}")["metadata"] for i in (2, 4)}
        for segment in segments:
            if not segment["base_unchanged"] or segment["process_memory"]["matched_sample_count"] == 0:
                raise ValueError("Missing frozen-base or GPU-memory evidence")
        summary.update(history=history, segment_results=segments, process_ids=pids,
                       restore_verification=restored["verification"], checkpoint_metadata=checkpoints,
                       selected_input_tokens=preflight["selection"]["input_tokens"],
                       selected_target_tokens=preflight["selection"]["target_tokens"],
                       torch_peak_allocated_mib=max(s["torch_peak_allocated_mib"] for s in segments),
                       torch_peak_reserved_mib=max(s["torch_peak_reserved_mib"] for s in segments))
    summary.update(artifact_audit="pass", verification_loaded_model=False)
    save_json(ROOT / config["result_report"], summary)
    print(json.dumps({"status": summary["status"], "artifact_audit": "pass", "ledger": ledger}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--worker", type=int, choices=(1, 2), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        sys.exit(launch())
    elif args.worker:
        sys.exit(worker(args.worker))
    else:
        verify()
