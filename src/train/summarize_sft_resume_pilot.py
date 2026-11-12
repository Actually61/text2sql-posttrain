"""只读整理成功与失败段的现有证据；标准库，不导入Torch或重新执行训练。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/runs/sft_resume_pilot_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    artifacts = read(RUN / "artifact_sha256.json")
    for name, expected in artifacts.items():
        if digest(RUN / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    controller = read(RUN / "result.json")
    ledger = read(RUN / "progress.json")
    if ledger != controller["ledger"] or not 0 <= ledger["completed_updates"] <= ledger["attempted_updates"] <= 4:
        raise ValueError("Controller/ledger mismatch or update cap exceeded")
    segments, all_history, pids = [], [], []
    for number in (1, 2):
        directory = RUN / f"segment_{number}"
        if not directory.exists():
            continue
        phases = read(directory / "phase_memory.json") if (directory / "phase_memory.json").exists() else []
        history = read(directory / "history.json") if (directory / "history.json").exists() else []
        final = read(directory / "result.json") if (directory / "result.json").exists() else None
        pid = read(directory / "worker_started.json")["pid"]
        pids.append(pid)
        all_history.extend(history)
        micro_phases = [phase for phase in phases if "_micro_" in phase["phase"]]
        segment = {"segment": number, "pid": pid, "complete_result_exists": final is not None,
                   "completed_updates": [row["update"] for row in history], "history": history,
                   "completed_micro_intervals": len(micro_phases),
                   "recorded_phase_max_allocated_mib": max((p.get("peak_allocated_mib", 0) for p in phases), default=None),
                   "recorded_phase_max_reserved_mib": max((p.get("peak_reserved_mib", 0) for p in phases), default=None),
                   "memory_coverage": "complete worker phase records" if final else "lower bound from completed phases only; unfinished interval missing",
                   "slowest_completed_micro_intervals": sorted(micro_phases, key=lambda row: row["seconds"], reverse=True)[:3],
                   "current_phase": read(directory / "current_phase.json") if (directory / "current_phase.json").exists() else None,
                   "phase_time_scope": "microbatch training interval including synchronization/monitoring and minor next-input transfer overhead; not pure kernel time"}
        if final:
            segment.update(status=final["status"], process_memory=final["process_memory"],
                           whole_gpu_peak_mib=final["whole_gpu_peak_mib"], base_unchanged=final["base_unchanged"])
        if (directory / "process_timing.json").exists():
            segment["process_timing"] = read(directory / "process_timing.json")
        segments.append(segment)
    completed = [row["update"] for row in all_history]
    if completed != list(range(1, ledger["completed_updates"] + 1)):
        raise ValueError("Committed-update history does not match the ledger")
    checkpoints = []
    checkpoint_root = RUN / "checkpoints"
    for directory in sorted(checkpoint_root.glob("update_*")):
        if not (directory / "complete.json").exists():
            continue
        manifest = read(directory / "complete.json")
        for name, expected in manifest["files"].items():
            if digest(directory / name) != expected:
                raise ValueError("Checkpoint hash mismatch")
        metadata = read(directory / "metadata.json")
        checkpoints.append({"path": directory.relative_to(ROOT).as_posix(),
                            "completed_updates": metadata["completed_updates"],
                            "next_group_index": metadata["next_group_index"],
                            "next_update_lr": metadata["next_update_lr"], "integrity": "pass"})
    restored_path = RUN / "segment_2/restore_verification.json"
    restored = read(restored_path) if restored_path.exists() else None
    if restored:
        if len(set(pids)) != 2 or not all(restored["verification"][key] for key in
                ("adapter_exact", "optimizer_exact", "rng_exact", "parameter_order_exact", "cuda_rng_restored")):
            raise ValueError("Missing new-process exact-state restoration evidence")
    durable = max((item["completed_updates"] for item in checkpoints), default=0)
    report = {"overall_status": controller["status"], "termination": controller["error"],
              "controller_seconds": controller["controller_seconds"], "configured_worker_budget_seconds": 600,
              "ledger": ledger, "segments": segments, "process_ids": pids,
              "completed_update_examples": sum(len(row["source_indices"]) for row in all_history),
              "completed_update_target_tokens": sum(row["total_target_tokens"] for row in all_history),
              "complete_checkpoints": checkpoints, "latest_durable_update": durable,
              "completed_but_not_checkpointed_updates": ledger["completed_updates"] - durable,
              "gpu_resume_state_verified": restored is not None,
              "resume_verification": restored["verification"] if restored else None,
              "resume_next_group_index": restored["metadata"]["next_group_index"] if restored else None,
              "resume_next_update_lr": restored["metadata"]["next_update_lr"] if restored else None,
              "artifact_hashes_verified": len(artifacts),
              "full_run_memory_peak_known": all(segment["complete_result_exists"] for segment in segments),
              "interpretation": ["A four-update limit is not proof that four updates completed",
                 "The latest durable checkpoint is older than the live completed-update ledger; do not rewind or auto-replay",
                 "Active allocated memory fell after completed samples while reserved memory grew; caching/fragmentation is a hypothesis, not a proved root cause",
                 "Substantial shared usage was directly observed; paging as the cause of slowdown was not measured directly",
                 "Hard timeout prevents the second worker finally-block from saving complete telemetry; its missing peak cannot be reconstructed"],
              "scope": {"new_model_loaded_by_this_script": False, "new_optimizer_updates": 0,
                        "evaluation_performed": False, "final_examples_read": False}}
    destination = ROOT / "reports/sft_resume_pilot_analysis.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "completed_updates": ledger["completed_updates"],
                      "durable_update": durable, "gpu_resume_state_verified": report["gpu_resume_state_verified"],
                      "full_run_memory_peak_known": report["full_run_memory_peak_known"]}))


if __name__ == "__main__":
    main()
