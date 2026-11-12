"""CPU-only parent watchdog. Every PID sample is durable before entering its queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


DEFAULT_LIMITS = {
    "total_soft_seconds": 240.0, "total_hard_seconds": 300.0,
    "micro_soft_seconds": 60.0, "micro_hard_seconds": 90.0,
    "shared_limit_mib": 512.0, "shared_duration_seconds": 1.0,
    "resource_grace_seconds": 5.0, "sample_interval_seconds": 0.1,
    "telemetry_stale_seconds": 2.0,
}


def read_optional(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_durable(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


class BudgetWatch:
    """Deterministic budget policy; callers supply clock values and new samples."""
    def __init__(self, started, limits):
        self.started = float(started)
        self.limits = dict(DEFAULT_LIMITS, **limits)
        if any(not isinstance(v, (int, float)) or v <= 0 for v in self.limits.values()):
            raise ValueError("All watchdog limits must be positive numbers")
        if (self.limits["total_soft_seconds"] >= self.limits["total_hard_seconds"]
                or self.limits["micro_soft_seconds"] >= self.limits["micro_hard_seconds"]):
            raise ValueError("Soft limits must precede hard limits")
        self.required_since = None
        self.latest_sample = None
        self.shared_since = None
        self.request = None
        self.micro_deadline = None

    def trigger(self, now, reason, kind):
        # Resource failures get a short grace period even after an earlier time stop.
        if self.request is None or (kind == "resource" and self.request["kind"] != "resource"):
            self.request = {"reason": reason, "kind": kind, "monotonic": now}
        return self.request

    def evaluate(self, now, stage, samples=(), worker_stop=None):
        stage = stage or {}
        required = bool(stage.get("telemetry_required", False))
        if required and self.required_since is None:
            self.required_since = now
        if worker_stop:
            kind = worker_stop.get("kind", "resource")
            if kind not in ("resource", "time"):
                kind = "resource"
            when = min(now, max(self.started, float(worker_stop.get("monotonic", now))))
            self.trigger(when, worker_stop.get("reason", "worker_stop"), kind)
        for sample in samples:
            sample_time = float(sample["sampled_monotonic"])
            if (self.latest_sample is not None
                    and sample_time - self.latest_sample > max(0.25, 3 * self.limits["sample_interval_seconds"])):
                self.shared_since = None  # A sampling gap does not prove sustained occupancy.
            self.latest_sample = sample_time
            usable = sample.get("status") == "ok" and sample.get("shared_bytes") is not None
            if not usable:
                self.shared_since = None
                # A sample taken before the GPU context registered is not a failure.
                if required and sample_time >= self.required_since:
                    self.trigger(now, "PID telemetry became unavailable", "resource")
                continue
            if sample["shared_bytes"] > self.limits["shared_limit_mib"] * 2**20:
                if self.shared_since is None:
                    self.shared_since = sample_time
                if sample_time - self.shared_since >= self.limits["shared_duration_seconds"]:
                    self.trigger(now, "PID shared memory exceeded the sustained limit", "resource")
            else:
                self.shared_since = None
        if required:
            last = max(self.required_since, self.latest_sample or self.required_since)
            if now - last >= self.limits["telemetry_stale_seconds"]:
                self.trigger(now, "PID telemetry stalled", "resource")
        elapsed = now - self.started
        micro_start = stage.get("micro_started_monotonic")
        if micro_start is not None:
            micro_start = float(micro_start)
            if micro_start < self.started or micro_start > now:
                self.trigger(now, "Invalid microbatch clock", "resource")
            elif now - micro_start >= self.limits["micro_soft_seconds"]:
                self.trigger(now, "Microbatch soft time limit", "time")
                deadline = micro_start + self.limits["micro_hard_seconds"]
                self.micro_deadline = min(self.micro_deadline or deadline, deadline)
        if elapsed >= self.limits["total_soft_seconds"]:
            self.trigger(now, "Total soft time limit", "time")
        hard_reason = None
        if elapsed >= self.limits["total_hard_seconds"]:
            hard_reason = "Total hard time limit"
        elif self.micro_deadline is not None and now >= self.micro_deadline:
            hard_reason = "Microbatch hard time limit"
        elif (self.request and self.request["kind"] == "resource"
              and now - self.request["monotonic"] >= self.limits["resource_grace_seconds"]):
            hard_reason = "Resource stop grace expired"
        return {"request": self.request, "hard_reason": hard_reason}


def run_supervised(command, cwd, env, run_dir, limits, *, sample_provider=None):
    """Run one child once. sample_provider(pid,stage) is a CPU-test injection only."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "watchdog_result.json").exists() or (run_dir / "console.log").exists():
        raise FileExistsError("Watchdog output already exists; no automatic replay")
    started = time.perf_counter()
    watch = BudgetWatch(started, limits)
    finished = threading.Event()
    samples_queue = queue.Queue()
    thread_errors = []
    events = []
    killed = False
    controller_error = None
    process = None
    stage_path = run_dir / "stage.json"
    stop_path = run_dir / "stop_request.json"
    emitted_request = None

    def sampler_loop(pid):
        sampler = None
        try:
            if sample_provider is None:
                from src.train.windows_gpu_memory import WindowsGPUMemorySampler
                sampler = WindowsGPUMemorySampler(pid, watch.limits["sample_interval_seconds"])
            with (run_dir / "pid_memory.jsonl").open("x", encoding="utf-8") as stream:
                while not finished.is_set():
                    stage = read_optional(stage_path) or {}
                    sample_started = time.perf_counter()
                    if sampler is not None:
                        sampler.stage = stage.get("stage", "before_gpu_ready")
                        row = sampler.sample()
                    else:
                        row = dict(sample_provider(pid, stage))
                    row.update(sampled_monotonic=sample_started, pid=pid,
                               stage=stage.get("stage", "before_gpu_ready"))
                    append_durable(stream, row)
                    samples_queue.put(row)
                    remaining = watch.limits["sample_interval_seconds"] - (time.perf_counter() - sample_started)
                    finished.wait(max(0, remaining))
        except Exception as error:
            thread_errors.append(f"{type(error).__name__}: {error}")
        finally:
            # This thread owns the PDH handle; avoid taking another sample after child exit.
            if sampler is not None and sampler._query is not None:
                sampler._query.close()

    with (run_dir / "console.log").open("x", encoding="utf-8") as console, \
         (run_dir / "watchdog_events.jsonl").open("x", encoding="utf-8") as event_stream:
        def event(event_name, **details):
            row = {"event": event_name, "monotonic": time.perf_counter(), **details}
            events.append(row)
            append_durable(event_stream, row)

        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=console,
                                       stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            event("child_started", pid=process.pid)
            thread = threading.Thread(target=sampler_loop, args=(process.pid,),
                                      name="durable-pid-sampler", daemon=True)
            thread.start()
            while process.poll() is None:
                now = time.perf_counter()
                stage = read_optional(stage_path) or {}
                new_samples = []
                while True:
                    try:
                        new_samples.append(samples_queue.get_nowait())
                    except queue.Empty:
                        break
                if thread_errors:
                    watch.trigger(now, thread_errors[0], "resource")
                decision = watch.evaluate(now, stage, new_samples, read_optional(stop_path))
                if decision["request"] and decision["request"] != emitted_request:
                    write_atomic(stop_path, decision["request"])
                    emitted_request = dict(decision["request"])
                    event("stop_requested", **emitted_request)
                if decision["hard_reason"]:
                    event("child_kill", reason=decision["hard_reason"])
                    killed = True
                    process.kill()
                    break
                finished.wait(0.025)
            process.wait(timeout=10)
            event("child_exited", exit_code=process.returncode)
        except Exception as error:
            controller_error = f"{type(error).__name__}: {error}"
            event("controller_error", error=controller_error)
            if process is not None and process.poll() is None:
                killed = True
                process.kill()
                process.wait(timeout=10)
        finally:
            finished.set()
            if process is not None and "thread" in locals():
                thread.join(timeout=2)
                if thread.is_alive():
                    thread_errors.append("PID sampling thread failed to close within 2 seconds")
    rows = []
    sample_file = run_dir / "pid_memory.jsonl"
    if sample_file.exists():
        for line in sample_file.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                thread_errors.append("Incomplete durable sample line")
    valid = [row for row in rows if row.get("status") == "ok"]
    result = {
        "status": "pass" if (process is not None and process.returncode == 0 and not killed
                               and not watch.request and not thread_errors and controller_error is None) else "failed",
        "pid": None if process is None else process.pid,
        "exit_code": None if process is None else process.returncode,
        "wall_seconds": time.perf_counter() - started, "killed": killed,
        "stop_request": watch.request, "controller_error": controller_error,
        "telemetry_errors": thread_errors, "sample_count": len(rows), "valid_sample_count": len(valid),
        "pid_dedicated_peak_mib": max((r["dedicated_bytes"] / 2**20 for r in valid), default=None),
        "pid_shared_peak_mib": max((r["shared_bytes"] / 2**20 for r in valid), default=None),
        "limits": watch.limits, "events": events,
        "sample_source": "injected_cpu_fixture" if sample_provider else "Windows PDH target PID",
        "new_training_processes": 1 if process else 0, "automatic_retries": 0,
    }
    write_atomic(run_dir / "watchdog_result.json", result)
    return result
