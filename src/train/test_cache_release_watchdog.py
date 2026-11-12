"""Watchdog policy and durable kill-log checks; no Torch or CUDA imports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from src.train.cache_release_watchdog import BudgetWatch, run_supervised


def sample(at, shared=0, status="ok"):
    return {"sampled_monotonic": at, "status": status,
            "shared_bytes": shared * 2**20 if status == "ok" else None,
            "dedicated_bytes": 100 * 2**20 if status == "ok" else None}


def main():
    checks = []
    watch = BudgetWatch(0, {})
    assert watch.evaluate(59, {"micro_started_monotonic": 0})["request"] is None
    decision = watch.evaluate(60, {"micro_started_monotonic": 0})
    assert decision["request"]["kind"] == "time" and decision["hard_reason"] is None
    assert watch.evaluate(90, {})["hard_reason"] == "Microbatch hard time limit"
    checks.append("micro_soft_60_hard_90_deadline_survives_stage_exit")

    watch = BudgetWatch(0, {})
    assert watch.evaluate(240, {})["request"]["kind"] == "time"
    assert watch.evaluate(300, {})["hard_reason"] == "Total hard time limit"
    checks.append("total_soft_240_hard_300")

    watch = BudgetWatch(0, {})
    for n in range(11):
        decision = watch.evaluate(n / 10, {}, [sample(n / 10, 513)])
    assert decision["request"]["kind"] == "resource"
    assert watch.evaluate(5.99, {})["hard_reason"] is None
    assert watch.evaluate(6.0, {})["hard_reason"] == "Resource stop grace expired"
    checks.append("shared_above_512_for_one_second_then_five_second_grace")

    watch = BudgetWatch(0, {})
    for n in range(21):
        assert watch.evaluate(n / 10, {}, [sample(n / 10, 512)])["request"] is None
    watch = BudgetWatch(0, {})
    watch.evaluate(0, {}, [sample(0, 513)])
    assert watch.evaluate(1.1, {}, [sample(1.1, 513)])["request"] is None
    checks.append("shared_exact_limit_allowed_and_sample_gaps_reset_duration")

    watch = BudgetWatch(0, {})
    assert watch.evaluate(0, {}, [sample(0, status="not_registered")])["request"] is None
    watch.evaluate(1, {"telemetry_required": True})
    assert watch.evaluate(1.1, {"telemetry_required": True}, [sample(1.1, status="unavailable")])["request"]["kind"] == "resource"
    watch = BudgetWatch(0, {})
    assert watch.evaluate(1, {"telemetry_required": True})["request"] is None
    assert watch.evaluate(3, {"telemetry_required": True})["request"]["reason"] == "PID telemetry stalled"
    checks.append("unregistered_only_before_required_and_stale_samples_stop")

    watch = BudgetWatch(0, {})
    watch.evaluate(240, {})
    decision = watch.evaluate(241, {}, worker_stop={"reason": "allocation peak", "kind": "resource", "monotonic": 240.5})
    assert decision["request"]["kind"] == "resource"
    assert watch.evaluate(245.5, {})["hard_reason"] == "Resource stop grace expired"
    checks.append("worker_resource_stop_overrides_time_stop_and_preserves_timestamp")

    attempt = ROOT / "outputs/checks/cache_release_watchdog_cpu" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt.mkdir(parents=True, exist_ok=False)
    child_code = "import time; print('cpu dummy started', flush=True); time.sleep(10)"
    result = run_supervised([sys.executable, "-c", child_code], ROOT, dict(os.environ), attempt / "hard_timeout",
                            {"total_soft_seconds": 0.3, "total_hard_seconds": 0.65,
                             "sample_interval_seconds": 0.025},
                            sample_provider=lambda pid, stage: {"status": "ok", "dedicated_bytes": 0, "shared_bytes": 0})
    assert result["status"] == "failed" and result["killed"] and result["automatic_retries"] == 0
    rows = [json.loads(line) for line in (attempt / "hard_timeout/pid_memory.jsonl").read_text().splitlines()]
    assert len(rows) >= 2 and any(event["event"] == "stop_requested" for event in result["events"])
    assert any(event["event"] == "child_kill" for event in result["events"])
    assert result["wall_seconds"] < 5
    checks.append("real_dummy_child_hard_kill_retains_incremental_telemetry_without_finally")
    try:
        run_supervised([sys.executable, "-c", child_code], ROOT, dict(os.environ), attempt / "hard_timeout", {})
    except FileExistsError:
        checks.append("same_run_directory_cannot_relaunch")
    else:
        raise AssertionError("Replay accepted")

    assert "torch" not in sys.modules
    paths = ["src/train/cache_release_watchdog.py", "src/train/test_cache_release_watchdog.py"]
    report = {"status": "pass", "checks": checks, "check_count": len(checks),
              "torch_imported": False, "cuda_initialized": False, "real_gpu_training": False,
              "attempt_dir": str(attempt), "dummy_child": result,
              "previous_attempt_directories": [str(p) for p in sorted(attempt.parent.iterdir()) if p.is_dir() and p != attempt],
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}}
    report_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (attempt / "report.json").write_text(report_text, encoding="utf-8")
    report_path = ROOT / "reports/cache_release_watchdog_cpu_checks.json"
    if report_path.exists():
        raise FileExistsError("Published report already exists")
    report_path.write_text(report_text, encoding="utf-8")
    print(json.dumps({"status": "pass", "checks": len(checks), "dummy_timeout_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
