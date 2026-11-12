"""CPU-only first-SFT plan: validate every full training row and freeze its update schedule."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.sft_features import collate_sft_batch

CONFIG = ROOT / "configs/sft_round1.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value, immutable=False):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise ValueError(f"Refuse to replace a different frozen schedule: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def merge_hashes(protected, additional):
    for name, expected in additional.items():
        if name in protected and protected[name] != expected:
            raise ValueError(f"Conflicting frozen hashes; do not overwrite history: {name}")
        protected[name] = expected


def learning_rate_for_update(update, total=796, warmup=24, peak=1e-4):
    """One-based next update; the last real update has positive LR, next is zero."""
    if not 0 < warmup < total or update < 1:
        raise ValueError("Invalid update/schedule boundary")
    if update > total:
        return 0.0
    if update <= warmup:
        return peak * update / warmup
    return peak * (total - update + 1) / (total - warmup + 1)


def build_groups(rows, upper_bounds, group_size, seed):
    """Shuffle length buckets without losing their remainders; only final group may be short."""
    if group_size < 1 or sorted(set(upper_bounds)) != upper_bounds:
        raise ValueError("Invalid buckets/group size")
    ids = [row["source_index"] for row in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError("Training IDs must be nonempty and unique")
    buckets = [[] for _ in upper_bounds]
    for row in rows:
        slot = next((i for i, bound in enumerate(upper_bounds) if row["sequence_length"] <= bound), None)
        if slot is None or row["sequence_length"] <= 0 or row["supervised_token_count"] <= 0:
            raise ValueError("Invalid/over-budget full training row; never truncate or drop it")
        buckets[slot].append(row)
    rng = random.Random(seed)
    full, leftovers = [], []
    for bucket in buckets:
        rng.shuffle(bucket)
        stop = len(bucket) // group_size * group_size
        full.extend(bucket[i:i + group_size] for i in range(0, stop, group_size))
        leftovers.extend(bucket[stop:])
    rng.shuffle(leftovers)
    stop = len(leftovers) // group_size * group_size
    full.extend(leftovers[i:i + group_size] for i in range(0, stop, group_size))
    tail = leftovers[stop:]
    rng.shuffle(full)
    groups = full + ([tail] if tail else [])
    if Counter(row["source_index"] for group in groups for row in group) != Counter(ids):
        raise AssertionError("Lost or repeated a training row")
    return [{"source_indices": [r["source_index"] for r in group],
             "sequence_lengths": [r["sequence_length"] for r in group],
             "target_counts": [r["supervised_token_count"] for r in group],
             "group_target_tokens": sum(r["supervised_token_count"] for r in group)} for group in groups]


def boundary_checks():
    # Leftovers from different buckets must combine, not silently add optimizer updates.
    rows = [{"source_index": i, "sequence_length": 3 if i < 10 else 7,
             "supervised_token_count": i % 3 + 1} for i in range(20)]
    groups = build_groups(rows, [4, 8], 8, 42)
    assert [len(g["source_indices"]) for g in groups] == [8, 8, 4]
    assert groups == build_groups(rows, [4, 8], 8, 42)
    assert sum(g["group_target_tokens"] for g in groups) == sum(r["supervised_token_count"] for r in rows)
    rejected = 0
    for invalid in (rows + [rows[0]], [{**rows[0], "sequence_length": 9}], []):
        try:
            build_groups(invalid, [4, 8], 8, 42)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Invalid schedule input was accepted")
    lr = [learning_rate_for_update(j) for j in range(1, 798)]
    assert 0 < lr[0] < lr[23] == 1e-4
    assert lr[23] > lr[24] > lr[795] > lr[796] == 0
    assert all(a <= b for a, b in zip(lr[:23], lr[1:24]))
    assert all(a >= b for a, b in zip(lr[23:-1], lr[24:]))
    return {"pooled_tail_and_exact_coverage": "pass", "deterministic_schedule": "pass",
            "invalid_inputs_rejected": rejected, "lr_boundaries_and_monotonicity": "pass"}


def prepare():
    started = time.perf_counter()
    config = read(CONFIG)
    report_path = ROOT / config["artifacts"]["preflight_report"]
    report = {"status": "started", "utc": datetime.now(timezone.utc).isoformat(),
              "real_model_loaded": False, "gpu_training_performed": False,
              "validation_examples_loaded": False, "final_examples_read": False}
    try:
        training = config["training"]
        if config["gpu_training_enabled"] or config["status"] != "cpu_preflight_only":
            raise ValueError("This entry point is a CPU plan only")
        if tuple(training[k] for k in ("epochs", "micro_batch_size", "gradient_accumulation_steps",
                                      "expected_updates", "expected_final_group_examples")) != (1, 1, 8, 796, 4):
            raise ValueError("This frozen first-round plan requires 1 epoch, batch1, accum8, 796 updates and a four-row tail")
        smoke = read(ROOT / config["model"]["inherit_loading_and_lora_from"])
        if (training["optimizer"] != smoke["optimizer"] or training["max_gradient_norm"] != 1.0
                or training["scheduler"]["warmup_updates"] != 24 or training["scheduler"]["total_updates"] != 796):
            raise ValueError("Optimizer or LR schedule no longer matches the checked first-round protocol")
        data = config["data"]
        if (data["max_sequence_length"] != 4096 or data["length_bucket_upper_bounds"] != [512, 1024, 2048, 4096]
                or any(data[k] for k in ("truncate", "drop_last", "packing"))
                or (data["expected_examples"], data["expected_databases"]) != (6364, 126)):
            raise ValueError("Data/bucketing policy changed")
        manifest = read(ROOT / data["manifest"])
        if (config["model"]["revision"] != manifest["model_revision"]
                or (config["model"]["lora_r"], config["model"]["lora_alpha"], config["model"]["lora_dropout"]) != (8, 16, 0.0)):
            raise ValueError("Frozen model/LoRA configuration changed")
        protected = dict(manifest["source_sha256"])
        merge_hashes(protected, {f"data/processed/spider_sft_tokens_v1/{name}": sha
                                for name, sha in manifest["output_sha256"].items()})
        prior = read(ROOT / "reports/qlora_target_probe_preflight.json")
        merge_hashes(protected, prior["protected_sha256"])
        run = ROOT / "outputs/runs/qlora_target_probe_v1"
        merge_hashes(protected, {(run / name).relative_to(ROOT).as_posix(): sha
                                for name, sha in read(run / "artifact_sha256.json").items()})
        cpu = read(ROOT / config["artifacts"]["cpu_equivalence_report"])
        if cpu["status"] != "pass" or cpu["device"] != "cpu" or cpu["gpu_used"]:
            raise ValueError("CPU accumulation equivalence must pass first")
        merge_hashes(protected, cpu["source_sha256"])
        for name in ("configs/sft_round1.json", "src/train/sft_round1_plan.py",
                     config["artifacts"]["cpu_equivalence_report"], data["manifest"],
                     "configs/baseline_full.json", "configs/evaluation.json",
                     "reports/baseline_full_eval.json", "reports/qlora_target_probe_result.json"):
            merge_hashes(protected, {name: digest(ROOT / name)})
        verify_hashes(protected)
        packages = {}
        for source in (manifest["packages"], prior["packages"], cpu["packages"]):
            for package, pinned in source.items():
                if package in packages and packages[package] != pinned:
                    raise ValueError(f"Conflicting frozen package versions: {package}")
                if version(package) != pinned:
                    raise ValueError(f"Frozen package changed: {package}")
                packages[package] = pinned
        stats, databases, bucket_counts = [], set(), Counter()
        with (ROOT / data["train"]).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                batch = collate_sft_batch([row], manifest["pad_token_id"], data["max_sequence_length"])
                length = batch["input_ids"].shape[1]
                count = int((batch["labels"][:, 1:] != -100).sum())
                if length != row["sequence_length"] or count != row["supervised_token_count"]:
                    raise ValueError("Unexpected collated length or next-token targets")
                if int(batch["input_ids"].max()) >= 151936:
                    raise ValueError("Token outside frozen vocabulary")
                stats.append({k: row[k] for k in ("source_index", "sequence_length", "supervised_token_count")})
                databases.add(row["db_id"])
                bucket_counts[next(b for b in data["length_bucket_upper_bounds"] if length <= b)] += 1
        if len(stats) != 6364 or len(databases) != 126:
            raise ValueError("Full training membership/count changed")
        validation = read(ROOT / config["validation"]["generation_config"])
        if databases.intersection(validation["expected_database_ids"]):
            raise ValueError("Training database overlaps internal validation")
        groups = build_groups(stats, data["length_bucket_upper_bounds"], 8, config["seed"])
        if len(groups) != 796 or [len(g["source_indices"]) for g in groups] != [8] * 795 + [4]:
            raise ValueError("Expected 795 full groups plus a four-example tail")
        if groups != build_groups(stats, data["length_bucket_upper_bounds"], 8, config["seed"]):
            raise AssertionError("Schedule is not deterministic")
        for index, group in enumerate(groups):
            group.update(update=index + 1, epoch_index=0, learning_rate=learning_rate_for_update(index + 1))
        schedule = {"version": config["version"], "epoch_index": 0, "seed": config["seed"],
                    "train_sha256": protected[data["train"]], "groups": groups}
        write(ROOT / config["artifacts"]["schedule"], schedule, immutable=True)
        protected[config["artifacts"]["schedule"]] = digest(ROOT / config["artifacts"]["schedule"])
        resume_checks = []
        for cursor in (0, 1, 200, 400, 600, 796):
            assert schedule["groups"][:cursor] + schedule["groups"][cursor:] == groups
            resume_checks.append({"completed_updates": cursor, "next_group_index": cursor,
                                  "next_update_lr": learning_rate_for_update(cursor + 1)})
        sequence_total = sum(r["sequence_length"] for r in stats)
        target_total = sum(g["group_target_tokens"] for g in groups)
        if sequence_total != 3932044 or target_total != 206073:
            raise ValueError("Full input or supervision token totals changed")
        assumptions = config["resources"]["per_bucket_forward_backward_seconds_assumptions"]
        estimated = [sum(bucket_counts[b] * assumptions[i][side] for i, b in enumerate(data["length_bucket_upper_bounds"])) / 3600
                     for side in (0, 1)]
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError("CPU preflight unexpectedly initialized CUDA")
        verify_hashes(protected)
        report.update(status="pass", config=config, checks=boundary_checks(), cpu_equivalence_status=cpu["status"],
            data={"examples": len(stats), "databases": len(databases), "input_tokens": sequence_total,
                  "target_tokens": target_total, "length_buckets": dict(bucket_counts),
                  "maximum_sequence_length": max(r["sequence_length"] for r in stats),
                  "maximum_target_tokens": max(r["supervised_token_count"] for r in stats),
                  "all_rows_collated_on_cpu": True, "truncated": 0, "dropped": 0, "all_ids_exactly_once": True},
            updates=len(groups), full_groups=795, final_group_size=4,
            schedule_sha256=protected[config["artifacts"]["schedule"]], resume_cursor_checks=resume_checks,
            learning_rate_boundaries={str(j): learning_rate_for_update(j) for j in (1, 24, 25, 796, 797)},
            forward_backward_scenario_hours=estimated, free_disk_gib=shutil.disk_usage(ROOT).free / 2**30,
            protected_sha256=protected, packages=packages, cuda_initialized=False, real_gpu_resume_validated=False)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write(report_path, report)
    print(json.dumps({k: report[k] for k in ("status", "data", "updates", "final_group_size", "seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    prepare()
