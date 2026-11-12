"""Audit only the frozen train/validation split; no model and no final data reads.

Usage: python check_gold_sql.py --pilot 100, then python check_gold_sql.py
Outputs are per-run snapshots. Original split files and databases are never edited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("gold_sql_worker.py")
CONFIG = ROOT / "configs/gold_check.json"
SPLIT_DIR = ROOT / "data/processed/spider_v1"
BASE_PYTHON = Path(sys.base_prefix) / "python.exe"
WORKER_PYTHON = str(BASE_PYTHON) if os.name == "nt" and BASE_PYTHON.is_file() else sys.executable
TIMEOUTS = {"query_timeout", "parent_timeout"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def load_inputs():
    """Hash only selected train/validation files and their 140 databases."""
    manifest_path = SPLIT_DIR / "split_manifest.json"
    reproducibility_path = ROOT / "reports/spider_split_reproducibility.json"
    manifest = read_json(manifest_path)
    expected = read_json(reproducibility_path)["outputs_sha256"]
    if sha256(manifest_path) != expected[manifest_path.name]:
        raise ValueError("Frozen split manifest changed")
    download_path = ROOT / manifest["download_provenance"]["manifest_path"]
    if sha256(download_path) != manifest["download_provenance"]["manifest_sha256"]:
        raise ValueError("Download manifest changed")
    download = {item["path"]: item for item in read_json(download_path)["files"]}
    rows = []
    hashes = {relative(p): sha256(p) for p in (manifest_path, reproducibility_path, download_path)}
    for split, count in (("train", 6375), ("validation", 625)):
        path = SPLIT_DIR / f"{split}.jsonl"
        digest = sha256(path)
        if digest != expected[path.name]:
            raise ValueError(f"Frozen {split} file changed")
        hashes[relative(path)] = digest
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(records) != count:
            raise ValueError(f"Wrong {split} count")
        ids = set(manifest["splits"][split]["db_ids"])
        for row in records:
            if row["project_split"] != split or row["db_id"] not in ids:
                raise ValueError("Split membership mismatch")
        if {r["db_id"] for r in records} != ids:
            raise ValueError("Database membership incomplete")
        rows.extend(records)
    if sorted(row["source_index"] for row in rows) != list(range(7000)):
        raise ValueError("Expected each original training source_index exactly once")
    train_ids = {r["db_id"] for r in rows if r["project_split"] == "train"}
    val_ids = {r["db_id"] for r in rows if r["project_split"] == "validation"}
    final_ids = set(manifest["splits"]["final_eval"]["db_ids"])
    if train_ids & val_ids or (train_ids | val_ids) & final_ids:
        raise ValueError("Database split overlap")
    for db_id in sorted(train_ids | val_ids):
        if Path(db_id).name != db_id or db_id in (".", ".."):
            raise ValueError("Invalid db_id")
        path = ROOT / "data/raw/spider/database" / db_id / f"{db_id}.sqlite"
        digest = sha256(path)
        source_key = relative(path)
        download_key = f"database/{db_id}/{db_id}.sqlite"
        if (digest != manifest["source_sha256"][source_key]
                or digest != download[download_key]["sha256"]
                or path.stat().st_size != download[download_key]["size"]):
            raise ValueError(f"Database differs from download/split snapshot: {db_id}")
        hashes[source_key] = digest
    return sorted(rows, key=lambda row: row["source_index"]), hashes


class WorkerClient:
    """Persistent real Python process, with a hard deadline on each response."""
    def __init__(self, worker_path, stderr_path, grace):
        self.worker_path = worker_path
        self.stderr = Path(stderr_path).open("a", encoding="utf-8")
        self.grace = grace
        self.process = None
        self.responses = None
        self.reader = None

    def start(self):
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            [WORKER_PYTHON, "-X", "utf8", "-u", str(self.worker_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding="utf-8", errors="strict", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        process, responses = self.process, self.responses

        def read_responses():
            try:
                for line in process.stdout:
                    responses.put(line)
            finally:
                responses.put(None)

        self.reader = threading.Thread(target=read_responses, daemon=True)
        self.reader.start()

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            self.reader.join(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None

    def close(self):
        self.stop()
        self.stderr.close()

    def call(self, request):
        started = time.perf_counter()
        try:
            if self.process is None:
                self.start()
            self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
            line = self.responses.get(timeout=request["timeout_seconds"] + self.grace)
            if line is None:
                raise RuntimeError("SQL worker exited without a response")
            result = json.loads(line)
            if not isinstance(result, dict) or result.get("request_id") != request["request_id"]:
                raise RuntimeError("Worker response ID mismatch")
            if not isinstance(result.get("status"), str) or result["status"] not in {
                    "ok", "sql_error", "query_timeout", "decode_error",
                    "query_rejected", "database_error", "worker_error"}:
                raise RuntimeError("Worker response has no valid status")
        except queue.Empty:
            result = {"request_id": request["request_id"], "status": "parent_timeout",
                      "error": "Hard process deadline exceeded", "row_count_complete": False}
            self.stop()
        except (OSError, ValueError, RuntimeError) as exc:
            result = {"request_id": request["request_id"], "status": "worker_error",
                      "error": str(exc), "row_count_complete": False}
            self.stop()
        result["timeout_seconds"] = request["timeout_seconds"]
        result["decode_mode"] = request["decode_mode"]
        result["parent_wall_seconds"] = round(time.perf_counter() - started, 6)
        return result


def audit_one(client, row, config):
    request = {"request_id": row["source_index"], "sql": row["query"],
               "db_path": str(ROOT / "data/raw/spider/database" / row["db_id"] / f"{row['db_id']}.sqlite"),
               "timeout_seconds": config["initial_timeout_seconds"], "decode_mode": "strict"}
    attempts = [client.call(request)]
    if attempts[-1]["status"] in TIMEOUTS:
        request["timeout_seconds"] = config["retry_timeout_seconds"]
        attempts.append(client.call(request))
    primary = attempts[-1]
    if primary["status"] == "decode_error":
        # This never changes the strict primary status or admits the row to training.
        request["decode_mode"] = "replace"
        request["timeout_seconds"] = config["retry_timeout_seconds"]
        attempts.append(client.call(request))
    return {"source_index": row["source_index"], "source_file": row["source_file"],
            "project_split": row["project_split"], "db_id": row["db_id"],
            "question": row["question"], "query": row["query"],
            "status": primary["status"], "row_count": primary.get("row_count"),
            "row_count_complete": primary.get("row_count_complete", False),
            "empty_result": primary.get("empty_result"), "attempts": attempts}


def derive_training(rows, checks):
    """Exact dedup and unresolved-error quarantine are separate, traceable steps."""
    lookup = {item["source_index"]: item for item in checks}
    keep, quarantine, duplicates = [], [], []
    representatives = {}
    for row in sorted(rows, key=lambda item: item["source_index"]):
        if row["project_split"] != "train":
            continue
        key = (row["db_id"], row["question"], row["query"])
        audit = lookup[row["source_index"]]
        if key in representatives:
            duplicates.append({**row, "duplicate_of_source_index": representatives[key],
                               "gold_status": audit["status"], "exclusion_reason": "exact_duplicate"})
            continue
        representatives[key] = row["source_index"]
        if audit["status"] == "ok" and audit["row_count_complete"]:
            keep.append(row)  # Original row, no rewritten SQL or token fields.
        else:
            quarantine.append({**row, "gold_status": audit["status"],
                               "exclusion_reason": "unresolved_gold_execution", "attempts": audit["attempts"]})
    return keep, quarantine, duplicates


def question_review(rows):
    groups = defaultdict(list)
    for row in rows:
        normalized = " ".join(unicodedata.normalize("NFKC", row["question"]).casefold().split())
        groups[(row["db_id"], normalized)].append(row)
    return [{"db_id": key[0], "normalized_question": key[1],
             "note": "Review only; different SQL text does not by itself prove conflicting meaning",
             "records": [{k: row[k] for k in ("source_index", "project_split", "question", "query")} for row in group]}
            for key, group in sorted(groups.items()) if len({r["query"] for r in group}) > 1]


def run(pilot):
    started = time.perf_counter()
    config = read_json(CONFIG)
    rows, before = load_inputs()
    selected = rows if not pilot else [rows[i * len(rows) // pilot] for i in range(pilot)]
    mode = "pilot" if pilot else "full"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + mode
    run_dir = ROOT / "outputs/gold_checks" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), WORKER, CONFIG):
        shutil.copyfile(path, run_dir / path.name)
    code_hashes = {relative(path): sha256(path) for path in (Path(__file__), WORKER, CONFIG)}
    snapshot_worker = run_dir / WORKER.name
    write_json(run_dir / "inputs.json", {"input_sha256": before, "code_sha256": code_hashes,
               "selected_source_indices": [r["source_index"] for r in selected],
               "final_data_read": False, "python": sys.version, "sqlite": sqlite3.sqlite_version,
               "worker_python": WORKER_PYTHON, "config": config})
    print(f"Run: {relative(run_dir)}; selected {len(selected)} train/validation records", flush=True)
    lock = threading.Lock()
    completed = 0
    counts = Counter()
    checks = []
    with (run_dir / "gold_checks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        def lane(number):
            nonlocal completed
            client = WorkerClient(snapshot_worker, run_dir / f"worker_{number}.stderr.log", config["hard_timeout_grace_seconds"])
            try:
                for row in selected[number::config["workers"]]:
                    result = audit_one(client, row, config)
                    with lock:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        checks.append(result)
                        completed += 1
                        counts[result["status"]] += 1
                        if completed % 250 == 0 or completed == len(selected):
                            print(f"{completed}/{len(selected)} {dict(counts)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            finally:
                client.close()
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = [executor.submit(lane, n) for n in range(config["workers"])]
            for future in futures:
                future.result()
    if sorted(c["source_index"] for c in checks) != sorted(r["source_index"] for r in selected):
        raise ValueError("Missing or duplicate audit records")
    # Verify after all worker processes have closed, including code/config identity.
    for name, digest in {**before, **code_hashes}.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Input or code changed during audit: {name}")
    checks.sort(key=lambda item: item["source_index"])
    write_jsonl(run_dir / "issues.jsonl", [c for c in checks if c["status"] != "ok"])
    summary = {"run_id": run_id, "mode": mode, "record_count": len(checks),
               "status_counts": dict(counts), "inputs_unchanged": True,
               "selected_databases_match_download": True, "final_data_read": False,
               "gpu_used": False, "sql_modified": False,
               "timeout_recovered": sum(c["attempts"][0]["status"] in TIMEOUTS and c["status"] == "ok" for c in checks),
               "decode_replacement_diagnostic_ok": sum(c["status"] == "decode_error" and c["attempts"][-1]["status"] == "ok" for c in checks),
               "empty_successful_results": sum(c["empty_result"] is True for c in checks),
               "split_stats": {s: {"count": len(part), "status_counts": dict(Counter(c["status"] for c in part)),
                                     "empty_successful_results": sum(c["empty_result"] is True for c in part)}
                               for s in ("train", "validation") if (part := [c for c in checks if c["project_split"] == s])},
               "audit_path": relative(run_dir / "gold_checks.jsonl"), "issues_path": relative(run_dir / "issues.jsonl"),
               "semantic_correctness_verified": False, "validation_filtering_allowed": False}
    if not pilot:
        keep, quarantine, duplicates = derive_training(rows, checks)
        derived = run_dir / "derived"
        derived.mkdir()
        write_jsonl(derived / "train_clean.jsonl", keep)
        write_jsonl(derived / "train_quarantine.jsonl", quarantine)
        write_jsonl(derived / "train_duplicates.jsonl", duplicates)
        review = question_review(rows)
        write_json(derived / "same_question_review.json", review)
        if len(keep) + len(quarantine) + len(duplicates) != 6375:
            raise ValueError("Training accounting does not reconcile")
        summary["training_disposition"] = {"original": 6375, "clean": len(keep), "quarantine": len(quarantine), "duplicate_redundant": len(duplicates)}
        summary["same_db_same_question_different_sql_groups"] = len(review)
        summary["derived_paths"] = {path.name: relative(path) for path in sorted(derived.iterdir())}
        summary["derived_sha256"] = {path.name: sha256(path) for path in sorted(derived.iterdir())}
        summary["validation_path_unchanged"] = "data/processed/spider_v1/validation.jsonl"
        summary["policy"] = config
    summary["wall_seconds"] = round(time.perf_counter() - started, 3)
    write_json(run_dir / "summary.json", summary)
    # A small latest-report pointer; every full run and its derived data remain separate.
    write_json(ROOT / "reports" / ("gold_sql_pilot.json" if pilot else "gold_sql_check.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, default=0, help="Evenly spaced sample count; 0 checks all 7000")
    args = parser.parse_args()
    if args.pilot < 0 or args.pilot > 7000:
        parser.error("--pilot must be between 0 and 7000")
    run(args.pilot)
