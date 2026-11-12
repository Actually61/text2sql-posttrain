"""Project validation EX using the pinned official comparator and read-only SQL.

check: verify source/dependencies/data without SQL execution.
controls: gold-as-prediction and deterministic same-database shuffled controls.
score: score a complete JSONL prediction file, aligned by source_index and db_id.
This module never reads final/dev/test examples or launches a language model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import queue
import random
import shutil
import subprocess
import sys
import sysconfig
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation.json"
WORKER = Path(__file__).with_name("spider_pair_worker.py")
BASE_PYTHON = Path(sys.base_prefix) / "python.exe"
WORKER_PYTHON = str(BASE_PYTHON) if os.name == "nt" and BASE_PYTHON.is_file() else sys.executable
PURELIB = sysconfig.get_paths()["purelib"]
BOOTSTRAP = "import site,runpy,sys;site.addsitedir(sys.argv[1]);runpy.run_path(sys.argv[2],run_name='__main__')"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(values):
    return "".join(json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for v in values).encode("utf-8")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path must stay within the project")
    return path


def expect_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Frozen file changed: {path}")
    return actual


def load_validation():
    config = read_json(CONFIG)
    if config["split"] != "validation" or config["plug_value"] or not config["keep_distinct"] or config["rewrite_sql"]:
        raise ValueError("This protocol requires validation, predicted values, DISTINCT and unchanged SQL")
    provenance_path = project_path(config["source_manifest"])
    provenance = read_json(provenance_path)
    if provenance["commit"] != config["upstream_commit"]:
        raise ValueError("Upstream commit mismatch")
    paths = [CONFIG, provenance_path, Path(__file__), WORKER, ROOT / "requirements.lock.txt"]
    for name, metadata in provenance["source_files"].items():
        path = project_path(provenance["local_dir"]) / name
        expect_hash(path, metadata["sha256"])
        paths.append(path)
    for name, expected in provenance["dependency_versions"].items():
        if version(name) != expected:
            raise ValueError(f"Evaluation dependency changed: {name}")
    inputs_manifest_path = project_path(config["inputs_manifest"])
    verified = read_json(ROOT / "reports/schema_inputs_verification.json")
    expect_hash(inputs_manifest_path, verified["manifest_sha256"])
    inputs = read_json(inputs_manifest_path)
    validation_path = project_path(config["validation_file"])
    expect_hash(validation_path, inputs["input_sha256"][relative(validation_path)])
    references_path = project_path(config["references_file"])
    expect_hash(references_path, inputs["output_sha256"][references_path.name])
    split_path = ROOT / "data/processed/spider_v1/split_manifest.json"
    expect_hash(split_path, inputs["input_sha256"][relative(split_path)])
    split = read_json(split_path)
    rows, references = read_jsonl(validation_path), read_jsonl(references_path)
    if len(rows) != 625 or len(references) != 625 or config["expected_examples"] != 625:
        raise ValueError("Expected the complete 625-example internal validation split")
    ref_lookup = {r["source_index"]: r for r in references}
    if len(ref_lookup) != len(rows) or len({r["source_index"] for r in rows}) != len(rows):
        raise ValueError("Repeated reference/source IDs")
    allowed = set(split["splits"]["validation"]["db_ids"])
    if {r["db_id"] for r in rows} != allowed or allowed & set(split["splits"]["final_eval"]["db_ids"]):
        raise ValueError("Validation database membership mismatch")
    for row in rows:
        ref = ref_lookup.get(row["source_index"])
        if row["project_split"] != "validation" or ref is None or ref["db_id"] != row["db_id"] or ref["gold_sql"] != row["query"]:
            raise ValueError("Reference/source association mismatch")
    for db_id in sorted(allowed):
        if Path(db_id).name != db_id or db_id in (".", ".."):
            raise ValueError("Invalid database ID")
        path = project_path(config["database_dir"]) / db_id / f"{db_id}.sqlite"
        expect_hash(path, split["source_sha256"][relative(path)])
        paths.append(path)
    paths.extend([inputs_manifest_path, validation_path, references_path, split_path,
                  ROOT / "reports/schema_inputs_verification.json"])
    fingerprints = {relative(p): digest(p) for p in paths}
    return config, sorted(rows, key=lambda r: r["source_index"]), fingerprints


def label_hardness(rows):
    # Use existing gold ASTs; do not parse questions, predicted SQL or final data.
    vendor = ROOT / "third_party/test-suite-sql-eval"
    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("pinned_spider_evaluation", vendor / "evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = module.Evaluator()
    return {row["source_index"]: evaluator.eval_hardness(row["sql"]) for row in rows}


def validate_predictions(predictions, rows):
    lookup = {}
    expected = {r["source_index"]: r["db_id"] for r in rows}
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("Prediction must be a JSON object")
        index = prediction.get("source_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in expected or index in lookup:
            raise ValueError("Prediction IDs missing, repeated or outside validation")
        if prediction.get("db_id") != expected[index] or not isinstance(prediction.get("predicted_sql"), str):
            raise ValueError("Prediction database or SQL field is invalid")
        lookup[index] = prediction
    if set(lookup) != set(expected):
        raise ValueError("Predictions must cover every validation source_index exactly once")
    return lookup


def make_control_predictions(rows, seed):
    gold = [{"source_index": r["source_index"], "db_id": r["db_id"], "predicted_sql": r["query"]} for r in rows]
    by_db = defaultdict(list)
    for row in rows:
        by_db[row["db_id"]].append(row)
    rng = random.Random(seed)
    shuffled = []
    for db_id, group in sorted(by_db.items()):
        group = sorted(group, key=lambda r: r["source_index"])
        if len(group) < 2:
            raise ValueError("Cannot derange a database with fewer than two questions")
        rng.shuffle(group)
        for i, row in enumerate(group):
            donor = group[(i+1) % len(group)]
            shuffled.append({"source_index": row["source_index"], "db_id": db_id,
                             "predicted_sql": donor["query"], "donor_source_index": donor["source_index"]})
    return gold, sorted(shuffled, key=lambda r: r["source_index"])


class PairClient:
    def __init__(self, worker_path, stderr_path, grace):
        self.worker_path = Path(worker_path)
        self.stderr = Path(stderr_path).open("a", encoding="utf-8")
        self.grace = grace
        self.process = None

    def start(self):
        self.responses = queue.Queue()
        env = os.environ.copy()
        env["TEXT2SQL_PROJECT_ROOT"] = str(ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.process = subprocess.Popen([WORKER_PYTHON, "-X", "utf8", "-u", "-c", BOOTSTRAP, PURELIB, str(self.worker_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True,
            encoding="utf-8", errors="strict", bufsize=1, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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
        hard_seconds = 2*request["query_timeout_seconds"] + request["comparison_timeout_seconds"] + self.grace
        try:
            if self.process is None:
                self.start()
            self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            self.process.stdin.flush()
            line = self.responses.get(timeout=hard_seconds)
            if line is None:
                raise RuntimeError("Worker exited without response")
            result = json.loads(line)
            if not isinstance(result, dict) or result.get("request_id") != request["request_id"]:
                raise ValueError("Malformed worker response or mismatched request ID")
            status = result.get("status")
            if not isinstance(status, str) or status not in {"scored", "gold_error", "prediction_error", "both_error", "comparison_error", "worker_error"}:
                raise ValueError("Unexpected worker status")
            if status == "scored" and (type(result.get("matched")) is not bool or result.get("prediction_executable") is not True):
                raise ValueError("Incomplete scored response")
        except queue.Empty:
            self.stop()
            result = {"request_id": request["request_id"], "status": "parent_timeout", "matched": None,
                      "prediction_executable": None, "gold": None, "prediction": None, "error": "Hard process deadline exceeded"}
        except (OSError, ValueError, RuntimeError) as exc:
            self.stop()
            result = {"request_id": request["request_id"], "status": "worker_error", "matched": None,
                      "prediction_executable": None, "gold": None, "prediction": None, "error": str(exc)}
        result["parent_wall_seconds"] = round(time.perf_counter()-started, 6)
        result["hard_timeout_seconds"] = hard_seconds
        return result


def aggregate(results):
    count = len(results)
    correct = sum(r["matched"] is True for r in results)
    valid = sum(r["prediction_executable"] is True for r in results)
    unknown = sum(r["prediction_executable"] is None for r in results)
    return {"count": count, "execution_correct": correct,
            "execution_accuracy": correct/count if count else None,
            "executable_predictions": valid, "prediction_executable_rate": valid/count if count else None,
            "prediction_execution_unknown": unknown,
            "status_counts": dict(Counter(r["status"] for r in results)),
            "gold_execution_errors": sum(isinstance(r.get("gold"), dict) and r["gold"]["status"] != "ok" for r in results),
            "both_results_empty": sum(isinstance(r.get("gold"), dict) and isinstance(r.get("prediction"), dict)
                and r["gold"].get("empty_result") is True and r["prediction"].get("empty_result") is True for r in results)}


def run_evaluation(predictions_path, label, config, rows, fingerprints):
    started = time.perf_counter()
    predictions = validate_predictions(read_jsonl(predictions_path), rows)
    prediction_digest = digest(predictions_path)
    hardness = label_hardness(rows)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + label
    run_dir = ROOT / "outputs/spider_eval" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), WORKER, CONFIG):
        shutil.copyfile(path, run_dir / path.name)
    (run_dir / "predictions.jsonl").write_bytes(Path(predictions_path).read_bytes())
    (run_dir / "inputs.json").write_bytes(json_bytes({"input_sha256": fingerprints, "prediction_sha256": prediction_digest,
        "prediction_path": str(predictions_path), "protocol": config, "source_index_order": [r["source_index"] for r in rows],
        "worker_python": WORKER_PYTHON, "worker_site_packages": PURELIB, "final_examples_read": False}))
    results = []
    lock = threading.Lock()
    with (run_dir / "per_example.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        def lane(number):
            client = PairClient(run_dir / WORKER.name, run_dir / f"worker_{number}.stderr.log", config["hard_timeout_grace_seconds"])
            try:
                for row in rows[number::config["workers"]]:
                    prediction = predictions[row["source_index"]]
                    request = {"request_id": row["source_index"],
                        "db_path": str(project_path(config["database_dir"]) / row["db_id"] / f"{row['db_id']}.sqlite"),
                        "gold_sql": row["query"], "predicted_sql": prediction["predicted_sql"],
                        "query_timeout_seconds": config["query_timeout_seconds"],
                        "comparison_timeout_seconds": config["comparison_timeout_seconds"], "seed": config["seed"]}
                    result = client.call(request)
                    result.update(source_index=row["source_index"], db_id=row["db_id"], question=row["question"],
                        gold_sql=row["query"], predicted_sql=prediction["predicted_sql"], hardness=hardness[row["source_index"]])
                    if "donor_source_index" in prediction:
                        result["donor_source_index"] = prediction["donor_source_index"]
                    with lock:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        results.append(result)
                        if len(results) % 100 == 0 or len(results) == len(rows):
                            print(f"{label}: {len(results)}/{len(rows)}", flush=True)
            finally:
                client.close()
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = [executor.submit(lane, i) for i in range(config["workers"])]
            for future in futures:
                future.result()
    if sorted(r["source_index"] for r in results) != [r["source_index"] for r in rows]:
        raise ValueError("Incomplete or repeated scoring records")
    for filename, expected in fingerprints.items():
        expect_hash(ROOT / filename, expected)
    expect_hash(predictions_path, prediction_digest)
    summary = {"run_id": run_id, "label": label, "metric": config["metric"], "protocol_version": config["version"],
        "overall": aggregate(results), "by_hardness": {h: aggregate([r for r in results if r["hardness"] == h]) for h in ("easy", "medium", "hard", "extra")},
        "per_example_path": relative(run_dir / "per_example.jsonl"), "run_dir": relative(run_dir),
        "wall_seconds": round(time.perf_counter()-started, 3), "source_files_unchanged": True,
        "final_examples_read": False, "model_generation_performed": False,
        "complete_test_suite_accuracy": False, "upstream_commit": config["upstream_commit"],
        "metric_limitations": ["single original database can allow accidental matches", "official comparator permits column permutations",
                               "two empty result sets match even with different result column counts",
                               "ordering follows substring 'order by' in gold.lower(); no custom float tolerance"]}
    (run_dir / "summary.json").write_bytes(json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def publish_controls(rows, config):
    gold, shuffled = make_control_predictions(rows, config["seed"])
    directory = ROOT / "data/processed/spider_eval_controls_v1"
    payloads = {"gold_as_prediction.jsonl": jsonl_bytes(gold), "shuffled_same_db.jsonl": jsonl_bytes(shuffled)}
    payloads["manifest.json"] = json_bytes({"purpose": "evaluator controls, not model predictions or training input", "seed": config["seed"],
        "count": len(rows), "shuffle_algorithm": "sorted db_ids; sorted rows; Random(42).shuffle each group; cyclic next donor",
        "no_source_index_fixed_points": all(r["source_index"] != r["donor_source_index"] for r in shuffled),
        "source_validation_sha256": digest(project_path(config["validation_file"])),
        "output_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}, "final_examples_read": False})
    for name, value in payloads.items():
        path = directory / name
        if path.exists() and path.read_bytes() != value:
            raise ValueError("Existing control data changed; refusing overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        path = directory / name
        if not path.exists():
            path.write_bytes(value)
    return directory


def main(stage, predictions_path=None):
    config, rows, fingerprints = load_validation()
    if stage == "check":
        report = {"status": "ready", "upstream_commit": config["upstream_commit"], "validation_examples": len(rows),
                  "databases": len({r["db_id"] for r in rows}), "hardness_counts": dict(Counter(label_hardness(rows).values())),
                  "input_sha256": fingerprints, "final_examples_read": False, "sql_executed": False}
        (ROOT / "reports/spider_evaluator_check.json").write_bytes(json_bytes(report))
        print(json.dumps({k:v for k,v in report.items() if k != "input_sha256"}, ensure_ascii=False, indent=2))
    elif stage == "controls":
        directory = publish_controls(rows, config)
        gold = run_evaluation(directory / "gold_as_prediction.jsonl", "gold_control", config, rows, fingerprints)
        shuffled = run_evaluation(directory / "shuffled_same_db.jsonl", "shuffled_control", config, rows, fingerprints)
        passed = gold["overall"]["execution_correct"] == 625 and gold["overall"]["executable_predictions"] == 625
        (ROOT / "reports/spider_eval_controls.json").write_bytes(json_bytes({"gold_control_passed": passed,
            "gold_as_prediction": gold, "same_database_shuffled": shuffled,
            "shuffled_score_is_observed_not_required_to_be_zero": True, "these_are_not_model_scores": True}))
        if not passed:
            raise RuntimeError("Gold control must score 625/625; inspect per-example errors before baseline")
    else:
        if not predictions_path:
            raise ValueError("--predictions is required for score")
        path = Path(predictions_path)
        path = path.resolve() if path.is_absolute() else project_path(predictions_path)
        summary = run_evaluation(path, "predictions", config, rows, fingerprints)
        (ROOT / "reports/spider_eval_latest.json").write_bytes(json_bytes(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "controls", "score"), required=True)
    parser.add_argument("--predictions", help="JSONL: source_index, db_id, predicted_sql for every validation example")
    args = parser.parse_args()
    main(args.stage, args.predictions)
