"""Score the one-shot 796-update SFT predictions with the frozen Spider EX rule.

Only the dispatcher differs from the baseline: infrastructure failures stop new
requests instead of restarting a worker. SQL/model errors remain scored errors.
No model imports, training, generation, final examples or SQL rewrites occur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
import traceback

import evaluate_spider as evaluation

ROOT = evaluation.ROOT
GENERATION_REPORT = ROOT / "reports/sft_round1_eval_v2_generation.json"
REPORT = ROOT / "reports/sft_round1_eval_v2_eval.json"
CLAIM = REPORT.with_suffix(".claim.json")
RUN_DIR = ROOT / "outputs/spider_eval/sft_round1_eval_v2"
LABEL = "sft_round1_eval_v2"
SCORING_BUDGET_SECONDS = 600.0


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(evaluation.json_bytes(value))
        stream.flush()


def infrastructure_problem(result):
    """Preserve bounded SQL outcomes, but do not bless a broken evaluator run."""
    if result.get("status") in {"worker_error", "parent_timeout"}:
        return result.get("error", result["status"])
    if result.get("status") in {"gold_error", "both_error"}:
        return f"Frozen gold SQL could not execute: {result.get('gold')}"
    for name in ("gold", "prediction"):
        metadata = result.get(name)
        if isinstance(metadata, dict) and metadata.get("status") in {"database_error", "worker_error"}:
            return f"{name}: {metadata.get('status')}: {metadata.get('error', '')}"
    # A returned comparator budget overrun is an existing frozen scoring
    # outcome. Unexpected comparator exceptions signal a broken run instead.
    if result.get("status") == "comparison_error" and result.get("error") != "Comparator returned after its time budget":
        return f"Unexpected comparator error: {result.get('error')}"
    return None


def check_generation(generation, predictions, expected_ids):
    for field in ("prediction_count", "generated_count", "attempted_count"):
        if generation.get(field) != 625:
            raise ValueError(f"Generation is incomplete: {field}")
    for field in ("complete_validation_records", "complete_validation_generated", "project_fine_tuning_applied"):
        if generation.get(field) is not True:
            raise ValueError(f"Generation did not attest {field}")
    for field in ("final_examples_read", "prediction_selection_performed", "reference_answers_loaded_for_inference", "validation_reference_files_read"):
        if generation.get(field) is not False:
            raise ValueError(f"Unexpected generation scope: {field}")
    if generation.get("generation_failure_or_skip_count") != 0 or len(predictions) != 625:
        raise ValueError("A failed or incomplete generator run must not be scored")
    if [row.get("source_index") for row in predictions] != expected_ids:
        raise ValueError("Prediction order differs from frozen validation order")
    for number, row in enumerate(predictions, 1):
        if (row.get("ordinal") != number or row.get("attempt_number") != 1
                or row.get("generation_ok") is not True or row.get("generation_status") != "generated"):
            raise ValueError("Every example must be freshly generated exactly once")


def dispatch(run_dir, config, rows, predictions, hardness, *, client_class=evaluation.PairClient):
    """Four original lanes; no new request after an observed fatal condition.

    At most three other already-dispatched calls can finish after the first
    failure. Their original 9-second hard deadline remains in force. A lane is
    never restarted and no source_index is replayed.
    """
    started = time.perf_counter()
    deadline = started + SCORING_BUDGET_SECONDS
    stop = threading.Event()
    lock = threading.Lock()
    results, failures, starts = [], [], []
    with (run_dir / "per_example.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        def fail(message, **context):
            with lock:
                failures.append({"error": message, **context})
                stop.set()

        def lane(number):
            client = None
            try:
                client = client_class(run_dir / evaluation.WORKER.name,
                    run_dir / f"worker_{number}.stderr.log", config["hard_timeout_grace_seconds"])
                for row in rows[number::config["workers"]]:
                    # Reserving dispatch under the same lock orders each start
                    # against the first observed fatal response.
                    with lock:
                        if stop.is_set():
                            break
                        if time.perf_counter() >= deadline:
                            failures.append({"error": "Scoring global budget exhausted", "lane": number})
                            stop.set()
                            break
                        starts.append(row["source_index"])
                    prediction = predictions[row["source_index"]]
                    request = {"request_id": row["source_index"],
                        "db_path": str(evaluation.project_path(config["database_dir"]) / row["db_id"] / f"{row['db_id']}.sqlite"),
                        "gold_sql": row["query"], "predicted_sql": prediction["predicted_sql"],
                        "query_timeout_seconds": config["query_timeout_seconds"],
                        "comparison_timeout_seconds": config["comparison_timeout_seconds"], "seed": config["seed"]}
                    result = client.call(request)
                    result.update(source_index=row["source_index"], db_id=row["db_id"], question=row["question"],
                        gold_sql=row["query"], predicted_sql=prediction["predicted_sql"], hardness=hardness[row["source_index"]])
                    problem = infrastructure_problem(result)
                    with lock:
                        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        results.append(result)
                        if problem is not None:
                            failures.append({"error": problem, "source_index": row["source_index"], "lane": number})
                            stop.set()
                        if len(results) % 100 == 0 or len(results) == len(rows):
                            print(f"{LABEL}: {len(results)}/{len(rows)}", flush=True)
                    if problem is not None:
                        break
            except BaseException as error:
                fail(f"{type(error).__name__}: {error}", lane=number, traceback=traceback.format_exc())
            finally:
                if client is not None:
                    try:
                        client.close()
                    except BaseException as error:
                        fail(f"Cleanup {type(error).__name__}: {error}", lane=number, traceback=traceback.format_exc())
        with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
            futures = [executor.submit(lane, number) for number in range(config["workers"])]
            for future in futures:
                future.result()
    return results, {"failures": failures, "attempted_source_indices": starts,
        "attempted_count": len(starts), "completed_count": len(results), "automatic_retries": 0,
        "workers": config["workers"], "global_budget_seconds": SCORING_BUDGET_SECONDS,
        "dispatch_seconds": time.perf_counter() - started,
        "stop_policy": "Stop new dispatch on infrastructure failure; retain bounded in-flight outcomes; no restart or replay"}


def main(predictions_path):
    started = time.perf_counter()
    if REPORT.exists() or RUN_DIR.exists():
        raise FileExistsError("This one-shot scoring run already exists; do not overwrite or retry")
    write_new(CLAIM, {"label": LABEL, "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_retries": 0, "final_examples_read": False})
    summary = {"status": "failed", "label": LABEL, "final_examples_read": False,
        "training_performed": False, "model_generation_performed": False, "complete_validation_scored": False}
    try:
        generation = evaluation.read_json(GENERATION_REPORT)
        controller_path = evaluation.project_path(generation["run_dir"]) / "controller_result.json"
        controller = evaluation.read_json(controller_path)
        if (controller.get("status") != "pass" or controller.get("exit_code") != 0
                or controller.get("job_active_pids_after_cleanup") != []):
            raise ValueError("Generation controller has not confirmed successful exit and complete cleanup")
        predictions_path = evaluation.project_path(predictions_path)
        if predictions_path != evaluation.project_path(generation["prediction_path"]):
            raise ValueError("Predictions must be the unchanged file named in this SFT generation report")
        evaluation.expect_hash(predictions_path, generation["prediction_sha256"])
        config, rows, fingerprints = evaluation.load_validation()
        baseline = evaluation.read_json(ROOT / "reports/baseline_full_eval.json")
        baseline_inputs = evaluation.read_json(ROOT / baseline["run_dir"] / "inputs.json")
        if config != baseline_inputs["protocol"]:
            raise ValueError("Scoring protocol differs from the complete baseline")
        if config["expected_examples"] != 625 or len(rows) != 625 or config["workers"] != 4:
            raise ValueError("Expected the frozen complete 625-example / four-worker protocol")
        raw_predictions = evaluation.read_jsonl(predictions_path)
        check_generation(generation, raw_predictions, [row["source_index"] for row in rows])
        predictions = evaluation.validate_predictions(raw_predictions, rows)
        fingerprints[evaluation.relative(GENERATION_REPORT)] = evaluation.digest(GENERATION_REPORT)
        fingerprints[evaluation.relative(controller_path)] = evaluation.digest(controller_path)
        fingerprints[evaluation.relative(Path(__file__))] = evaluation.digest(Path(__file__))
        hardness = evaluation.label_hardness(rows)
        RUN_DIR.mkdir(parents=True, exist_ok=False)
        summary.update(run_id=LABEL, run_dir=evaluation.relative(RUN_DIR))
        for path in (Path(evaluation.__file__), evaluation.WORKER, evaluation.CONFIG, Path(__file__)):
            shutil.copyfile(path, RUN_DIR / path.name)
        shutil.copyfile(GENERATION_REPORT, RUN_DIR / "generation_summary.json")
        shutil.copyfile(controller_path, RUN_DIR / "generation_controller_result.json")
        shutil.copyfile(predictions_path, RUN_DIR / "predictions.jsonl")
        write_new(RUN_DIR / "inputs.json", {"input_sha256": fingerprints,
            "prediction_sha256": generation["prediction_sha256"], "prediction_path": str(predictions_path),
            "protocol": config, "source_index_order": [row["source_index"] for row in rows],
            "worker_python": evaluation.WORKER_PYTHON, "worker_site_packages": evaluation.PURELIB,
            "final_examples_read": False})
        results, dispatch_record = dispatch(RUN_DIR, config, rows, predictions, hardness)
        write_new(RUN_DIR / "dispatch.json", dispatch_record)
        summary.update(dispatch=dispatch_record, partial_record_count=len(results),
            per_example_path=evaluation.relative(RUN_DIR / "per_example.jsonl"))
        if dispatch_record["failures"]:
            raise RuntimeError("Scoring infrastructure failure; see dispatch.json; incomplete benchmark is not scored as 625")
        if (sorted(row["source_index"] for row in results) != [row["source_index"] for row in rows]
                or len(dispatch_record["attempted_source_indices"]) != 625
                or len(set(dispatch_record["attempted_source_indices"])) != 625):
            raise ValueError("Incomplete, repeated or replayed scoring requests")
        for filename, expected in fingerprints.items():
            evaluation.expect_hash(ROOT / filename, expected)
        evaluation.expect_hash(predictions_path, generation["prediction_sha256"])
        summary.update(status="pass", metric=config["metric"], protocol_version=config["version"],
            generation_controller_verified=True,
            overall=evaluation.aggregate(results),
            by_hardness={level: evaluation.aggregate([row for row in results if row["hardness"] == level])
                         for level in ("easy", "medium", "hard", "extra")},
            wall_seconds=round(time.perf_counter() - started, 3), source_files_unchanged=True,
            complete_validation_scored=True, complete_test_suite_accuracy=False, upstream_commit=config["upstream_commit"],
            metric_limitations=baseline["metric_limitations"],
            sample_scope="complete 625-question internal validation, not final evaluation",
            generation_run_id=generation["run_id"], prediction_file_sha256=generation["prediction_sha256"],
            prediction_file=evaluation.relative(predictions_path),
            generation_report_sha256=evaluation.digest(GENERATION_REPORT),
            evaluation_pipeline_wall_seconds=round(time.perf_counter() - started, 3),
            evaluation_pipeline_timing_scope="main entry through validation/scoring/integrity checks; excludes Python startup and final summary write")
    except BaseException as error:
        summary.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            evaluation_pipeline_wall_seconds=round(time.perf_counter() - started, 3))
    if RUN_DIR.exists():
        write_new(RUN_DIR / "summary.json", summary)
    write_new(REPORT, summary)
    print(json.dumps({key: summary.get(key) for key in ("status", "overall", "error", "run_dir", "evaluation_pipeline_wall_seconds")}, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="原始625条SFT预测JSONL路径，与生成报告一致")
    raise SystemExit(main(parser.parse_args().predictions))
