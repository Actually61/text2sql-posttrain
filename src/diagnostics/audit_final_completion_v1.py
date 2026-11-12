"""Independently audit saved final-v3 outputs; never generate or execute SQL.

Run only after both generation controllers, scorers and comparison have passed.
The result is exclusive: an incomplete or failed audit is preserved, not retried.
Only Python standard-library modules are imported. Raw final examples/reference
contents are not loaded; their bytes are hashed against existing frozen records.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/final_holdout_v3"
DESTINATION = ROOT / "outputs/checks/final_completion_v1"
COUNT, PREFIX_COUNT = 1034, 616
LEVELS = ("easy", "medium", "hard", "extra")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid nonnegative measurement: {name}")
    return value


def same(actual, expected, label):
    """Compare complete structures; allow only numerical roundoff for floats."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Object keys differ: {label}")
        for key in expected:
            same(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"List shape differs: {label}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            same(left, right, f"{label}[{index}]")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7), f"Numeric value differs: {label}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"Value differs: {label}")


class Evidence:
    def __init__(self):
        self.hashes = {}

    def path(self, name):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), f"Evidence path escaped project: {name}")
        require(path.is_file(), f"Required completed artifact is missing: {path}")
        return path

    def remember(self, path, value):
        key = path.relative_to(ROOT).as_posix()
        require(key not in self.hashes or self.hashes[key] == value, f"Evidence changed during audit: {key}")
        self.hashes[key] = value
        return value

    def digest(self, name):
        path = self.path(name)
        with path.open("rb") as stream:
            value = hashlib.file_digest(stream, "sha256").hexdigest()
        return self.remember(path, value)

    def payload(self, name):
        path = self.path(name)
        payload = path.read_bytes()
        self.remember(path, hashlib.sha256(payload).hexdigest())
        return payload

    def load(self, name, *, lines=False):
        payload = self.payload(name).decode("utf-8")
        return [json.loads(line) for line in payload.splitlines()] if lines else json.loads(payload)

    def verify(self, hashes, label):
        require(isinstance(hashes, dict) and hashes, f"Missing hash mapping: {label}")
        for name, expected in hashes.items():
            require(self.digest(name) == expected, f"Frozen hash mismatch: {label}/{name}")


def aggregate(rows):
    """Recompute saved metric definitions without importing the scoring module."""
    count = len(rows)
    correct = sum(row["matched"] is True for row in rows)
    executable = sum(row["prediction_executable"] is True for row in rows)
    return {
        "count": count, "execution_correct": correct,
        "execution_accuracy": correct / count if count else None,
        "executable_predictions": executable,
        "prediction_executable_rate": executable / count if count else None,
        "prediction_execution_unknown": sum(row["prediction_executable"] is None for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "gold_execution_errors": sum(isinstance(row.get("gold"), dict) and row["gold"]["status"] != "ok" for row in rows),
        "both_results_empty": sum(isinstance(row.get("gold"), dict) and isinstance(row.get("prediction"), dict)
            and row["gold"].get("empty_result") is True and row["prediction"].get("empty_result") is True for row in rows),
    }


def first_error(row):
    metadata = row.get("prediction")
    if metadata is None:
        return "execution_unknown_" + row["status"]
    status = metadata["status"]
    if status == "ok":
        return "none" if row["matched"] is not None else "comparison_or_gold_unresolved"
    if status != "sql_error":
        return status
    error = metadata.get("error", "").lower()
    for prefix, category in (
        ("no such column:", "missing_column"), ("no such table:", "missing_table"),
        ("ambiguous column name:", "ambiguous_column"), ("no such function:", "unsupported_function"),
        ("wrong number of arguments to function", "function_argument_count"),
        ("misuse of aggregate", "aggregate_misuse"), ("misuse of window", "window_function_misuse"),
        ("incomplete input", "incomplete_input"), ("sql is empty", "empty_sql"),
        ("you can only execute one statement", "multiple_statements"),
    ):
        if error.startswith(prefix):
            return category
    if "syntax error" in error or "unrecognized token" in error:
        return "syntax_error"
    if "same number of result columns" in error:
        return "set_operation_column_count"
    if "sub-select returns" in error:
        return "subquery_column_count"
    return "other_sql_error"


def outcome_counts(rows):
    outcomes = []
    for row in rows:
        if row["matched"] is True:
            outcome = "ex_match"
        elif row["prediction_executable"] is True:
            outcome = "executable_mismatch" if row["matched"] is False else "executable_comparison_unresolved"
        else:
            outcome = "execution_unknown" if row["prediction_executable"] is None else "execution_failed"
        outcomes.append(outcome)
    return {"outcomes": dict(Counter(outcomes)),
            "first_error_categories": dict(Counter(first_error(row) for row in rows))}


def group(indices, scores):
    before = aggregate([scores["base"][index] for index in indices])
    after = aggregate([scores["sft"][index] for index in indices])
    cells = Counter((scores["base"][index]["matched"] is True,
                     scores["sft"][index]["matched"] is True) for index in indices)
    require(cells[(False, True)] - cells[(True, False)] == after["execution_correct"] - before["execution_correct"],
            "Paired EX difference does not reconcile")
    return {"count": len(indices), "base": before, "sft": after,
        "ex_delta_percentage_points": 100 * (after["execution_accuracy"] - before["execution_accuracy"]) if indices else None,
        "executable_delta_percentage_points": 100 * (after["prediction_executable_rate"] - before["prediction_executable_rate"]) if indices else None,
        "improved": cells[(False, True)], "regressed": cells[(True, False)],
        "both_correct": cells[(True, True)], "both_wrong": cells[(False, False)]}


def freeze_checks(evidence):
    cfg = evidence.load("configs/final_holdout_resume_v3.json")
    require(cfg["run_root"] == RUN.relative_to(ROOT).as_posix(), "Unexpected final run root")
    require(cfg["expected_examples"] == COUNT and cfg["expected_databases"] == 20, "Frozen final population differs")
    require(cfg["training"] is False and cfg["automatic_retries"] == 0 and cfg["truncate"] is False,
            "Frozen scope permits training/retries/truncation")
    old = evidence.load("reports/final_holdout_v1_freeze.json")
    current = evidence.load(cfg["frozen_report"])
    require(old["status"] == "frozen_before_final_examples", "Original freeze status invalid")
    require(current["config"] == cfg, "Current freeze/config differ")
    evidence.verify(old["source_sha256"], "original freeze")
    evidence.verify(current["source_sha256"], "v3 freeze")
    parent = evidence.load(cfg["parent_preflight"])
    checked = evidence.load(cfg["preflight_report"])
    require(parent["status"] == checked["status"] == "pass", "Preflight did not pass")
    require(checked["config"] == cfg, "V3 preflight/config differ")
    require(checked["frozen_report_sha256"] == evidence.digest(cfg["frozen_report"]), "V3 freeze binding differs")
    for field in ("source_indices", "model_config", "weight_path", "weight_sha256", "adapter_fingerprint",
                  "input_tokens_max", "input_manifest_sha256", "output_sha256", "generation_config"):
        require(checked[field] == parent[field], f"Prepared protocol changed: {field}")
    require(checked["source_indices"] == list(range(COUNT)), "Prepared final order differs")
    require(checked["training_performed"] is False and checked["cuda_initialized"] is False,
            "CPU preflight scope changed")
    evidence.verify(parent["output_sha256"], "prepared inputs/references")
    manifest_path = ROOT / cfg["input_dir"] / "manifest.json"
    require(evidence.digest(manifest_path) == parent["input_manifest_sha256"], "Prepared manifest changed")
    manifest = evidence.load(manifest_path)
    require(manifest["recipe_freeze_sha256"] == evidence.digest("reports/final_holdout_v1_freeze.json"),
            "Input manifest does not bind original freeze")
    require(manifest["training_performed"] is False and manifest["final_used_for_tuning"] is False,
            "Prepared-data no-training/no-tuning declaration invalid")
    return cfg, checked, {"original_frozen_file_count": len(old["source_sha256"]),
                         "v3_frozen_file_count": len(current["source_sha256"])}


def audit_generation(evidence, arm, cfg, checked):
    directory = RUN / arm
    summary = evidence.load(directory / "summary.json")
    controller = evidence.load(directory / "controller_result.json")
    require(summary["status"] == controller["status"] == "pass", f"Incomplete generation/controller: {arm}")
    require(controller["exit_code"] == 0 and controller["job_active_pids_after_cleanup"] == [],
            f"Worker exit/Job cleanup failed: {arm}")
    require(summary["arm"] == controller["arm"] == arm, "Generation arm identity differs")
    require(all(summary[key] == COUNT for key in ("prediction_count", "generated_count", "attempted_count")),
            f"Incomplete canonical generation count: {arm}")
    for key in ("training_performed", "reference_answers_loaded_for_inference", "databases_opened", "prediction_selection_performed"):
        require(summary[key] is False, f"Unexpected generation scope: {arm}/{key}")
    for key in ("automatic_retries", "warmup_generations", "generation_failure_or_skip_count"):
        require(summary[key] == 0, f"Unexpected retries/warmup/failures: {arm}/{key}")
    require(summary["all_inputs_complete"] is True and summary["is_loaded_in_4bit"] is True,
            f"Incomplete input or changed quantization: {arm}")
    require(summary["project_fine_tuning_applied"] is (arm == "sft"), "Adapter application scope differs")
    for key in ("generation_config", "model_config", "packages"):
        require(summary[key] == checked[key], f"Frozen inference setting differs: {arm}/{key}")
    require(summary["model_weight_sha256"] == checked["weight_sha256"], "Model weight identity differs")
    require(summary["adapter_fingerprint"] == (checked["adapter_fingerprint"] if arm == "sft" else None),
            "Adapter fingerprint differs")
    require(summary["adapter_path"] == (cfg["adapter_path"] if arm == "sft" else None), "Adapter checkpoint differs")
    prefix_count = PREFIX_COUNT if arm == "base" else 0
    require(summary["reused_prefix_count"] == prefix_count and summary["new_generated_count"] == COUNT-prefix_count,
            "Prefix/continuation counts differ")
    prediction_path = directory / "predictions.jsonl"
    require(summary["prediction_path"] == prediction_path.relative_to(ROOT).as_posix(), "Prediction path differs")
    require(evidence.digest(prediction_path) == summary["prediction_sha256"], "Prediction hash differs")
    raw = evidence.payload(prediction_path)
    rows = evidence.load(prediction_path, lines=True)
    require(len(rows) == COUNT and [row["source_index"] for row in rows] == list(range(COUNT)),
            "Canonical predictions must be unique and ordered")
    prompts = evidence.load(directory / "prompts.jsonl", lines=True)
    require(len(prompts) == COUNT and [row["source_index"] for row in prompts] == list(range(COUNT)),
            "Prompt population/order differs")
    require(evidence.digest(directory / "prompts.jsonl") == summary["prompts_sha256"], "Prompt hash differs")
    require(evidence.load(directory / "generation_config.json") == summary["generation_config"], "Generation config artifact differs")
    for ordinal, (row, prompt) in enumerate(zip(rows, prompts, strict=True), 1):
        require(row["ordinal"] == ordinal and row["attempt_number"] == 1
                and row["generation_ok"] is True and row["generation_status"] == "generated", "Not one valid attempt per input")
        require(row["db_id"] == prompt["db_id"] and row["input_token_count"] == prompt["input_token_count"], "Prediction/prompt identity differs")
        require(isinstance(row["predicted_sql"], str), "Missing predicted SQL")
        require(type(row["generated_token_count"]) is int and 0 < row["generated_token_count"] <= 256
                and len(row["new_token_ids"]) == row["generated_token_count"], "Generated token count invalid")
        require(type(row["ended_with_eos"]) is bool and type(row["reached_token_limit"]) is bool
                and row["ended_with_eos"] != row["reached_token_limit"], "Unexpected generation stop condition")
        require(row["effective_max_time_seconds"] == 30.0, "Per-question time guard changed")
        finite_nonnegative(row["generation_seconds"], "per-example generation_seconds")
        for key, limit in (("peak_allocated_mib", cfg["allocated_limit_mib"]), ("peak_reserved_mib", cfg["reserved_limit_mib"])):
            require(finite_nonnegative(row[key], key) <= limit, "Prediction memory threshold exceeded")
    require(len({row["db_id"] for row in rows}) == 20, "Final database population differs")
    require(summary["source_indices"] == list(range(COUNT)), "Summary IDs differ")
    require(summary["input_tokens_max"] == max(row["input_token_count"] for row in rows) == checked["input_tokens_max"],
            "Input length maximum differs")
    require(summary["ended_with_eos_count"] == sum(row["ended_with_eos"] for row in rows), "EOS total differs")
    require(summary["token_limit_count"] == sum(row["reached_token_limit"] for row in rows), "Token-limit total differs")
    require(summary["generated_tokens_total"] == sum(row["generated_token_count"] for row in rows), "Generated-token total differs")
    same(summary["generation_seconds_total"], sum(row["generation_seconds"] for row in rows), "Canonical generation seconds")
    same(summary["generation_seconds_median"], statistics.median(row["generation_seconds"] for row in rows), "Median generation seconds")
    prior, new = rows[:prefix_count], rows[prefix_count:]
    same(summary["prior_generation_seconds_total"], sum(row["generation_seconds"] for row in prior), "Prefix generation seconds")
    same(summary["new_generation_seconds_total"], sum(row["generation_seconds"] for row in new), "New generation seconds")
    for suffix in ("allocated_mib", "reserved_mib"):
        same(summary[f"generation_peak_{suffix}"], max(row[f"peak_{suffix}"] for row in new), f"New peak {suffix}")
        same(summary[f"canonical_generation_peak_{suffix}"], max(row[f"peak_{suffix}"] for row in rows), f"Canonical peak {suffix}")
        same(summary[f"prior_generation_peak_{suffix}"], max((row[f"peak_{suffix}"] for row in prior), default=None), f"Prior peak {suffix}")
    gpu = evidence.load(directory / "gpu_samples.json")
    telemetry = evidence.load(directory / "telemetry_events.json")
    require(gpu["errors"] == [] and telemetry["errors"] == [] and telemetry["fatal"] is None,
            "New telemetry has errors")
    require(telemetry["helper_launches"] == 1 and telemetry["automatic_retries"] == telemetry["automatic_restarts"] == 0,
            "Telemetry helper restarted")
    require(telemetry["close"]["status"] == "pass" and telemetry["close"]["alive_threads"] == [],
            "Telemetry helper cleanup did not pass")
    require(summary["nvidia_smi_sample_count"] == len(gpu["samples"]) == telemetry["sample_count"], "Telemetry sample count differs")
    require(summary["nvidia_smi_sampling_error_count"] == 0, "Nonzero new telemetry error count")
    same(summary["nvidia_smi_sampled_peak_used_mib"], max(row["used_mib"] for row in gpu["samples"]), "New whole-GPU sampled peak")
    if prefix_count:
        prefix = evidence.payload(cfg["prefix_path"])
        require(evidence.digest(cfg["prefix_path"]) == cfg["prefix_sha256"] == summary["reused_prefix_sha256"], "Prefix hash differs")
        require(prefix.endswith(b"\n") and len(prefix.splitlines()) == PREFIX_COUNT
                and b"".join(raw.splitlines(keepends=True)[:PREFIX_COUNT]) == prefix,
                "Original 616-row prefix bytes changed")
        old_controller = evidence.load(Path(cfg["prefix_path"]).parent / "controller_result.json")
        old_gpu = evidence.load(Path(cfg["prefix_path"]).parent / "gpu_samples.json")
        require(old_controller["status"] == "failed" and old_controller["job_active_pids_after_cleanup"] == [],
                "Original failure/cleanup evidence differs")
        same(summary["prior_controller_seconds"], old_controller["controller_seconds"], "Prior controller time")
        require(summary["prior_telemetry_error_count"] == len(old_gpu["errors"]), "Prior telemetry error total differs")
        same(summary["prior_nvidia_smi_sampled_peak_used_mib"], max(row["used_mib"] for row in old_gpu["samples"]), "Prior whole-GPU sampled peak")
    else:
        require(summary["prior_controller_seconds"] == 0.0 and summary["prior_telemetry_error_count"] == 0,
                "SFT must not reuse prior generation")
    for key in ("end_to_end_seconds", "load_seconds", "generation_loop_wall_seconds"):
        finite_nonnegative(summary[key], key)
    finite_nonnegative(controller["controller_seconds"], "controller_seconds")
    cost = {
        "canonical_generated_question_count": COUNT, "reused_prefix_count": prefix_count,
        "new_generated_count": len(new), "canonical_per_question_generation_seconds": summary["generation_seconds_total"],
        "old_prefix_per_question_generation_seconds": summary["prior_generation_seconds_total"],
        "new_per_question_generation_seconds": summary["new_generation_seconds_total"],
        "old_failed_controller_seconds": summary["prior_controller_seconds"],
        "new_worker_end_to_end_seconds": summary["end_to_end_seconds"],
        "new_controller_seconds": controller["controller_seconds"],
        "both_generation_controller_windows_seconds": summary["prior_controller_seconds"] + controller["controller_seconds"],
        "new_generation_loop_seconds": summary["generation_loop_wall_seconds"],
        "memory": {key: summary[key] for key in (
            "load_peak_allocated_mib", "load_peak_reserved_mib", "generation_peak_allocated_mib",
            "generation_peak_reserved_mib", "canonical_generation_peak_allocated_mib",
            "canonical_generation_peak_reserved_mib", "prior_generation_peak_allocated_mib",
            "prior_generation_peak_reserved_mib", "nvidia_smi_sampled_peak_used_mib",
            "prior_nvidia_smi_sampled_peak_used_mib")},
        "new_process_allocator_counters": {key: summary[key] for key in ("num_ooms", "num_alloc_retries")},
        "old_process_final_allocator_counters_available": False if prefix_count else None,
        "timing_limitation": "Controller-window sum includes old failure overhead and new process, excludes pauses, CPU preparation and documentation; not uninterrupted elapsed time",
        "memory_limitation": "Allocated is nested in reserved; never add them. Whole-GPU sampled occupancy includes other applications; maxima are not summed across processes.",
    }
    return summary, rows, cost


def audit_scores(evidence, arm, predictions):
    directory = RUN / f"scores_{arm}"
    summary = evidence.load(directory / "summary.json")
    require(summary["status"] == "pass" and summary["complete_final_scored"] is True, f"Scoring incomplete: {arm}")
    require(summary["arm"] == arm and summary["training_performed"] is False
            and summary["model_generation_performed"] is False and summary["automatic_retries"] == 0,
            f"Unexpected scoring scope: {arm}")
    require(summary["complete_test_suite_accuracy"] is False and summary["source_files_unchanged"] is True,
            "Scorer limitations/integrity declaration differs")
    require(summary["prediction_sha256"] == evidence.digest(RUN / arm / "predictions.jsonl"), "Scorer prediction hash differs")
    rows = evidence.load(directory / "per_example.jsonl", lines=True)
    require(len(rows) == COUNT and sorted(row["source_index"] for row in rows) == list(range(COUNT)),
            "Scored records incomplete/duplicated")
    scores = {row["source_index"]: row for row in rows}
    for index, prediction in enumerate(predictions):
        row = scores[index]
        require(row["db_id"] == prediction["db_id"] and row["predicted_sql"] == prediction["predicted_sql"],
                f"Scored SQL differs from actual one-shot output: {arm}/{index}")
        require(row["hardness"] in LEVELS, "Unexpected saved difficulty")
        require(row["gold"]["status"] == "ok", "Complete scorer contains gold execution failure")
        require(row["status"] in {"scored", "prediction_error", "comparison_error"}, "Complete scorer contains infrastructure failure")
        require(row["matched"] is None or type(row["matched"]) is bool, "Invalid EX boolean")
        require(type(row["prediction_executable"]) is bool, "Unknown executability in completed score")
        require(row["prediction_executable"] is (row["prediction"]["status"] == "ok"), "Executability metadata inconsistent")
        if row["status"] == "scored":
            require(type(row["matched"]) is bool and row["prediction_executable"] is True, "Scored record invalid")
        else:
            require(row["matched"] is None, "Unscored record claims EX")
        if row["status"] == "comparison_error":
            require(row["error"] == "Comparator returned after its time budget", "Unexpected comparator exception")
    same(summary["overall"], aggregate(rows), f"{arm} overall")
    same(summary["by_hardness"], {level: aggregate([row for row in rows if row["hardness"] == level]) for level in LEVELS}, f"{arm} hardness")
    same(summary["by_database"], {db: aggregate([row for row in rows if row["db_id"] == db]) for db in sorted({row["db_id"] for row in rows})}, f"{arm} databases")
    same(summary["failure_types"], outcome_counts(rows), f"{arm} failure types")
    dispatch = evidence.load(directory / "dispatch.json")
    require(dispatch == summary["dispatch"] and dispatch["failures"] == [] and dispatch["automatic_retries"] == 0,
            "Scoring dispatch differs/failed")
    require(dispatch["attempted_count"] == dispatch["completed_count"] == COUNT
            and sorted(dispatch["attempted_source_indices"]) == list(range(COUNT)), "Dispatch coverage/replay failure")
    protocol = evidence.load(directory / "protocol.json")
    require(protocol["expected_examples"] == COUNT and protocol["split"] == "final" and protocol["workers"] == 4,
            "Final scoring protocol population differs")
    original = evidence.load("configs/evaluation.json")
    for key in ("metric", "version", "comparator", "upstream_commit", "database_dir", "plug_value", "keep_distinct",
                "rewrite_sql", "query_timeout_seconds", "comparison_timeout_seconds", "hard_timeout_grace_seconds", "workers", "seed"):
        require(protocol[key] == original[key], f"EX protocol changed: {key}")
    finite_nonnegative(summary["wall_seconds"], "scoring wall_seconds")
    return summary, scores


def audit(evidence):
    cfg, checked, freeze_counts = freeze_checks(evidence)
    generations, predictions, costs, reports, scores = {}, {}, {}, {}, {}
    for arm in ("base", "sft"):
        generations[arm], predictions[arm], costs[arm] = audit_generation(evidence, arm, cfg, checked)
        reports[arm], scores[arm] = audit_scores(evidence, arm, predictions[arm])
        costs[arm]["scoring_wall_seconds"] = reports[arm]["wall_seconds"]
        costs[arm]["generation_controller_windows_plus_scoring_seconds"] = (
            costs[arm]["both_generation_controller_windows_seconds"] + reports[arm]["wall_seconds"])
    for filename in ("prompts.jsonl", "generation_config.json"):
        require(evidence.payload(RUN / "base" / filename) == evidence.payload(RUN / "sft" / filename),
                f"Generation artifacts differ: {filename}")
    require(evidence.payload(RUN / "base" / "prompts.jsonl") == evidence.payload("outputs/final_holdout_v1/base/prompts.jsonl"),
            "Prompts changed after first prefix")
    require(evidence.load(RUN / "scores_base/protocol.json") == evidence.load(RUN / "scores_sft/protocol.json"),
            "Two-arm EX protocols differ")
    for index in range(COUNT):
        for key in ("source_index", "db_id", "hardness", "question", "gold_sql", "order_matters"):
            require(scores["base"][index].get(key) == scores["sft"][index].get(key), f"Paired metadata differs: {index}/{key}")
    comparison = evidence.load(RUN / "comparison.json")
    require(comparison["status"] == "pass", "Paired comparison incomplete")
    for key in ("training_performed", "final_result_used_for_tuning", "prediction_replay_or_selection_performed"):
        require(comparison[key] is False, f"Invalid paired scope: {key}")
    require(comparison["automatic_retries"] == 0 and comparison["same_prompts_and_generation_config"] is True
            and comparison["same_ex_protocol"] is True, "Paired fairness declaration differs")
    overall = group(list(range(COUNT)), scores)
    by_hardness = {level: group([index for index in range(COUNT) if scores["base"][index]["hardness"] == level], scores) for level in LEVELS}
    failure_types = {arm: outcome_counts(list(scores[arm].values())) for arm in ("base", "sft")}
    same(comparison["overall"], overall, "Paired overall")
    same(comparison["by_hardness"], by_hardness, "Paired hardness")
    same(comparison["failure_types"], failure_types, "Paired failure types")
    improved = [index for index in range(COUNT) if scores["base"][index]["matched"] is not True and scores["sft"][index]["matched"] is True]
    regressed = [index for index in range(COUNT) if scores["base"][index]["matched"] is True and scores["sft"][index]["matched"] is not True]
    same(comparison["improved_source_indices"], improved, "Improved IDs")
    same(comparison["regressed_source_indices"], regressed, "Regressed IDs")
    require(comparison["baseline_prefix_reused_count"] == PREFIX_COUNT
            and comparison["baseline_new_generated_count"] == COUNT-PREFIX_COUNT
            and comparison["sft_new_generated_count"] == COUNT
            and comparison["baseline_prefix_sha256"] == cfg["prefix_sha256"], "Comparison lineage differs")
    evidence.verify(comparison["source_sha256"], "paired comparison evidence")
    require(reports["base"]["metric_limitations"] == reports["sft"]["metric_limitations"] == comparison["metric_limitations"],
            "Metric limitation declarations differ")
    # Rehash every read artifact once at the end; never rewrite frozen evidence.
    evidence.verify(dict(evidence.hashes), "final audit integrity")
    return {
        "checks": {"frozen_sources_unchanged": True, "same_1034_ordered_once_generated_predictions": True,
            "original_616_prediction_bytes_retained": True, "real_worker_exit_and_empty_jobs_verified": True,
            "same_prompts_generation_and_ex_protocol": True, "scores_bound_to_actual_predicted_SQL": True,
            "independently_recomputed_overall_difficulty_database_error_and_paired_metrics": True,
            "no_training_and_no_final_result_tuning_declarations_verified": True,
            "no_new_model_or_SQL_execution_during_audit": True},
        "freeze_counts": freeze_counts,
        "report_consumables": {
            "scope": "1034 project-held-out official-dev questions on20 databases; base versus preselected round1 update796",
            "overall": overall, "by_hardness": by_hardness, "failure_types": failure_types,
            "costs": costs, "combined_recorded_generation_controller_windows_and_scoring_seconds": sum(
                cost["generation_controller_windows_plus_scoring_seconds"] for cost in costs.values()),
            "input_tokens_min": checked["input_tokens_min"], "input_tokens_max": checked["input_tokens_max"],
            "metric_limitations": comparison["metric_limitations"],
            "claims_boundary": "Declaration and source-hash verification, not proof that no external unrecorded activity ever occurred; no statistical significance claim",
            "final_tuning_performed": False,
        },
    }


def main():
    started = time.perf_counter()
    DESTINATION.mkdir(parents=True, exist_ok=False)
    result = {"status": "failed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_version": "final_completion_v1", "automatic_retries": 0,
        "models_loaded": False, "SQL_executed": False, "training_performed": False,
        "raw_final_examples_or_reference_contents_loaded": False}
    evidence = Evidence()
    try:
        evidence.digest(Path(__file__))
        result.update(audit(evidence))
        result["status"] = "pass"
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    result["source_sha256"] = evidence.hashes
    result["audit_seconds"] = time.perf_counter() - started
    with (DESTINATION / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({key: result.get(key) for key in ("status", "error", "audit_seconds")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
