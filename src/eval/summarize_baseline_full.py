"""Audit and summarize one completed, uniformly regenerated 625-question run.

Only reads saved generation/scoring evidence; never imports a model, opens a
SQLite connection, or reads final examples. Default execution requires the full
generation and evaluation reports. --self-test uses synthetic objects only.
The old 32 predictions are compared, never selected or spliced into the new run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("easy", "medium", "hard", "extra")
POPULATION = {"easy": 138, "medium": 247, "hard": 152, "extra": 88}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_path(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), f"Evidence must stay inside the project: {value}")
    # Split manifests may describe final membership, but final examples/DBs
    # themselves must never be consumed by this analysis.
    relative = path.relative_to(ROOT).as_posix().lower()
    require("/test_database/" not in relative and path.name.lower() not in {
        "dev.json", "test.json", "final_eval.jsonl", "final_eval_inputs.jsonl",
        "final_eval_references.jsonl", "test.jsonl", "dev.jsonl",
    }, f"Final/test example access is out of scope: {value}")
    return path


class Evidence:
    def __init__(self):
        self.hashes = {}

    def remember(self, path, sha):
        key = path.relative_to(ROOT).as_posix()
        require(key not in self.hashes or self.hashes[key] == sha,
                f"Evidence changed while reading: {key}")
        self.hashes[key] = sha
        return sha

    def digest(self, value):
        path = project_path(value)
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        return self.remember(path, sha)

    def load(self, value, lines=False):
        path = project_path(value)
        payload = path.read_bytes()
        self.remember(path, hashlib.sha256(payload).hexdigest())
        text = payload.decode("utf-8")
        return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)

    def verify(self, declared):
        require(isinstance(declared, dict) and declared, "Missing source hash chain")
        for path, expected in declared.items():
            require(self.digest(path) == expected, f"Recorded source hash mismatch: {path}")


def index_rows(rows, expected_count, label):
    require(len(rows) == expected_count, f"{label}: expected {expected_count} rows")
    result = {}
    for row in rows:
        index = row.get("source_index")
        require(type(index) is int and index not in result, f"{label}: invalid/duplicate source ID")
        require(isinstance(row.get("db_id"), str), f"{label}: missing database")
        result[index] = row
    return result


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid nonnegative measurement: {name}")
    return value


def execution_outcome(row):
    flag, matched = row["prediction_executable"], row["matched"]
    if matched is True:
        return "ex_match"
    if flag is True:
        return "executable_mismatch" if matched is False else "executable_comparison_unresolved"
    return "execution_unknown" if flag is None else "execution_failed"


def first_error_category(row):
    """Bucket the observed first error, independently of generation stop reason."""
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
        ("ambiguous column name:", "ambiguous_column"),
        ("no such function:", "unsupported_function"),
        ("wrong number of arguments to function", "function_argument_count"),
        ("misuse of aggregate", "aggregate_misuse"),
        ("misuse of window", "window_function_misuse"),
        ("incomplete input", "incomplete_input"),
        ("sql is empty", "empty_sql"),
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


def stop_category(row):
    if row["generation_ok"] is not True:
        return "generation_failed_or_skipped"
    if row["ended_with_eos"]:
        return "eos"
    if row["reached_token_limit"]:
        return "token_limit"
    # The pilot runner did not instrument the stopping criterion. A non-EOS,
    # sub-budget output alone is NOT proof that max_time fired.
    raw = row.get("stop_reason", "")
    if raw in {"max_time", "time_limit", "cooperative_timeout", "generation_timeout"}:
        return "recorded_time_limit"
    return "time_or_other_stop_unresolved"


def aggregate(rows):
    count = len(rows)
    valid = sum(row["prediction_executable"] is True for row in rows)
    correct = sum(row["matched"] is True for row in rows)
    return {
        "count": count, "execution_correct": correct,
        "execution_accuracy": correct / count if count else None,
        "executable_predictions": valid,
        "prediction_executable_rate": valid / count if count else None,
        "prediction_execution_unknown": sum(row["prediction_executable"] is None for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "gold_execution_errors": sum(isinstance(row.get("gold"), dict) and
                                     row["gold"]["status"] != "ok" for row in rows),
        "both_results_empty": sum((row.get("gold") or {}).get("empty_result") is True and
                                  (row.get("prediction") or {}).get("empty_result") is True for row in rows),
    }


def load_generation(evidence, report_path, count):
    report = evidence.load(report_path)
    require(report["prediction_count"] == count, "Wrong generation report sample size")
    require(report["project_fine_tuning_applied"] is False and
            report["final_examples_read"] is False,
            "Generation scope/provenance is incompatible")
    if count == 625:
        require(report["reference_answers_loaded_for_inference"] is False and
                report["validation_reference_files_read"] is False and
                report["pilot_predictions_reused"] is False and
                report["prediction_selection_performed"] is False,
                "Full generation inference scope is incompatible")
        # The generator hashes protected pilot files as opaque bytes, including
        # old evaluation evidence. This is not loading gold answers for inference.
        require(isinstance(report["pilot_evidence_hash_policy"], str), "Missing pilot hash policy")
        evidence.verify(report["protected_pilot_sha256"])
    else:
        require(report["reference_answers_read_by_generator"] is False,
                "Pilot generation inference scope is incompatible")
    run_dir = project_path(report["run_dir"])
    predictions_path = project_path(report["prediction_path"])
    require(predictions_path.parent == run_dir, "Predictions are not in their generation run")
    require(evidence.digest(predictions_path) == report["prediction_sha256"],
            "Prediction SHA differs from generation report")
    saved_summary = evidence.load(run_dir / "summary.json")
    require(saved_summary == report, "Published generation report differs from saved run summary")
    evidence.verify(report["source_sha256"])
    predictions = index_rows(evidence.load(predictions_path, lines=True), count, "generation")
    for row in predictions.values():
        require(isinstance(row.get("predicted_sql"), str), "Missing predicted SQL")
        require(type(row.get("generation_ok")) is bool and type(row.get("ended_with_eos")) is bool
                and type(row.get("reached_token_limit")) is bool, "Missing generation flags")
        nonnegative(row["generation_seconds"], "generation_seconds")
        require(type(row["generated_token_count"]) is int and row["generated_token_count"] >= 0,
                "Invalid generated token count")
        if "new_token_ids" in row:
            require(len(row["new_token_ids"]) == row["generated_token_count"], "Token count mismatch")
    require(sum(r["generated_token_count"] for r in predictions.values()) == report["generated_tokens_total"],
            "Generation token total mismatch")
    require(math.isclose(sum(r["generation_seconds"] for r in predictions.values()),
                         report["generation_seconds_total"], rel_tol=1e-8, abs_tol=1e-6),
            "Generation elapsed total mismatch")
    require(sum(r["generation_ok"] for r in predictions.values()) == report["generated_count"],
            "Generation success count mismatch")
    for key in ("load_seconds", "end_to_end_seconds", "generation_loop_wall_seconds"):
        nonnegative(report[key], key)
    return report, predictions


def load_evaluation(evidence, report_path, generation, predictions, count):
    report = evidence.load(report_path)
    run_dir = project_path(report["run_dir"])
    inputs = evidence.load(run_dir / "inputs.json")
    saved = evidence.load(run_dir / "summary.json")
    for key, value in saved.items():
        require(report.get(key) == value, f"Published evaluation differs from run summary: {key}")
    require(inputs["prediction_sha256"] == generation["prediction_sha256"],
            "Generation/evaluation prediction SHA mismatch")
    require(evidence.digest(run_dir / "predictions.jsonl") == generation["prediction_sha256"],
            "Evaluation prediction snapshot changed")
    if "prediction_file_sha256" in report:
        require(report["prediction_file_sha256"] == generation["prediction_sha256"],
                "Published evaluation prediction SHA mismatch")
    if "generation_run_id" in report:
        require(report["generation_run_id"] == generation["run_id"], "Generation run ID mismatch")
    protocol = inputs["protocol"]
    require(protocol["expected_examples"] == count and protocol["split"] ==
            ("validation" if count == 625 else "validation_pilot"), "Wrong evaluation split/count")
    require(protocol["keep_distinct"] is True and protocol["rewrite_sql"] is False
            and protocol["plug_value"] is False, "Evaluation protocol changed SQL semantics")
    require(inputs["final_examples_read"] is False and report["final_examples_read"] is False,
            "Unexpected final-data use")
    require(report["metric"] == protocol["metric"] and report["protocol_version"] == protocol["version"]
            and report["upstream_commit"] == protocol["upstream_commit"], "Protocol metadata mismatch")
    evidence.verify(inputs["input_sha256"])
    scored_path = project_path(report["per_example_path"])
    require(scored_path.parent == run_dir, "Scores are not in their evaluation run")
    scores = index_rows(evidence.load(scored_path, lines=True), count, "evaluation")
    require(set(scores) == set(predictions) == set(inputs["source_index_order"])
            and len(inputs["source_index_order"]) == count, "Evaluation ID coverage mismatch")
    for index, row in scores.items():
        pred = predictions[index]
        require(row["db_id"] == pred["db_id"] and row["predicted_sql"] == pred["predicted_sql"],
                f"Generation/score differs for source_index={index}")
        require(row["prediction_executable"] is None or type(row["prediction_executable"]) is bool,
                "Invalid executable flag")
        require(row["matched"] is None or type(row["matched"]) is bool, "Invalid match flag")
        require(row["hardness"] in LEVELS, "Unknown hardness")
        require(row["matched"] is not True or row["prediction_executable"] is True,
                "Non-executable SQL cannot be an EX match")
    require(aggregate(list(scores.values())) == report["overall"], "Overall scoring totals mismatch")
    for level in LEVELS:
        require(aggregate([r for r in scores.values() if r["hardness"] == level]) == report["by_hardness"][level],
                f"Hardness scoring totals mismatch: {level}")
    return report, inputs, scores


def measurements(values):
    return {"count": len(values), "total": sum(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def analyze_group(indices, predictions, scores):
    selected_scores = [scores[i] for i in indices]
    return {
        "evaluation": aggregate(selected_scores),
        "generation_seconds": measurements([predictions[i]["generation_seconds"] for i in indices]),
        "generated_token_count": measurements([predictions[i]["generated_token_count"] for i in indices]),
        "input_token_count": measurements([predictions[i]["input_token_count"] for i in indices]),
        "stop_category_counts": dict(Counter(stop_category(predictions[i]) for i in indices)),
        "raw_stop_reason_counts": dict(Counter(predictions[i].get("stop_reason", "not_recorded") for i in indices)),
        "generation_status_counts": dict(Counter(predictions[i]["generation_status"] for i in indices)),
        "outcome_counts": dict(Counter(execution_outcome(r) for r in selected_scores)),
        "first_execution_error_counts": dict(Counter(first_error_category(r) for r in selected_scores
                                                   if r["prediction_executable"] is not True)),
        "both_empty_match_source_indices": [r["source_index"] for r in selected_scores if r["matched"] is True
            and (r.get("gold") or {}).get("empty_result") is True
            and (r.get("prediction") or {}).get("empty_result") is True],
    }


def main():
    evidence = Evidence()
    generation, predictions = load_generation(evidence, "reports/baseline_full_generation.json", 625)
    evaluation, eval_inputs, scores = load_evaluation(
        evidence, "reports/baseline_full_eval.json", generation, predictions, 625)
    require(generation["complete_validation_records"] is True, "Not a complete full record set")
    require(generation["complete_validation_generated"] == (generation["generated_count"] == 625),
            "Full generation completion flag differs from success count")
    validation = index_rows(evidence.load(eval_inputs["protocol"]["validation_file"], lines=True), 625, "validation")
    require(set(validation) == set(predictions), "Predictions do not cover all frozen validation IDs")
    for index, source in validation.items():
        require(source["project_split"] == "validation" and source["db_id"] == predictions[index]["db_id"]
                and source["query"] == scores[index]["gold_sql"], "Validation/generation/gold association differs")
    require(dict(Counter(r["hardness"] for r in scores.values())) == POPULATION,
            "Full hardness distribution differs from frozen population")
    pilot_gen, pilot_predictions = load_generation(evidence, "reports/baseline_pilot_generation.json", 32)
    require(generation["generation_protocol_matches_pilot"] is True and
            generation["generation_config"] == pilot_gen["generation_config"],
            "Resolved full generation configuration differs from pilot")
    for key in ("model_config", "model_weight_sha256", "quantization", "packages"):
        require(generation[key] == pilot_gen[key], f"Full/pilot inference provenance differs: {key}")
    pilot_eval, pilot_inputs, pilot_scores = load_evaluation(
        evidence, "reports/baseline_pilot_eval.json", pilot_gen, pilot_predictions, 32)
    manifest = evidence.load("reports/baseline_pilot_manifest.json")
    require(set(pilot_predictions) == set(manifest["selected_source_indices"]) and len(manifest["selected_source_indices"]) == 32,
            "Pilot IDs differ from the frozen sample")
    require(evidence.digest("reports/baseline_pilot_manifest.json") == pilot_gen["pilot_manifest_sha256"]
            == pilot_eval["pilot_manifest_sha256"], "Pilot manifest SHA mismatch")
    require(set(pilot_predictions) <= set(predictions), "Pilot is not contained in full validation")
    excluded = {"split", "expected_examples", "denominator_policy"}
    protocol_differences = {k: {"pilot": pilot_inputs["protocol"].get(k), "full": eval_inputs["protocol"].get(k)}
        for k in set(pilot_inputs["protocol"]) | set(eval_inputs["protocol"])
        if k not in excluded and pilot_inputs["protocol"].get(k) != eval_inputs["protocol"].get(k)}
    require(not protocol_differences, "Pilot/full scoring protocol differs beyond expected scope")
    overlap = []
    for index in sorted(pilot_predictions):
        before, after = pilot_predictions[index], predictions[index]
        old_score, new_score = pilot_scores[index], scores[index]
        require(before["db_id"] == after["db_id"] and old_score["hardness"] == new_score["hardness"],
                "Pilot/full metadata differs")
        overlap.append({"source_index": index, "db_id": after["db_id"], "hardness": new_score["hardness"],
            "sql_identical": before["predicted_sql"] == after["predicted_sql"],
            "token_ids_identical": before.get("new_token_ids") == after.get("new_token_ids"),
            "pilot_predicted_sql": before["predicted_sql"], "full_predicted_sql": after["predicted_sql"],
            "pilot_matched": old_score["matched"], "full_matched": new_score["matched"],
            "pilot_executable": old_score["prediction_executable"], "full_executable": new_score["prediction_executable"],
            "pilot_stop_reason": before.get("stop_reason"), "full_stop_reason": after.get("stop_reason"),
            "pilot_seconds": before["generation_seconds"], "full_seconds": after["generation_seconds"]})
    pilot_analysis = evidence.load("reports/baseline_pilot_analysis.json")
    evidence.verify(pilot_analysis["source_sha256"])
    projected = pilot_analysis["full_625_estimate"]["projected_total_seconds"]
    require(nonnegative(projected, "pilot projected seconds") > 0, "Projection must be positive")
    actual_scoring = evaluation.get("evaluation_pipeline_wall_seconds", evaluation["wall_seconds"])
    nonnegative(actual_scoring, "evaluation pipeline seconds")
    actual = generation["end_to_end_seconds"] + actual_scoring
    details = []
    for index in sorted(predictions):
        pred, score = predictions[index], scores[index]
        metadata = score.get("prediction") or {}
        details.append({"source_index": index, "db_id": pred["db_id"], "hardness": score["hardness"],
            "outcome": execution_outcome(score), "prediction_executable": score["prediction_executable"],
            "matched": score["matched"], "first_error_category": first_error_category(score),
            "prediction_execution_status": metadata.get("status"), "first_sqlite_error": metadata.get("error"),
            "sqlite_errorcode": metadata.get("sqlite_errorcode"), "sqlite_errorname": metadata.get("sqlite_errorname"),
            "evaluation_status": score["status"], "evaluation_error": score.get("error"),
            "generation_status": pred["generation_status"], "generation_stop_category": stop_category(pred),
            "raw_stop_reason": pred.get("stop_reason"), "generated_token_count": pred["generated_token_count"],
            "generation_seconds": pred["generation_seconds"],
            "sql_utf8_sha256": hashlib.sha256(pred["predicted_sql"].encode("utf-8")).hexdigest()})
    evidence.digest(Path(__file__))
    output = {
        "version": "baseline_full_analysis_v1", "scope": "one complete uniformly regenerated 625-example internal validation baseline",
        "generation_run_id": generation["run_id"], "evaluation_run_id": evaluation["run_id"],
        "prediction_path": generation["prediction_path"], "prediction_sha256": generation["prediction_sha256"],
        "per_example_path": evaluation["per_example_path"], "source_sha256": evidence.hashes,
        "all_625_ids_databases_sql_and_prediction_bytes_verified": True,
        "complete_validation_generated": generation["complete_validation_generated"],
        "generation_failure_or_skip_count": generation["generation_failure_or_skip_count"],
        "protected_pilot_sha256_verified": generation["protected_pilot_sha256"],
        "pilot_evidence_hash_policy": generation["pilot_evidence_hash_policy"],
        "scoring_protocol": eval_inputs["protocol"], "evaluation": evaluation["overall"],
        "overall": analyze_group(sorted(predictions), predictions, scores),
        "by_hardness": {level: analyze_group([i for i in sorted(scores) if scores[i]["hardness"] == level],
                                             predictions, scores) for level in LEVELS},
        "per_example_analysis": details,
        "pilot_overlap_comparison": {
            "count": 32, "policy": "report all 32 old/new comparisons; never replace or select predictions by result",
            "old_evaluation": pilot_eval["overall"],
            "new_evaluation_on_same_ids": aggregate([scores[i] for i in sorted(pilot_predictions)]),
            "changed_sql_source_indices": [r["source_index"] for r in overlap if not r["sql_identical"]],
            "changed_ex_result_source_indices": [r["source_index"] for r in overlap if r["pilot_matched"] != r["full_matched"]],
            "details": overlap,
        },
        "cost": {
            "pilot_projected_total_seconds": projected, "pilot_projected_total_minutes": projected / 60,
            "full_generation_seconds_total": generation["generation_seconds_total"],
            "full_generation_loop_wall_seconds": generation["generation_loop_wall_seconds"],
            "full_load_seconds": generation["load_seconds"], "full_generation_end_to_end_seconds": generation["end_to_end_seconds"],
            "full_scoring_wall_seconds": evaluation["wall_seconds"], "full_scoring_pipeline_seconds": actual_scoring,
            "actual_generation_plus_scoring_seconds": actual, "actual_generation_plus_scoring_minutes": actual / 60,
            "actual_minus_pilot_projection_seconds": actual - projected,
            "actual_over_pilot_projection": actual / projected,
            "note": "Measured stage durations exclude gaps between commands and Python process startup; scorer pipeline timing includes validation/hash checks when supplied. Pilot projection included scorer wall only. No claim of a whole-session stopwatch.",
        },
        "memory": {k: v for k, v in generation.items() if "mib" in k or k in {
            "nvidia_smi_scope", "nvidia_smi_sample_count", "nvidia_smi_sampling_error_count", "gpu_name"}},
        "model_identity": {k: generation[k] for k in ("model_config", "model_weight_sha256", "quantization",
                                                     "is_loaded_in_4bit", "project_fine_tuning_applied")},
        "interpretation": [
            "All rates retain 625 examples, including generation failures, SQL errors and timeouts.",
            "execution_failed means the saved executable flag is false; its error bucket distinguishes SQL invalidity from query timeout, rejection or infrastructure failure. execution_unknown means the flag is null.",
            "First SQL error buckets describe the observed error only; token-limit and generation-time flags are independent and do not prove root cause.",
            "Non-EOS sub-budget generation is time_or_other_stop_unresolved unless an instrumented stop reason explicitly records time expiry.",
            "Both-empty EX matches remain counted but do not prove semantic equivalence on other database contents.",
            "This is original-database EX, not complete Spider test-suite accuracy; controls are evaluator checks rather than model scores.",
            "Hashes link locally saved evidence and detect later changes; they are not an external authenticity guarantee.",
        ],
        "training_performed": False, "final_examples_read": False,
        "model_generation_performed_by_analysis": False, "sql_executed_by_analysis": False,
        "pilot_predictions_reused_in_full_run": False,
    }
    destination = ROOT / "reports/baseline_full_analysis.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "prediction_count": 625, "evaluation": evaluation["overall"],
                      "actual_minutes": actual / 60, "pilot_projected_minutes": projected / 60}, ensure_ascii=False, indent=2))


def self_test():
    """No artifact, model or SQL reads; exercise classification and ID guards."""
    row = {"status": "prediction_error", "prediction_executable": False, "matched": None,
           "prediction": {"status": "sql_error", "error": "no such column: xyz"}}
    assert first_error_category(row) == "missing_column"
    assert execution_outcome(row) == "execution_failed"
    assert first_error_category({**row, "prediction": {"status": "sql_error", "error": "incomplete input"}}) == "incomplete_input"
    unknown = {**row, "prediction_executable": None, "prediction": None, "status": "parent_timeout"}
    assert execution_outcome(unknown) == "execution_unknown"
    assert first_error_category(unknown) == "execution_unknown_parent_timeout"
    assert first_error_category({**row, "prediction": {"status": "query_timeout", "error": "interrupted"}}) == "query_timeout"
    assert execution_outcome({**row, "prediction_executable": True, "matched": False}) == "executable_mismatch"
    assert execution_outcome({**row, "prediction_executable": True}) == "executable_comparison_unresolved"
    generated = {"generation_ok": True, "ended_with_eos": False, "reached_token_limit": False,
                 "stop_reason": "time_or_other_stopping_criterion"}
    assert stop_category(generated) == "time_or_other_stop_unresolved"
    assert stop_category({**generated, "reached_token_limit": True}) == "token_limit"
    assert stop_category({**generated, "ended_with_eos": True}) == "eos"
    assert stop_category({**generated, "generation_ok": False}) == "generation_failed_or_skipped"
    for bad in ([{"source_index": 1, "db_id": "x"}] * 2,
                [{"source_index": True, "db_id": "x"}, {"source_index": 2, "db_id": "x"}]):
        try:
            index_rows(bad, 2, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid/repeated ID was accepted")
    print("Synthetic checks passed; no model, SQL or saved result was read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test() if args.self_test else main()
