"""Link the completed 32 predictions to their scores and estimate future cost.

Reads saved artifacts only: no model, SQL execution, training or final split.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    generation_path = ROOT / "reports/baseline_pilot_generation.json"
    evaluation_path = ROOT / "reports/baseline_pilot_eval.json"
    manifest_path = ROOT / "reports/baseline_pilot_manifest.json"
    gen, evaluation, manifest = map(read, (generation_path, evaluation_path, manifest_path))
    prediction_path = ROOT / gen["prediction_path"]
    actual_digest = digest(prediction_path)
    if actual_digest != gen["prediction_sha256"] or actual_digest != evaluation["prediction_file_sha256"]:
        raise ValueError("Generation and evaluation do not refer to the same prediction bytes")
    if digest(manifest_path) != gen["pilot_manifest_sha256"] or digest(manifest_path) != evaluation["pilot_manifest_sha256"]:
        raise ValueError("Pilot manifest version mismatch")
    predictions = rows(prediction_path)
    scored_path = ROOT / evaluation["per_example_path"]
    scored = rows(scored_path)
    ids = set(manifest["selected_source_indices"])
    if len(predictions) != 32 or len(scored) != 32 or {r["source_index"] for r in predictions} != ids or {r["source_index"] for r in scored} != ids:
        raise ValueError("This analysis only accepts the frozen complete 32 examples")
    prediction_by_id = {r["source_index"]: r for r in predictions}
    for result in scored:
        pred = prediction_by_id[result["source_index"]]
        if result["predicted_sql"] != pred["predicted_sql"] or result["db_id"] != pred["db_id"]:
            raise ValueError("Per-example score does not match the saved prediction")
    if not all(r["generation_ok"] for r in predictions):
        raise ValueError("Cannot use successful-generation timing assumptions when generation failed")
    hardness = {r["source_index"]: r["hardness"] for r in manifest["selected_examples"]}
    groups = {}
    for level in ("easy", "medium", "hard", "extra"):
        group = [r for r in predictions if hardness[r["source_index"]] == level]
        average = statistics.mean(r["generation_seconds"] for r in group)
        population = manifest["hardness_population_counts"][level]
        groups[level] = {"pilot_count": len(group), "full_count": population,
                         "mean_generation_seconds": average,
                         "projected_generation_seconds": average * population}
    projected_generation = sum(v["projected_generation_seconds"] for v in groups.values())
    fixed_overhead = gen["end_to_end_seconds"] - gen["generation_loop_wall_seconds"]
    loop_overhead = (gen["generation_loop_wall_seconds"] - gen["generation_seconds_total"]) * 625 / 32
    projected_evaluation = evaluation["wall_seconds"] * 625 / 32
    projected_total = projected_generation + fixed_overhead + loop_overhead + projected_evaluation
    outcomes = Counter("ex_match" if r["matched"] is True else "executable_mismatch" if r["prediction_executable"] is True else "execution_failure_or_unknown" for r in scored)
    error_groups = {}
    for r in scored:
        if r["prediction_executable"] is True:
            continue
        error = (r.get("prediction") or {}).get("error", "")
        category = "missing_column" if error.startswith("no such column:") else "token_limit_with_incomplete_output" if prediction_by_id[r["source_index"]]["reached_token_limit"] else "other_sql_error"
        error_groups.setdefault(category, []).append(r["source_index"])
    output = {
        "scope": "post-hoc analysis of saved 32-question pilot; no additional generation or evaluation",
        "generation_run_id": gen["run_id"], "evaluation_run_id": evaluation["run_id"],
        "source_sha256": {str(p.relative_to(ROOT).as_posix()): digest(p) for p in (generation_path, evaluation_path, manifest_path, prediction_path, scored_path)},
        "prediction_evaluation_link_verified": True, "outcome_counts": dict(outcomes),
        "execution_failure_source_indices": {k: sorted(v) for k, v in error_groups.items()},
        "token_limit_source_indices": [r["source_index"] for r in predictions if r["reached_token_limit"]],
        "both_empty_match_source_indices": [r["source_index"] for r in scored if r["matched"] is True and r["gold"]["empty_result"] and r["prediction"]["empty_result"]],
        "full_625_estimate": {
            "actually_run": False, "formula": "sum(population_count_h * pilot_mean_generation_seconds_h) + one startup + scaled loop overhead + scaled evaluation wall",
            "by_hardness": groups, "weighted_generation_seconds": projected_generation,
            "unweighted_generation_seconds": gen["generation_seconds_total"] / 32 * 625,
            "one_time_startup_shutdown_seconds": fixed_overhead,
            "projected_non_generation_loop_seconds": loop_overhead,
            "projected_cpu_evaluation_seconds": projected_evaluation,
            "projected_total_seconds": projected_total, "projected_total_minutes": projected_total / 60,
            "planning_minutes": [25, 40], "planning_interval_is_statistical_confidence_interval": False,
            "assumptions": ["same model, prompt, NF4, greedy, batch=1 and output budget", "load model once and generate 625 serially", "similar hardware power, temperature and background activity", "pilot may not represent output lengths or every database; token-limited outputs are retained", "CPU projection includes startup costs repeatedly and is approximate", "no dedicated warmup; first query cold cost remains in estimate", "32-only runner and 600-second pilot guard must not be reused as the full-run controller"]
        },
        "memory": {k: gen[k] for k in ("load_peak_allocated_mib", "load_peak_reserved_mib", "generation_peak_allocated_mib", "generation_peak_reserved_mib", "nvidia_smi_before_load_used_mib", "nvidia_smi_sampled_peak_used_mib", "nvidia_smi_sample_count", "nvidia_smi_sampling_error_count", "nvidia_smi_scope")},
        "final_examples_read": False, "training_performed": False,
    }
    destination = ROOT / "reports/baseline_pilot_analysis.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"outcome_counts": dict(outcomes), "estimated_full_minutes": projected_total / 60,
                      "report": str(destination), "full_625_actually_run": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
