"""Read-only final audit: base generation v3, SFT generation v4, decode scoring v4.

The original strict-scoring audit is retained untouched. No models or SQL are
executed; only existing evidence JSON/JSONL and file hashes are inspected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.diagnostics import audit_final_completion_v1 as core

Evidence, require, same = core.Evidence, core.require, core.same
aggregate, outcome_counts, group = core.aggregate, core.outcome_counts, core.group
COUNT, PREFIX_COUNT, LEVELS = core.COUNT, core.PREFIX_COUNT, core.LEVELS
CONFIG = ROOT / "configs/final_decode_v4.json"
RUN = ROOT / "outputs/final_holdout_decode_v4"
GENERATION = {"base": ROOT / "outputs/final_holdout_v3/base", "sft": ROOT / "outputs/final_holdout_v4/sft"}
DESTINATION = ROOT / "outputs/checks/final_completion_v2"
DECODER = "utf8_surrogateescape_text_only_v1"


def freeze_checks(evidence):
    cfg = evidence.load(CONFIG)
    require(cfg["run_root"] == RUN.relative_to(ROOT).as_posix()
            and cfg["generation_roots"] == {arm: path.relative_to(ROOT).as_posix() for arm, path in GENERATION.items()},
            "Unexpected generation/scoring roots")
    require(cfg["decoder_revision"] == DECODER and cfg["training"] is False
            and cfg["regenerate_predictions"] is False and cfg["reuse_old_scores"] is False
            and cfg["automatic_retries"] == 0, "Unexpected scoring revision scope")
    authorization = evidence.load(cfg["authorization_record"])
    require(authorization["authorized"] is True and authorization["authorization_source"] == "user"
            and bool(authorization["user_message"].strip()), "Explicit user authorization absent")
    scope = authorization["scope"]
    require(scope["decoder"] == DECODER and scope["models"] == ["base", "sft"]
            and scope["examples_per_model"] == COUNT
            and scope["allow_training"] is False and scope["allow_regeneration"] is False,
            "User authorization scope differs")
    frozen = evidence.load(cfg["frozen_report"])
    checked = evidence.load(cfg["preflight_report"])
    require(frozen["status"] == "frozen_after_explicit_decoder_authorization" and frozen["config"] == cfg,
            "Scoring freeze mismatch")
    require(checked["status"] == "pass" and checked["config"] == cfg
            and checked["frozen_report_sha256"] == evidence.digest(cfg["frozen_report"]), "Scoring preflight mismatch")
    require(frozen["equivalent_to_original_strict_failure_policy"] is False
            and frozen["old_failed_strict_scores_reused"] is False,
            "Scoring revision is concealed or mixes old scores")
    evidence.verify(frozen["source_sha256"], "authorized decode freeze")
    require(frozen["authorization_sha256"] == evidence.digest(cfg["authorization_record"]), "Authorization hash differs")
    for name in (cfg["decoder_cpu_evidence"], cfg["worker_cpu_evidence"]):
        require(evidence.load(name)["status"] == "pass", "Decoder CPU evidence did not pass")
    old = evidence.load("reports/final_holdout_v1_freeze.json")
    evidence.verify(old["source_sha256"], "original final freeze")
    parent = evidence.load("reports/final_holdout_v1_preflight.json")
    require(parent["status"] == "pass", "Original final preparation incomplete")
    evidence.verify(parent["output_sha256"], "prepared final inputs/references")
    input_manifest = "data/processed/spider_final_holdout_v1/manifest.json"
    require(evidence.digest(input_manifest) == parent["input_manifest_sha256"], "Prepared manifest changed")
    manifest = evidence.load(input_manifest)
    require(manifest["recipe_freeze_sha256"] == evidence.digest("reports/final_holdout_v1_freeze.json")
            and manifest["training_performed"] is False and manifest["final_used_for_tuning"] is False,
            "Original prepared-data scope/binding differs")
    generation_configs, generation_preflights = {}, {}
    counts = {"original_final": len(old["source_sha256"]), "authorized_decode": len(frozen["source_sha256"])}
    for arm in ("base", "sft"):
        generation_cfg = evidence.load(cfg["generation_configs"][arm])
        generation_freeze = evidence.load(generation_cfg["frozen_report"])
        generation_checked = evidence.load(generation_cfg["preflight_report"])
        require(generation_freeze["config"] == generation_cfg and generation_checked["config"] == generation_cfg
                and generation_checked["status"] == "pass", "Generation freeze/preflight incomplete")
        require(generation_checked["frozen_report_sha256"] == evidence.digest(generation_cfg["frozen_report"]),
                "Generation source freeze binding differs")
        require(generation_cfg["training"] is False and generation_cfg["automatic_retries"] == 0
                and generation_cfg["truncate"] is False, "Generation scope changed")
        evidence.verify(generation_freeze["source_sha256"], f"{arm} generation freeze")
        for key in ("source_indices", "model_config", "weight_path", "weight_sha256", "adapter_fingerprint",
                    "input_tokens_max", "input_manifest_sha256", "output_sha256", "generation_config"):
            require(generation_checked[key] == parent[key], f"Frozen inference/data changed: {arm}/{key}")
        generation_configs[arm], generation_preflights[arm] = generation_cfg, generation_checked
        counts[arm + "_generation"] = len(generation_freeze["source_sha256"])
    return cfg, parent, generation_configs, generation_preflights, counts


def decode_counts(rows):
    answer = {}
    for side in ("gold", "prediction"):
        values = [row[side] for row in rows if isinstance(row.get(side), dict)]
        require(all(value.get("decode_mode") == "utf-8/surrogateescape"
                    and type(value.get("invalid_text_values")) is int
                    and value["invalid_text_values"] >= 0 for value in values), "Invalid new decoder metadata")
        answer[side] = {"queries_with_invalid_text": sum(value["invalid_text_values"] > 0 for value in values),
                        "decoded_invalid_text_values": sum(value["invalid_text_values"] for value in values)}
    answer["scope"] = "Only TEXT values returned through this query's text_factory; not a whole-database corruption count"
    return answer


def audit_scores(evidence, arm, predictions, cfg):
    directory = RUN / f"scores_{arm}"
    summary = evidence.load(directory / "summary.json")
    require(summary["status"] == "pass" and summary["complete_final_scored"] is True and summary["arm"] == arm,
            f"Scoring incomplete: {arm}")
    require(summary["training_performed"] is False and summary["model_generation_performed"] is False
            and summary["automatic_retries"] == 0 and summary["source_files_unchanged"] is True
            and summary["complete_test_suite_accuracy"] is False, "Scoring scope/limitations changed")
    require(summary["prediction_sha256"] == evidence.digest(GENERATION[arm] / "predictions.jsonl"), "Scoring bound to wrong predictions")
    rows = evidence.load(directory / "per_example.jsonl", lines=True)
    require(len(rows) == COUNT and sorted(row["source_index"] for row in rows) == list(range(COUNT)),
            "New scoring population incomplete or duplicated")
    scores = {row["source_index"]: row for row in rows}
    for index, prediction in enumerate(predictions):
        row = scores[index]
        require(row["db_id"] == prediction["db_id"] and row["predicted_sql"] == prediction["predicted_sql"],
                f"Scoring changed predicted SQL: {arm}/{index}")
        require(row["hardness"] in LEVELS and row["gold"]["status"] == "ok", "Unexpected hardness/gold failure")
        require(row["status"] in {"scored", "prediction_error", "comparison_error"}, "Infrastructure failure in complete scoring")
        require(row["matched"] is None or type(row["matched"]) is bool, "Invalid EX boolean")
        require(type(row["prediction_executable"]) is bool
                and row["prediction_executable"] is (row["prediction"]["status"] == "ok"), "Executability inconsistent")
        if row["status"] == "scored":
            require(type(row["matched"]) is bool and row["prediction_executable"] is True, "Invalid scored record")
        else:
            require(row["matched"] is None, "Unscored response claims EX")
        if row["status"] == "comparison_error":
            require(row["error"] == "Comparator returned after its time budget", "Unexpected comparator exception")
    same(summary["overall"], aggregate(rows), f"{arm} overall")
    same(summary["by_hardness"], {level: aggregate([row for row in rows if row["hardness"] == level]) for level in LEVELS}, f"{arm} difficulty")
    same(summary["by_database"], {db: aggregate([row for row in rows if row["db_id"] == db]) for db in sorted({row["db_id"] for row in rows})}, f"{arm} database")
    same(summary["failure_types"], outcome_counts(rows), f"{arm} failure types")
    same(summary["text_decode"], decode_counts(rows), f"{arm} text decoding")
    dispatch = evidence.load(directory / "dispatch.json")
    require(dispatch == summary["dispatch"] and dispatch["failures"] == [] and dispatch["automatic_retries"] == 0
            and dispatch["attempted_count"] == dispatch["completed_count"] == COUNT
            and sorted(dispatch["attempted_source_indices"]) == list(range(COUNT)), "Scoring dispatch/replay mismatch")
    protocol = evidence.load(directory / "protocol.json")
    require(protocol["version"] == "spider_original_db_ex_lossless_text_v2"
            and summary["protocol_version"] == protocol["version"]
            and protocol["text_decoder"] == "utf-8/surrogateescape" and protocol["decoder_revision"] == DECODER
            and protocol["authorization_record"] == cfg["authorization_record"]
            and protocol["worker_source"] == cfg["worker_source"]
            and protocol["raw_sql_unchanged"] is True and protocol["scores_reexecuted_for_both_models"] is True,
            "Revised text protocol is not declared correctly")
    require(protocol["expected_examples"] == COUNT and protocol["split"] == "final", "Final protocol population changed")
    original = evidence.load("configs/evaluation.json")
    require(protocol["parent_version"] == original["version"], "Original protocol lineage missing")
    for key in ("metric", "comparator", "upstream_commit", "database_dir", "plug_value", "keep_distinct", "rewrite_sql",
                "query_timeout_seconds", "comparison_timeout_seconds", "hard_timeout_grace_seconds", "workers", "seed"):
        require(protocol[key] == original[key], f"Unauthorized non-decoding protocol change: {key}")
    require(evidence.digest(directory / Path(cfg["worker_source"]).name) == evidence.digest(cfg["worker_source"]),
            "Executed worker snapshot differs from frozen decoder")
    inputs = evidence.load(directory / "inputs.json")
    require(inputs["predictions_sha256"] == summary["prediction_sha256"]
            and inputs["source_index_order"] == list(range(COUNT)), "Scoring input binding differs")
    core.finite_nonnegative(summary["wall_seconds"], "scoring wall_seconds")
    return summary, scores


def audit(evidence):
    cfg, parent, generation_configs, generation_preflights, counts = freeze_checks(evidence)
    generations, predictions, costs, reports, scores = {}, {}, {}, {}, {}
    original_core_root = core.RUN
    try:
        for arm in ("base", "sft"):
            # Reuse the independent read-only v1 generation checks against each
            # explicitly selected generation directory, without changing files.
            core.RUN = GENERATION[arm].parent
            generations[arm], predictions[arm], costs[arm] = core.audit_generation(
                evidence, arm, generation_configs[arm], generation_preflights[arm])
            reports[arm], scores[arm] = audit_scores(evidence, arm, predictions[arm], cfg)
            costs[arm]["scoring_wall_seconds"] = reports[arm]["wall_seconds"]
            costs[arm]["generation_controller_windows_plus_scoring_seconds"] = (
                costs[arm]["both_generation_controller_windows_seconds"] + reports[arm]["wall_seconds"])
    finally:
        core.RUN = original_core_root
    for filename in ("prompts.jsonl", "generation_config.json"):
        require(evidence.payload(GENERATION["base"] / filename) == evidence.payload(GENERATION["sft"] / filename),
                f"Generation artifacts differ: {filename}")
    require(evidence.payload(GENERATION["base"] / "prompts.jsonl") == evidence.payload("outputs/final_holdout_v1/base/prompts.jsonl"),
            "Prompts differ from original prefix")
    require(evidence.load(RUN / "scores_base/protocol.json") == evidence.load(RUN / "scores_sft/protocol.json"),
            "Two arms did not use identical revised scoring protocol")
    for index in range(COUNT):
        for key in ("source_index", "db_id", "hardness", "question", "gold_sql", "order_matters"):
            require(scores["base"][index].get(key) == scores["sft"][index].get(key), f"Paired metadata changed: {index}/{key}")
    comparison = evidence.load(RUN / "comparison.json")
    require(comparison["status"] == "pass" and comparison["generation_roots"] == cfg["generation_roots"], "Comparison incomplete or wrong roots")
    for key in ("training_performed", "final_result_used_for_tuning", "prediction_replay_or_selection_performed", "predictions_regenerated_for_scoring"):
        require(comparison[key] is False, f"Unexpected comparison scope: {key}")
    require(comparison["automatic_retries"] == 0 and comparison["same_prompts_and_generation_config"] is True
            and comparison["same_ex_protocol"] is True and comparison["scoring_protocol_revised"] is True
            and comparison["decoder_revision"] == DECODER
            and comparison["authorization_record"] == cfg["authorization_record"], "Unacknowledged/unfair scoring revision")
    overall = group(list(range(COUNT)), scores)
    by_hardness = {level: group([index for index in range(COUNT) if scores["base"][index]["hardness"] == level], scores) for level in LEVELS}
    failure_types = {arm: outcome_counts(list(scores[arm].values())) for arm in ("base", "sft")}
    decoding = {arm: decode_counts(list(scores[arm].values())) for arm in ("base", "sft")}
    same(comparison["overall"], overall, "Paired overall")
    same(comparison["by_hardness"], by_hardness, "Paired difficulty")
    same(comparison["failure_types"], failure_types, "Paired failure types")
    same(comparison["text_decode"], decoding, "Paired decoding counts")
    improved = [index for index in range(COUNT) if scores["base"][index]["matched"] is not True and scores["sft"][index]["matched"] is True]
    regressed = [index for index in range(COUNT) if scores["base"][index]["matched"] is True and scores["sft"][index]["matched"] is not True]
    same(comparison["improved_source_indices"], improved, "Improved IDs")
    same(comparison["regressed_source_indices"], regressed, "Regressed IDs")
    require(comparison["baseline_prefix_reused_count"] == PREFIX_COUNT
            and comparison["baseline_new_generated_count"] == COUNT-PREFIX_COUNT
            and comparison["sft_new_generated_count"] == COUNT
            and comparison["baseline_prefix_sha256"] == generation_configs["base"]["prefix_sha256"], "Prediction lineage differs")
    evidence.verify(comparison["source_sha256"], "comparison evidence")
    require(reports["base"]["metric_limitations"] == reports["sft"]["metric_limitations"] == comparison["metric_limitations"],
            "Metric limitation declarations differ")
    old_directory = ROOT / cfg["preserved_failed_score_dir"]
    strict_summary = evidence.load(old_directory / "summary.json")
    strict_rows = evidence.load(old_directory / "per_example.jsonl", lines=True)
    require(strict_summary["status"] == "failed" and len(strict_rows) == 454
            and strict_summary["complete_final_scored"] is False, "Strict partial failure record not preserved")
    valid_old = [row for row in strict_rows if row["status"] in {"scored", "prediction_error"}]
    changed_old = [row["source_index"] for row in valid_old
        if (row["matched"], row["prediction_executable"]) != (
            scores["base"][row["source_index"]]["matched"], scores["base"][row["source_index"]]["prediction_executable"])]
    historical = {"strict_attempt_returned_records": len(strict_rows),
        "strict_attempt_valid_outcomes": len(valid_old),
        "strict_valid_outcomes_changed_in_revised_full_score": changed_old,
        "strict_attempt_wall_seconds": strict_summary["wall_seconds"],
        "old_partial_scores_reused": False,
        "policy_revision": "strict UTF-8 result TEXT decoding to reversible surrogateescape; BLOB remains bytes; comparison and queries unchanged"}
    evidence.verify(dict(evidence.hashes), "final audit evidence integrity")
    return {"checks": {
            "generation_and_score_roots_explicit": True, "original_and_all_new_frozen_hashes_verified": True,
            "original_616_prediction_bytes_retained": True, "canonical_1034_once_generated_per_model": True,
            "controllers_exited_successfully_and_jobs_empty": True, "same_prompts_and_effective_generation_config": True,
            "same_authorized_revised_scoring_protocol": True, "scores_bound_to_saved_predicted_SQL": True,
            "metrics_and_paired_cells_independently_recomputed": True, "no_training_or_final_tuning_declarations_verified": True,
            "no_model_or_SQL_execution_in_audit": True},
        "freeze_counts": counts,
        "report_consumables": {
            "scope": "1034 official-dev project holdout /20 databases; base vs preregistered first-round update796",
            "overall": overall, "by_hardness": by_hardness, "failure_types": failure_types,
            "text_decode": decoding, "costs": costs, "strict_scoring_history": historical,
            "combined_recorded_generation_controller_windows_and_scoring_seconds": sum(
                cost["generation_controller_windows_plus_scoring_seconds"] for cost in costs.values()),
            "combined_windows_including_failed_strict_scoring_seconds": sum(
                cost["generation_controller_windows_plus_scoring_seconds"] for cost in costs.values()) + historical["strict_attempt_wall_seconds"],
            "input_tokens_min": parent["input_tokens_min"], "input_tokens_max": parent["input_tokens_max"],
            "decoder_revision": DECODER, "scoring_protocol_revised": True,
            "authorization_record": cfg["authorization_record"], "metric_limitations": comparison["metric_limitations"],
            "final_tuning_performed": False,
            "claims_boundary": "Verifies recorded declarations and hashes; not proof of absent unrecorded external activity or statistical significance",
        }}


def main():
    started = time.perf_counter()
    DESTINATION.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    result = {"status": "failed", "audit_version": "final_completion_v2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_retries": 0, "models_loaded": False, "SQL_executed": False, "training_performed": False,
        "raw_final_examples_or_reference_contents_loaded": False}
    try:
        evidence.digest(Path(__file__)); evidence.digest(Path(core.__file__))
        result.update(audit(evidence)); result["status"] = "pass"
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
