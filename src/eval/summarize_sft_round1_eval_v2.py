"""Read-only paired comparison of saved baseline and final SFT validation results.

No model imports, generation, training, SQLite connections, or final-set reads.
The analysis accepts only complete, once-generated 625-question results and
checks byte hashes, prompts, input identities and identical scoring protocols.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from summarize_baseline_full import (
    ROOT, LEVELS, POPULATION, Evidence, aggregate, analyze_group,
    execution_outcome, first_error_category, index_rows, load_evaluation,
    load_generation, nonnegative, project_path, require, stop_category,
)

COUNT = 625
FROZEN_SHARED = (
    "configs/model.yaml", "configs/prompt.yaml",
    "data/processed/spider_inputs_v1/validation_inputs.jsonl",
    "models/qwen2.5-coder-1.5b-instruct/tokenizer.json",
    "models/qwen2.5-coder-1.5b-instruct/tokenizer_config.json",
    "models/qwen2.5-coder-1.5b-instruct/vocab.json",
    "models/qwen2.5-coder-1.5b-instruct/merges.txt",
    "models/qwen2.5-coder-1.5b-instruct/config.json",
    "models/qwen2.5-coder-1.5b-instruct/generation_config.json",
)


def load_sft_generation(evidence, report_path):
    report = evidence.load(report_path)
    require(report.get("project_fine_tuning_applied") is True,
            "SFT report must identify the applied adapter")
    for key in ("final_examples_read", "reference_answers_loaded_for_inference",
                "validation_reference_files_read", "prediction_selection_performed"):
        require(report.get(key) is False, f"Invalid inference scope: {key}")
    require(report.get("prediction_count") == report.get("generated_count") == COUNT,
            "Only a complete successful 625-question generation can be compared")
    require(report.get("complete_validation_records") is True and
            report.get("complete_validation_generated") is True, "Incomplete SFT generation")
    run_dir = project_path(report["run_dir"])
    prediction_path = project_path(report["prediction_path"])
    require(prediction_path.parent == run_dir, "Prediction file escaped its run directory")
    require(evidence.load(run_dir / "summary.json") == report,
            "Published SFT generation report differs from saved summary")
    require(evidence.digest(prediction_path) == report["prediction_sha256"],
            "SFT prediction hash mismatch")
    evidence.verify(report["source_sha256"])
    rows = evidence.load(prediction_path, lines=True)
    predictions = index_rows(rows, COUNT, "SFT generation")
    for ordinal, row in enumerate(rows, 1):
        require(row.get("generation_ok") is True and row.get("attempt_number") == 1,
                "SFT prediction was not generated exactly once")
        require(row.get("ordinal") == ordinal, "Unexpected SFT prediction order")
        require(isinstance(row.get("predicted_sql"), str), "Missing SFT SQL")
        require(type(row.get("ended_with_eos")) is bool and
                type(row.get("reached_token_limit")) is bool, "Invalid generation stop flags")
        require(type(row.get("generated_token_count")) is int and
                0 <= row["generated_token_count"] <= 256, "Invalid generated token count")
        require(type(row.get("input_token_count")) is int and row["input_token_count"] > 0,
                "Missing input token count")
        if "new_token_ids" in row:
            require(len(row["new_token_ids"]) == row["generated_token_count"], "Token IDs/count differ")
        nonnegative(row["generation_seconds"], "per-question generation seconds")
    require(sum(r["generated_token_count"] for r in rows) == report["generated_tokens_total"],
            "SFT generated token total differs")
    require(math.isclose(sum(r["generation_seconds"] for r in rows),
                         report["generation_seconds_total"], rel_tol=1e-8, abs_tol=1e-6),
            "SFT generation duration total differs")
    for key in ("load_seconds", "end_to_end_seconds", "generation_loop_wall_seconds"):
        nonnegative(report[key], key)
    return report, predictions


def transition(before, after):
    """False and unknown both remain non-success in the fixed metric denominator."""
    if before is True:
        return "both_correct" if after is True else "regressed"
    return "improved" if after is True else "neither_correct"


def compare_group(indices, before, after, before_predictions, after_predictions):
    baseline = analyze_group(indices, before_predictions, before)
    sft = analyze_group(indices, after_predictions, after)
    metrics = ("execution_accuracy", "prediction_executable_rate")
    ex_transitions = Counter(transition(before[i]["matched"], after[i]["matched"]) for i in indices)
    exec_transitions = Counter(transition(before[i]["prediction_executable"],
                                         after[i]["prediction_executable"]) for i in indices)
    correct_delta = sft["evaluation"]["execution_correct"] - baseline["evaluation"]["execution_correct"]
    executable_delta = sft["evaluation"]["executable_predictions"] - baseline["evaluation"]["executable_predictions"]
    require(correct_delta == ex_transitions["improved"] - ex_transitions["regressed"], "EX pairing identity failed")
    require(executable_delta == exec_transitions["improved"] - exec_transitions["regressed"], "Executable pairing identity failed")
    return {
        "count": len(indices), "baseline": baseline, "sft": sft,
        "delta_percentage_points": {key: 100 * (sft["evaluation"][key] - baseline["evaluation"][key])
            if indices else None for key in metrics},
        "delta_correct_count": correct_delta, "delta_executable_count": executable_delta,
        "ex_transitions": {key: ex_transitions[key] for key in
                           ("both_correct", "improved", "regressed", "neither_correct")},
        "executability_transitions": {key: exec_transitions[key] for key in
                           ("both_correct", "improved", "regressed", "neither_correct")},
        "executability_transition_note": "correct means executable in this second transition table",
    }


def side_details(score, prediction):
    executed = score.get("prediction") or {}
    gold = score.get("gold") or {}
    return {
        "sql": prediction["predicted_sql"], "ex_match": score["matched"],
        "executable": score["prediction_executable"], "outcome": execution_outcome(score),
        "evaluation_status": score["status"], "first_error_category": first_error_category(score),
        "first_execution_error": executed.get("error"),
        "execution_status": executed.get("status"),
        "result_rows": executed.get("row_count"), "result_columns": executed.get("column_count"),
        "both_results_empty": gold.get("empty_result") is True and executed.get("empty_result") is True,
        "generation_seconds": prediction["generation_seconds"],
        "generated_token_count": prediction["generated_token_count"],
        "generation_stop_category": stop_category(prediction),
    }


def paired_rows(indices, before, after, before_predictions, after_predictions):
    pairs = []
    for index in indices:
        old, new = before[index], after[index]
        for key in ("source_index", "db_id", "hardness", "question", "gold_sql", "order_matters"):
            require(old.get(key) == new.get(key), f"Paired metadata differs: {index}/{key}")
        pairs.append({"source_index": index, "db_id": new["db_id"], "hardness": new["hardness"],
            "question": new["question"], "gold_sql": new["gold_sql"], "order_matters": new["order_matters"],
            "ex_transition": transition(old["matched"], new["matched"]),
            "executable_transition": transition(old["prediction_executable"], new["prediction_executable"]),
            "sql_identical": before_predictions[index]["predicted_sql"] == after_predictions[index]["predicted_sql"],
            "input_token_count": after_predictions[index]["input_token_count"],
            "baseline": side_details(old, before_predictions[index]),
            "sft": side_details(new, after_predictions[index])})
    return pairs


def choose_cases(pairs, category, limit=3):
    """Deterministic examples for explanation, never for selecting predictions."""
    candidates = sorted((row for row in pairs if row["ex_transition"] == category),
                        key=lambda row: (row["baseline"]["both_results_empty"] or row["sft"]["both_results_empty"],
                                         row["source_index"]))
    chosen, seen_databases = [], set()
    for row in candidates:
        if row["db_id"] not in seen_databases:
            chosen.append(row); seen_databases.add(row["db_id"])
            if len(chosen) == limit:
                break
    for row in candidates:
        if len(chosen) >= limit:
            break
        if row not in chosen:
            chosen.append(row)
    return chosen


def compare_cost(generation, evaluation):
    scoring = evaluation.get("evaluation_pipeline_wall_seconds", evaluation["wall_seconds"])
    nonnegative(scoring, "scoring seconds")
    memory_keys = {"nvidia_smi_scope", "nvidia_smi_sample_count", "nvidia_smi_sampling_error_count",
                   "gpu_name", "pytorch_memory_scope", "pdh_scope", "pdh_valid_samples"}
    return {
        "model_load_seconds": generation["load_seconds"],
        "generation_seconds_total": generation["generation_seconds_total"],
        "generation_loop_wall_seconds": generation["generation_loop_wall_seconds"],
        "generation_end_to_end_seconds": generation["end_to_end_seconds"],
        "scoring_seconds": scoring,
        "generation_plus_scoring_seconds": generation["end_to_end_seconds"] + scoring,
        "memory": {key: value for key, value in generation.items() if "mib" in key or key in memory_keys},
    }


def main(args):
    started = time.perf_counter()
    destination = project_path(args.output)
    require(not destination.exists(), "Analysis report already exists; refusing to overwrite")
    evidence = Evidence()
    baseline, base_predictions = load_generation(evidence, args.baseline_generation_report, COUNT)
    base_eval, base_inputs, base_scores = load_evaluation(
        evidence, args.baseline_evaluation_report, baseline, base_predictions, COUNT)
    sft, sft_predictions = load_sft_generation(evidence, args.generation_report)
    sft_eval, sft_inputs, sft_scores = load_evaluation(
        evidence, args.evaluation_report, sft, sft_predictions, COUNT)
    for key in ("model_config", "model_weight_sha256", "quantization", "generation_config"):
        require(baseline[key] == sft[key], f"Frozen inference setting differs: {key}")
    require(sft.get("is_loaded_in_4bit") is True, "SFT generation was not loaded in 4-bit")
    for key, value in baseline["packages"].items():
        require(sft["packages"].get(key) == value, f"Inference package version differs: {key}")
    for path in FROZEN_SHARED:
        require(baseline["source_sha256"].get(path) is not None and
                baseline["source_sha256"][path] == sft["source_sha256"].get(path),
                f"Frozen prompt/tokenizer/input hash differs: {path}")
    cfg = sft["generation_config"]
    require(cfg["max_new_tokens"] == 256 and cfg["do_sample"] is False and
            cfg["num_beams"] == cfg["num_return_sequences"] == 1, "Not one greedy 256-token generation")
    require(base_inputs["protocol"] == sft_inputs["protocol"], "EX protocol differs between baseline and SFT")
    require(base_eval["metric_limitations"] == sft_eval["metric_limitations"], "Metric limitations differ")
    require(set(base_scores) == set(sft_scores) == set(base_predictions) == set(sft_predictions),
            "Paired sample populations differ")
    require(dict(Counter(row["hardness"] for row in sft_scores.values())) == POPULATION,
            "SFT difficulty population differs")
    base_prompts = evidence.load(project_path(baseline["run_dir"]) / "prompts.jsonl", lines=True)
    sft_prompts = evidence.load(project_path(sft["run_dir"]) / "prompts.jsonl", lines=True)
    require(base_prompts == sft_prompts and len(sft_prompts) == COUNT, "Rendered prompts or input token counts differ")
    prompt_ids = [row["source_index"] for row in sft_prompts]
    require(list(sft_predictions) == list(base_predictions) == prompt_ids,
            "The generation order differs from frozen prompts")
    for row in sft_prompts:
        index = row["source_index"]
        require(row["db_id"] == sft_predictions[index]["db_id"] and
                row["input_token_count"] == sft_predictions[index]["input_token_count"] ==
                base_predictions[index]["input_token_count"], "Prediction/prompt input identity differs")
    pairs = paired_rows(sorted(sft_scores), base_scores, sft_scores, base_predictions, sft_predictions)
    overall = compare_group(sorted(sft_scores), base_scores, sft_scores, base_predictions, sft_predictions)
    by_hardness = {level: compare_group([i for i in sorted(sft_scores) if sft_scores[i]["hardness"] == level],
        base_scores, sft_scores, base_predictions, sft_predictions) for level in LEVELS}
    adapter_provenance = {key: value for key, value in sft.items() if "adapter" in key or "checkpoint" in key}
    evidence.digest(Path(__file__))
    evidence.digest(ROOT / "src/eval/summarize_baseline_full.py")
    result = {
        "version": "sft_round1_eval_v2_analysis", "status": "pass",
        "scope": "paired comparison of all 625 internal validation examples using the final 796-update adapter",
        "baseline_generation_run_id": baseline["run_id"], "sft_generation_run_id": sft["run_id"],
        "baseline_evaluation_run_id": base_eval["run_id"], "sft_evaluation_run_id": sft_eval["run_id"],
        "baseline_prediction_path": baseline["prediction_path"], "sft_prediction_path": sft["prediction_path"],
        "baseline_prediction_sha256": baseline["prediction_sha256"], "sft_prediction_sha256": sft["prediction_sha256"],
        "scoring_protocol": sft_inputs["protocol"], "adapter_provenance": adapter_provenance,
        "checks": {"same_625_ids_and_metadata": True, "same_model_quantization_and_generation_config": True,
            "same_rendered_prompts_and_token_counts": True, "same_tokenizer_input_hashes": True,
            "one_attempt_per_sft_question": True, "same_ex_protocol": True,
            "paired_transition_counts_equal_metric_delta": True, "evidence_hashes_verified": True},
        "overall": overall, "by_hardness": by_hardness,
        "improved_source_indices": [p["source_index"] for p in pairs if p["ex_transition"] == "improved"],
        "regressed_source_indices": [p["source_index"] for p in pairs if p["ex_transition"] == "regressed"],
        "paired_examples": pairs,
        "illustrative_cases": {category: choose_cases(pairs, category) for category in ("improved", "regressed")},
        "case_selection_policy": "Deterministic source-index order, prefer non-empty results and different databases; all 625 remain in metrics and the paired list. Cases do not select or replace predictions.",
        "cost": {"baseline": compare_cost(baseline, base_eval), "sft": compare_cost(sft, sft_eval),
            "timing_limit": "Saved stage timings exclude gaps between commands; environments/times differ, so this is not a controlled speed benchmark."},
        "interpretation": [
            "EX is on one original SQLite database, not full Spider test-suite accuracy or proof of semantic equivalence.",
            "Both-empty matches remain counted according to the frozen comparator and are flagged in paired cases.",
            "Unknown/failed execution and non-matches remain in the denominator of 625; no failed question is dropped.",
            "First execution errors and generation stop categories are observations, not proven semantic root causes.",
            "This compares one final checkpoint to the frozen base; no checkpoint selection or new training was performed.",
            "The internal validation set is used for development; this is not a final held-out score.",
        ],
        "source_sha256": evidence.hashes, "analysis_seconds": time.perf_counter() - started,
        "model_generation_performed_by_analysis": False, "sql_executed_by_analysis": False,
        "training_performed": False, "final_examples_read": False,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "report": str(destination),
        "baseline": overall["baseline"]["evaluation"], "sft": overall["sft"]["evaluation"],
        "delta_percentage_points": overall["delta_percentage_points"],
        "ex_transitions": overall["ex_transitions"]}, ensure_ascii=False))


def self_test():
    require(transition(True, True) == "both_correct", "Matched pair classification")
    require(transition(None, True) == "improved", "Execution failure must remain a non-match")
    require(transition(True, False) == "regressed", "Regression classification")
    require(transition(False, None) == "neither_correct", "Unknown comparison classification")
    score = {"prediction_executable": True, "matched": True, "status": "scored",
             "gold": {"status": "ok", "empty_result": False},
             "prediction": {"status": "ok", "empty_result": False}}
    pred = {"generation_seconds": 1.0, "generated_token_count": 2, "input_token_count": 5,
            "generation_ok": True, "ended_with_eos": True, "reached_token_limit": False,
            "generation_status": "generated"}
    wrong = {**score, "matched": False}
    unknown = {**score, "matched": None, "prediction_executable": None, "prediction": None,
               "status": "parent_timeout"}
    before = {1: score, 2: wrong, 3: score, 4: unknown}
    after = {1: score, 2: score, 3: wrong, 4: score}
    preds = {i: pred for i in before}
    result = compare_group(list(before), before, after, preds, preds)
    require(result["ex_transitions"] == {"both_correct": 1, "improved": 2, "regressed": 1, "neither_correct": 0}, "Paired transition counts")
    require(result["delta_correct_count"] == 1 and result["delta_percentage_points"]["execution_accuracy"] == 25.0,
            "Paired EX delta must use the full denominator including unknown results")
    require(result["delta_executable_count"] == 1, "Executability improvement count")
    require(choose_cases([], "improved") == [], "No improvement is valid")
    print("SFT paired analysis synthetic checks passed; no model, SQLite or saved data read.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-report", default="reports/sft_round1_eval_v2_generation.json")
    parser.add_argument("--evaluation-report", default="reports/sft_round1_eval_v2_eval.json")
    parser.add_argument("--baseline-generation-report", default="reports/baseline_full_generation.json")
    parser.add_argument("--baseline-evaluation-report", default="reports/baseline_full_eval.json")
    parser.add_argument("--output", default="reports/sft_round1_eval_v2_analysis.json")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    self_test() if arguments.self_test else main(arguments)
