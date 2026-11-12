"""Final holdout v2 EX with an unchanged 616-row baseline prefix and pinned scorer.

The official dev split is the project's held-out final set, not official test.
No model imports, generation, SQL repair, retries or training occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_spider as evaluation
from evaluate_sft_round1_v2 import infrastructure_problem
from summarize_baseline_full import first_error_category, execution_outcome
from src.eval.final_holdout_common_v1 import (
    CONFIG as V1_CONFIG, read_json, sha256, relative,
    verify_frozen as verify_v1_frozen, verify_prepared_binding,
)

CONFIG = ROOT / "configs/final_holdout_resume_v2.json"
V2_FREEZE = ROOT / "reports/final_holdout_resume_v2_freeze.json"
PREFIX_PATH = ROOT / "outputs/final_holdout_v1/base/predictions.jsonl"
PREFIX_COUNT = 616


def verify_frozen():
    """Bind repaired infrastructure to both the old recipe and new CPU preflight."""
    verify_v1_frozen()
    cfg = read_json(CONFIG)
    frozen = read_json(V2_FREEZE)
    hashes = frozen.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing v2 freeze hashes")
    for path in (CONFIG, Path(__file__), PREFIX_PATH):
        if relative(path) not in hashes:
            raise ValueError(f"v2 freeze does not bind required file: {relative(path)}")
    for filename, expected in hashes.items():
        evaluation.expect_hash(evaluation.project_path(filename), expected)
    checked = read_json(ROOT / cfg["preflight_report"])
    if checked.get("status") != "pass" or checked.get("config") != cfg:
        raise ValueError("Successful matching v2 CPU preflight required")
    if ("frozen_report_sha256" in checked
            and checked["frozen_report_sha256"] != sha256(V2_FREEZE)):
        raise ValueError("v2 preflight freeze fingerprint differs")
    return frozen

COUNT = 1034
LEVELS = ("easy", "medium", "hard", "extra")
SCORING_BUDGET_SECONDS = 300.0
CLEANUP_RESERVE_SECONDS = 20.0
METRIC_LIMITATIONS = [
    "One original SQLite database per question; not full official test-suite accuracy",
    "A matching execution result on one database does not prove SQL semantic equivalence",
    "Official comparator allows a single global column permutation",
    "Two empty results match even if result column counts differ",
    "Ordering follows substring 'order by' in gold.lower(); no custom float tolerance",
    "Official dev is this project's held-out final set, not the official test set",
    "Public pretraining exposure to Spider cannot be excluded",
]


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(evaluation.json_bytes(value))
        stream.flush()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_final_protocol():
    """Use frozen raw/processed references without filtering, editing or deduping."""
    verify_prepared_binding()
    verify_frozen()
    recipe = read_json(CONFIG)
    if COUNT != recipe['expected_examples'] or SCORING_BUDGET_SECONDS != recipe['scoring_budget_seconds']:
        raise ValueError('Final scorer count or budget differs from frozen recipe')
    original = read_json(ROOT / "configs/evaluation.json")
    if (original["plug_value"] is not False or original["keep_distinct"] is not True
            or original["rewrite_sql"] is not False or original["workers"] != 4
            or original["query_timeout_seconds"] != 2.0
            or original["comparison_timeout_seconds"] != 2.0
            or original["hard_timeout_grace_seconds"] != 3.0):
        raise ValueError("Frozen original EX protocol changed")
    provenance = read_json(ROOT / original["source_manifest"])
    if provenance["commit"] != original["upstream_commit"]:
        raise ValueError("Pinned comparator commit mismatch")
    for name, metadata in provenance["source_files"].items():
        evaluation.expect_hash(ROOT / provenance["local_dir"] / name, metadata["sha256"])
    for name, expected in provenance["dependency_versions"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen scorer dependency differs: {name}")
    folder = ROOT / "data/processed/spider_final_holdout_v1"
    references = read_lines(folder / "references.jsonl")
    inputs = read_lines(folder / "inputs.jsonl")
    sealed = read_json(ROOT / "data/processed/spider_v1/final_eval_manifest.json")
    if len(references) != COUNT or len(inputs) != COUNT:
        raise ValueError("The entire 1034-example final split is required")
    if [row["source_index"] for row in references] != list(range(COUNT)):
        raise ValueError("Final reference order/IDs changed")
    if [row["source_index"] for row in inputs] != list(range(COUNT)):
        raise ValueError("Final input order/IDs changed")
    expected_db_ids = set(sealed["db_ids"])
    if {row["db_id"] for row in references} != expected_db_ids or len(expected_db_ids) != 20:
        raise ValueError("Final database membership differs")
    split = read_json(ROOT / "data/processed/spider_v1/split_manifest.json")
    rows = []
    for reference, source in zip(references, inputs, strict=True):
        if (reference["source_file"] != "data/raw/spider/dev.json" or reference["project_split"] != "final"
                or reference["db_id"] != source["db_id"] or source["project_split"] != "final"
                or source["source_file"] != reference["source_file"]
                or not isinstance(reference["gold_sql"], str)
                or not isinstance(reference["sql"], dict)):
            raise ValueError("Final reference/input identity mismatch")
        rows.append({**reference, "query": reference["gold_sql"]})
    for db_id in sorted(expected_db_ids):
        if Path(db_id).name != db_id or db_id in (".", ".."):
            raise ValueError("Invalid database path component")
        path = ROOT / original["database_dir"] / db_id / f"{db_id}.sqlite"
        evaluation.expect_hash(path, split["source_sha256"][relative(path)])
    config = dict(original)
    config.update(split="final", expected_examples=COUNT,
        validation_file=relative(folder / "references.jsonl"),
        references_file=relative(folder / "references.jsonl"),
        inputs_manifest=relative(folder / "manifest.json"), final_examples_read=True,
        denominator_policy="All 1034 frozen final examples, including prediction errors/timeouts")
    return config, rows


def load_generation(arm, root, rows):
    directory = root / arm
    generation = read_json(directory / "summary.json")
    checked = read_json(ROOT / read_json(V1_CONFIG)['preflight_report'])
    if (generation.get('arm') != arm or generation.get('project_fine_tuning_applied') != (arm == 'sft')
            or generation.get('generation_config') != checked['generation_config']
            or generation.get('model_config') != checked['model_config']
            or (arm == 'sft' and generation.get('adapter_fingerprint') != checked['adapter_fingerprint'])):
        raise ValueError('Wrong model/adapter/effective generation protocol')
    controller = read_json(directory / "controller_result.json")
    if (generation.get("status") != "pass" or controller.get("status") != "pass"
            or controller.get("exit_code") != 0 or controller.get("job_active_pids_after_cleanup") != []):
        raise ValueError("Generation and process cleanup must both pass")
    for name in ("prediction_count", "generated_count", "attempted_count"):
        if generation.get(name) != COUNT:
            raise ValueError(f"Incomplete final generation: {name}")
    if (generation.get("generation_failure_or_skip_count") != 0
            or generation.get("automatic_retries") != 0
            or generation.get("reference_answers_loaded_for_inference") is not False
            or generation.get("training_performed") is not False):
        raise ValueError("Unexpected generation scope/retries/failures")
    path = directory / "predictions.jsonl"
    evaluation.expect_hash(path, generation["prediction_sha256"])
    expected_reused = PREFIX_COUNT if arm == "base" else 0
    if (generation.get("reused_prefix_count") != expected_reused
            or generation.get("new_generated_count") != COUNT - expected_reused):
        raise ValueError("Unexpected reused prefix/new generation population")
    if arm == "base":
        prefix = PREFIX_PATH.read_bytes()
        if not prefix.endswith(b"\n") or len(prefix.splitlines(keepends=True)) != PREFIX_COUNT:
            raise ValueError("Preserved baseline prefix is not exactly 616 complete rows")
        canonical = path.read_bytes()
        if not canonical.startswith(prefix):
            raise ValueError("Canonical baseline changed the original 616 prediction bytes")
        if b"".join(canonical.splitlines(keepends=True)[:PREFIX_COUNT]) != prefix:
            raise ValueError("Canonical prefix row boundaries differ from preserved evidence")
    raw = read_lines(path)
    if [row.get("source_index") for row in raw] != list(range(COUNT)):
        raise ValueError("Generation IDs/order differ from complete final split")
    for ordinal, row in enumerate(raw, 1):
        if (row.get("ordinal") != ordinal or row.get("attempt_number") != 1
                or row.get("generation_ok") is not True or row.get("generation_status") != "generated"):
            raise ValueError("Each final input must be generated once, without retries")
    return generation, evaluation.validate_predictions(raw, rows)


def dispatch(directory, config, rows, predictions, hardness, deadline):
    """Keep original SQL deadlines, reserve bounded cleanup time, never restart."""
    started = time.perf_counter()
    stop, lock = threading.Event(), threading.Lock()
    results, failures, starts = [], [], []
    with (directory / "per_example.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        def fail(message, **context):
            with lock:
                failures.append({"error": message, **context})
                stop.set()

        def lane(number):
            client = None
            try:
                client = evaluation.PairClient(directory / evaluation.WORKER.name,
                    directory / f"worker_{number}.stderr.log", config["hard_timeout_grace_seconds"])
                for row in rows[number::config["workers"]]:
                    with lock:
                        if stop.is_set():
                            break
                        if time.perf_counter() >= deadline - CLEANUP_RESERVE_SECONDS:
                            failures.append({"error": "Scoring budget exhausted; preserve cleanup reserve", "lane": number})
                            stop.set()
                            break
                        starts.append(row["source_index"])
                    prediction = predictions[row["source_index"]]
                    request = {"request_id": row["source_index"],
                        "db_path": str(ROOT / config["database_dir"] / row["db_id"] / f"{row['db_id']}.sqlite"),
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
                        if len(results) % 100 == 0 or len(results) == COUNT:
                            print(f"{directory.name}: {len(results)}/{COUNT}", flush=True)
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
        "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
        "dispatch_seconds": time.perf_counter() - started,
        "stop_policy": "Stop new dispatch at first observed infrastructure failure; bounded in-flight calls may finish; no replay"}


def outcome_counts(rows):
    return {"outcomes": dict(Counter(execution_outcome(row) for row in rows)),
        "first_error_categories": dict(Counter(first_error_category(row) for row in rows))}


def compare_saved(root):
    """Summarize fixed saved results; this never invokes SQLite or a model."""
    destinations = {arm: root / f"scores_{arm}" for arm in ("base", "sft")}
    if not all((path / "summary.json").is_file() for path in destinations.values()):
        return
    reports = {arm: read_json(path / "summary.json") for arm, path in destinations.items()}
    if not all(report.get("status") == "pass" for report in reports.values()):
        raise ValueError("Cannot compare an incomplete/failed final scoring arm")
    output = root / "comparison.json"
    if output.exists():
        raise FileExistsError("Final comparison already exists; refusing overwrite")
    scores = {}
    sources = {}
    for arm, directory in destinations.items():
        rows = read_lines(directory / "per_example.jsonl")
        if len(rows) != COUNT or sorted(row["source_index"] for row in rows) != list(range(COUNT)):
            raise ValueError("Incomplete or duplicate scored final population")
        scores[arm] = {row["source_index"]: row for row in rows}
        sources[relative(directory / "per_example.jsonl")] = sha256(directory / "per_example.jsonl")
        sources[relative(directory / "summary.json")] = sha256(directory / "summary.json")
    for name in ("generation_config.json", "prompts.jsonl"):
        if (root / "base" / name).read_bytes() != (root / "sft" / name).read_bytes():
            raise ValueError(f"Unfair final generation: {name} differs")
        sources[relative(root / "base" / name)] = sha256(root / "base" / name)
        sources[relative(root / "sft" / name)] = sha256(root / "sft" / name)
    if read_json(destinations["base"] / "protocol.json") != read_json(destinations["sft"] / "protocol.json"):
        raise ValueError("Final EX protocols differ")
    pairs = []
    for index in range(COUNT):
        before, after = scores["base"][index], scores["sft"][index]
        for name in ("source_index", "db_id", "hardness", "question", "gold_sql", "order_matters"):
            if before.get(name) != after.get(name):
                raise ValueError(f"Paired final metadata differs: {index}/{name}")
        pairs.append({"source_index": index, "db_id": after["db_id"], "hardness": after["hardness"],
            "base_ex": before["matched"] is True, "sft_ex": after["matched"] is True,
            "base_executable": before["prediction_executable"] is True,
            "sft_executable": after["prediction_executable"] is True})

    def group(selected):
        indices = [row["source_index"] for row in selected]
        before = evaluation.aggregate([scores["base"][index] for index in indices])
        after = evaluation.aggregate([scores["sft"][index] for index in indices])
        improved = sum(not row["base_ex"] and row["sft_ex"] for row in selected)
        regressed = sum(row["base_ex"] and not row["sft_ex"] for row in selected)
        if improved - regressed != after["execution_correct"] - before["execution_correct"]:
            raise ValueError("Paired EX arithmetic mismatch")
        return {"count": len(selected), "base": before, "sft": after,
            "ex_delta_percentage_points": 100*(after["execution_accuracy"]-before["execution_accuracy"]) if selected else None,
            "executable_delta_percentage_points": 100*(after["prediction_executable_rate"]-before["prediction_executable_rate"]) if selected else None,
            "improved": improved, "regressed": regressed,
            "both_correct": sum(row["base_ex"] and row["sft_ex"] for row in selected),
            "both_wrong": sum(not row["base_ex"] and not row["sft_ex"] for row in selected)}

    verify_frozen()
    write_new(output, {"status": "pass", "sample_scope": "All 1034 held-out official-dev questions / 20 databases",
        "overall": group(pairs), "by_hardness": {level: group([row for row in pairs if row["hardness"] == level]) for level in LEVELS},
        "failure_types": {arm: outcome_counts(list(rows.values())) for arm, rows in scores.items()},
        "improved_source_indices": [row["source_index"] for row in pairs if not row["base_ex"] and row["sft_ex"]],
        "regressed_source_indices": [row["source_index"] for row in pairs if row["base_ex"] and not row["sft_ex"]],
        "source_sha256": sources, "same_prompts_and_generation_config": True, "same_ex_protocol": True,
        "baseline_prefix_reused_count": PREFIX_COUNT, "baseline_new_generated_count": COUNT - PREFIX_COUNT,
        "baseline_prefix_sha256": sha256(PREFIX_PATH), "sft_new_generated_count": COUNT,
        "prediction_replay_or_selection_performed": False,
        "metric_limitations": METRIC_LIMITATIONS, "training_performed": False,
        "final_result_used_for_tuning": False, "automatic_retries": 0})


def main(arm):
    started = time.perf_counter()
    cfg = read_json(CONFIG)
    root = ROOT / cfg["run_root"]
    directory = root / f"scores_{arm}"
    directory.mkdir(parents=False, exist_ok=False)
    write_new(directory / "claim.json", {"arm": arm, "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_retries": 0, "budget_seconds": SCORING_BUDGET_SECONDS, "final_examples_read": True})
    summary = {"status": "failed", "arm": arm, "run_dir": relative(directory),
        "final_examples_read": True, "training_performed": False, "model_generation_performed": False,
        "automatic_retries": 0, "complete_final_scored": False}
    try:
        config, rows = load_final_protocol()
        generation, predictions = load_generation(arm, root, rows)
        hardness = evaluation.label_hardness(rows)
        if not set(hardness.values()).issubset(LEVELS):
            raise ValueError("Unexpected Spider difficulty level")
        for path in (Path(evaluation.__file__), evaluation.WORKER, evaluation.CONFIG, Path(__file__)):
            shutil.copyfile(path, directory / path.name)
        write_new(directory / "protocol.json", config)
        write_new(directory / "inputs.json", {"source_index_order": list(range(COUNT)),
            "predictions_sha256": generation["prediction_sha256"],
            "references_sha256": sha256(ROOT / config["references_file"]),
            "worker_python": evaluation.WORKER_PYTHON, "worker_site_packages": evaluation.PURELIB,
            "hardness_counts": dict(Counter(hardness.values()))})
        results, dispatched = dispatch(directory, config, rows, predictions, hardness,
            started + SCORING_BUDGET_SECONDS)
        write_new(directory / "dispatch.json", dispatched)
        summary.update(dispatch=dispatched, partial_record_count=len(results),
            per_example_path=relative(directory / "per_example.jsonl"))
        if dispatched["failures"]:
            raise RuntimeError("Final scoring infrastructure failure; preserve records and stop without replay")
        if (len(results) != COUNT or sorted(row["source_index"] for row in results) != list(range(COUNT))
                or len(dispatched["attempted_source_indices"]) != COUNT
                or len(set(dispatched["attempted_source_indices"])) != COUNT):
            raise ValueError("Incomplete or repeated final scoring requests")
        verify_frozen()
        evaluation.expect_hash(root / arm / "predictions.jsonl", generation["prediction_sha256"])
        if time.perf_counter() - started > SCORING_BUDGET_SECONDS:
            raise TimeoutError("Final scoring exceeded its 300-second budget")
        summary.update(status="pass", complete_final_scored=True, metric=config["metric"],
            protocol_version=config["version"], upstream_commit=config["upstream_commit"],
            overall=evaluation.aggregate(results),
            by_hardness={level: evaluation.aggregate([row for row in results if row["hardness"] == level]) for level in LEVELS},
            by_database={db: evaluation.aggregate([row for row in results if row["db_id"] == db]) for db in sorted({row["db_id"] for row in rows})},
            failure_types=outcome_counts(results), metric_limitations=METRIC_LIMITATIONS,
            complete_test_suite_accuracy=False, source_files_unchanged=True,
            prediction_sha256=generation["prediction_sha256"])
    except BaseException as error:
        summary.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    summary["wall_seconds"] = time.perf_counter() - started
    write_new(directory / "summary.json", summary)
    print(json.dumps({key: summary.get(key) for key in ("status", "arm", "overall", "error", "wall_seconds")}, ensure_ascii=False), flush=True)
    if summary["status"] != "pass":
        return 1
    try:
        compare_saved(root)
    except BaseException as error:
        write_new(root / "comparison_failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "automatic_retries": 0})
        raise
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("base", "sft"))
    raise SystemExit(main(parser.parse_args().arm))
