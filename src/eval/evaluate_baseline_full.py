"""只评分本轮统一重新生成的625题；复用原有EX协议，独立保存汇总。"""
import argparse
import json
from pathlib import Path
import shutil
import time

import evaluate_spider as evaluation

ROOT = Path(__file__).resolve().parents[2]


def main(predictions_path):
    started = time.perf_counter()
    generation_path = ROOT / "reports/baseline_full_generation.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    predictions_path = Path(predictions_path).resolve()
    expected_path = (ROOT / generation["prediction_path"]).resolve()
    if predictions_path != expected_path:
        raise ValueError("必须使用本轮全量生成报告指定的原始预测文件，不能混入pilot或其它轮次")
    if generation["prediction_count"] != 625:
        raise ValueError("全量评测固定625条记录，失败也不能删除")
    evaluation.expect_hash(predictions_path, generation["prediction_sha256"])
    config, rows, fingerprints = evaluation.load_validation()
    if config["expected_examples"] != 625 or len(rows) != 625:
        raise ValueError("全量验证范围发生变化")
    evaluation.validate_predictions(evaluation.read_jsonl(predictions_path), rows)
    fingerprints[evaluation.relative(generation_path)] = evaluation.digest(generation_path)
    fingerprints[evaluation.relative(Path(__file__))] = evaluation.digest(Path(__file__))
    summary = evaluation.run_evaluation(predictions_path, "baseline_full", config, rows, fingerprints)
    run_dir = ROOT / summary["run_dir"]
    shutil.copyfile(Path(__file__), run_dir / Path(__file__).name)
    shutil.copyfile(generation_path, run_dir / "generation_summary.json")
    summary.update(
        sample_scope="complete 625-question internal validation, not final evaluation",
        generation_run_id=generation["run_id"],
        prediction_file_sha256=generation["prediction_sha256"],
        prediction_file=evaluation.relative(predictions_path),
        generation_report_sha256=evaluation.digest(generation_path),
        evaluation_pipeline_wall_seconds=round(time.perf_counter()-started, 3),
        evaluation_pipeline_timing_scope="from main entry through validation and scoring; excludes Python startup/imports and final summary write",
        training_performed=False,
    )
    (run_dir / "summary.json").write_bytes(evaluation.json_bytes(summary))
    (ROOT / "reports/baseline_full_eval.json").write_bytes(evaluation.json_bytes(summary))
    print("FULL_BASELINE_EVALUATION: PASS; independent report saved; fixed denominator=625", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="本次生成报告中的625条原始预测JSONL绝对路径")
    main(parser.parse_args().predictions)
