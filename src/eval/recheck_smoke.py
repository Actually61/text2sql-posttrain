"""只重新执行已保存的20条SQL，不加载模型，不修改预测结果。"""

import hashlib
import json
from pathlib import Path

import yaml

from smoke_checks import WORKER_PYTHON, evaluate_prediction, run_self_checks, validate_suite


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((ROOT / "reports/phase0_smoke_summary.json").read_text(encoding="utf-8"))
    suite_path = ROOT / summary["config"]["suite_config"]
    if digest(suite_path) != summary["source_sha256"][summary["config"]["suite_config"]]:
        raise ValueError("题目文件已改变，不能将本次结果称为同一套件的复核。")
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    predictions_path = Path(summary["run_dir"]) / "predictions.jsonl"
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    cases = {case["id"]: case for case in suite["cases"]}
    if len(predictions) != 20 or {row["id"] for row in predictions} != set(cases):
        raise ValueError("保存结果与20道固定题不一致。")
    validation = validate_suite(suite)
    self_checks = run_self_checks()
    reviews = []
    for row in predictions:
        new = evaluate_prediction(suite, cases[row["id"]], row["predicted_sql"])
        old = row["evaluation"]
        keys = ("fixture_id", "valid", "matches_expected", "predicted_rows", "expected_rows", "column_count", "error_type")
        old_details = [{key: result.get(key) for key in keys} for result in old["fixture_results"]]
        new_details = [{key: result.get(key) for key in keys} for result in new["fixture_results"]]
        reviews.append({"id": row["id"], "unchanged": old_details == new_details, "evaluation": new})
    result = {"run_id": summary["run_id"], "scope": "execution_only_no_model_generation",
              "predictions_sha256": digest(predictions_path), "suite_sha256": digest(suite_path),
              "original_checker_sha256": summary["source_sha256"]["src/eval/smoke_checks.py"],
              "current_checker_sha256": digest(ROOT / "src/eval/smoke_checks.py"),
              "worker_python": WORKER_PYTHON, "validation": validation, "self_checks": self_checks,
              "all_unchanged": all(row["unchanged"] for row in reviews), "reviews": reviews,
              "all_valid_count": sum(row["evaluation"]["all_valid"] for row in reviews),
              "all_match_count": sum(row["evaluation"]["all_match"] for row in reviews)}
    output = ROOT / "reports/smoke_recheck.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["all_unchanged"]:
        raise RuntimeError(f"执行复核出现差异，请检查 {output}")
    print(f"标准结果 {validation['fixture_execution_count']} 次通过；判分自检 {self_checks['passed_count']} 项通过。")
    print(f"同一20条SQL重新执行：两份均可执行 {result['all_valid_count']}/20；两份均匹配 {result['all_match_count']}/20。")
    print("SMOKE_RECHECK: PASS；40份实际结果、错误类型和判分均与原记录一致，未重新生成。")
    print(f"复核证据：{output}")


if __name__ == "__main__":
    main()
