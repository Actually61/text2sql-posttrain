"""合成题的独立执行器与比较器，不导入模型库，不替代官方评测。"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


ABS_TOL = 1e-6
REL_TOL = 0.0
WORKER = Path(__file__).with_name("sql_worker.py")
# Windows venv 的 python.exe 可能再启动底层 Python。worker 仅用标准库，
# 直接运行同一安装的解释器，确保父进程 timeout 终止的是执行 SQL 的进程。
_base_python = Path(sys.base_prefix) / "python.exe"
WORKER_PYTHON = str(_base_python) if os.name == "nt" and _base_python.is_file() else sys.executable


def values_equal(left, right):
    """NULL 与文本严格区分；int/float 视为数值并使用明确容差。"""
    if left is None or right is None:
        return left is None and right is None
    if type(left) is int and type(right) is int:
        return left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, abs_tol=ABS_TOL, rel_tol=REL_TOL)
    return type(left) is type(right) and left == right


def rows_equal(left, right):
    return len(left) == len(right) and all(values_equal(a, b) for a, b in zip(left, right))


def compare_rows(predicted_rows, expected_rows, ordered):
    """保留列顺序和重复行；无序情况求二分图完美匹配，避免容差贪心误配。"""
    if len(predicted_rows) != len(expected_rows):
        return False
    if ordered:
        return all(rows_equal(a, b) for a, b in zip(predicted_rows, expected_rows))
    edges = [[j for j, expected in enumerate(expected_rows) if rows_equal(predicted, expected)]
             for predicted in predicted_rows]
    matched = [-1] * len(expected_rows)

    def augment(index, visited):
        for candidate in edges[index]:
            if candidate in visited:
                continue
            visited.add(candidate)
            if matched[candidate] == -1 or augment(matched[candidate], visited):
                matched[candidate] = index
                return True
        return False

    return all(augment(index, set()) for index in range(len(predicted_rows)))


def _execute_fixture(database, fixture, sql, timeout_seconds=5, max_rows=200, query_timeout_seconds=2):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or max_rows < 1:
        raise ValueError("timeout_seconds 和 max_rows 必须为正数。")
    payload = {"schema": database["schema"], "rows": fixture["rows"], "sql": sql,
               "query_timeout_seconds": query_timeout_seconds, "max_rows": max_rows}
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [WORKER_PYTHON, "-I", "-B", "-X", "utf8", str(WORKER)],
            input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
            capture_output=True, text=True, encoding="utf-8", timeout=timeout_seconds,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        return {"valid": False, "predicted_rows": None, "column_count": None,
                "error_type": "parent_timeout", "error": f"独立 SQL 进程超过 {timeout_seconds} 秒，已终止。",
                "seconds": time.perf_counter() - started}
    if process.returncode != 0:
        return {"valid": False, "predicted_rows": None, "column_count": None,
                "error_type": "worker_exit", "error": f"退出码 {process.returncode}: {process.stderr[-500:]}",
                "seconds": time.perf_counter() - started}
    try:
        result = json.loads(process.stdout)
        if not isinstance(result, dict) or type(result.get("valid")) is not bool:
            raise ValueError("缺少有效状态。")
    except (json.JSONDecodeError, ValueError) as error:
        return {"valid": False, "predicted_rows": None, "column_count": None,
                "error_type": "worker_protocol", "error": str(error),
                "seconds": time.perf_counter() - started}
    result["process_seconds"] = time.perf_counter() - started
    return result


def _expected_column_count(case):
    widths = {len(row) for rows in case["expected_rows"].values() for row in rows}
    if len(widths) > 1:
        raise ValueError(f"题目 {case['id']} 的预期行列数不一致。")
    return next(iter(widths), None)


def evaluate_prediction(suite, case, predicted_sql, timeout_seconds=5, max_rows=200):
    database = suite["databases"][case["database_id"]]
    results = []
    expected_width = _expected_column_count(case)
    for fixture in database["fixtures"]:
        expected = case["expected_rows"][fixture["id"]]
        result = _execute_fixture(database, fixture, predicted_sql, timeout_seconds, max_rows)
        width = expected_width
        if width is None:
            # 即使正确结果为空，也须检查列数；避免 SELECT 错列碰巧同为空。
            gold = _execute_fixture(database, fixture, case["gold_sql"], timeout_seconds, max_rows)
            if not gold["valid"]:
                raise ValueError(f"题目 {case['id']} 的 gold 无法确定空结果列数：{gold}")
            width = gold["column_count"]
        matches = bool(result["valid"] and result["column_count"] == width
                       and compare_rows(result["predicted_rows"], expected, case["ordered"]))
        results.append({"fixture_id": fixture["id"], **result,
                        "expected_rows": expected, "expected_column_count": width,
                        "matches_expected": matches})
    return {"all_valid": all(result["valid"] for result in results),
            "all_match": all(result["matches_expected"] for result in results),
            "fixture_results": results,
            "comparison": {"ordered": case["ordered"], "preserve_duplicates": True,
                           "preserve_column_order": True, "numeric_abs_tol": ABS_TOL,
                           "numeric_rel_tol": REL_TOL, "null_distinct_from_text": True}}


def validate_suite(suite, timeout_seconds=5, max_rows=200):
    if not isinstance(suite, dict) or not isinstance(suite.get("suite_id"), str):
        raise ValueError("套件必须有 suite_id。")
    databases = suite.get("databases")
    cases = suite.get("cases")
    if not isinstance(databases, dict) or not databases or not isinstance(cases, list) or not cases:
        raise ValueError("套件必须包含数据库与题目。")
    for database_id, database in databases.items():
        if not isinstance(database.get("schema"), str) or not database["schema"].strip():
            raise ValueError(f"数据库 {database_id} 缺少 schema。")
        fixtures = database.get("fixtures")
        if not isinstance(fixtures, list) or len(fixtures) < 2:
            raise ValueError(f"数据库 {database_id} 至少需要两份 fixture。")
        fixture_ids = [fixture.get("id") for fixture in fixtures]
        if any(not isinstance(item, str) or not item for item in fixture_ids) or len(set(fixture_ids)) != len(fixture_ids):
            raise ValueError(f"数据库 {database_id} 的 fixture ID 无效或重复。")
        if any(not isinstance(fixture.get("rows"), dict) for fixture in fixtures):
            raise ValueError(f"数据库 {database_id} 的 rows 必须按表名组织。")
    case_ids = []
    checks = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("每道题必须是对象。")
        for key in ("id", "database_id", "category", "question", "gold_sql"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"题目缺少非空 {key}。")
        case_ids.append(case["id"])
        if case["database_id"] not in databases or type(case.get("ordered")) is not bool:
            raise ValueError(f"题目 {case['id']} 的数据库或 ordered 设置无效。")
        expected = case.get("expected_rows")
        fixture_ids = {fixture["id"] for fixture in databases[case["database_id"]]["fixtures"]}
        if not isinstance(expected, dict) or set(expected) != fixture_ids:
            raise ValueError(f"题目 {case['id']} 的预期答案未覆盖所有 fixture。")
        if any(not isinstance(rows, list) or any(not isinstance(row, list) for row in rows) for rows in expected.values()):
            raise ValueError(f"题目 {case['id']} 的预期答案须为二维列表。")
        checked = evaluate_prediction(suite, case, case["gold_sql"], timeout_seconds, max_rows)
        if not checked["all_match"]:
            raise ValueError(f"gold 与手工预期不一致：{case['id']}；{json.dumps(checked, ensure_ascii=False)}")
        checks.extend({"case_id": case["id"], **result} for result in checked["fixture_results"])
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("题目 ID 重复。")
    return {"suite_id": suite["suite_id"], "case_count": len(cases),
            "database_count": len(databases), "fixture_execution_count": len(checks),
            "gold_all_match": True, "checks": checks,
            "scope": "synthetic_smoke_only_not_spider_evaluation"}


def run_self_checks():
    database = {"schema": "CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT, score INTEGER);",
                "fixtures": [{"id": "base", "rows": {"items": [
                    [1, "A", 10], [2, "A", 20], [3, "B", None], [4, None, 10], [5, "None", 30]]}}]}
    suite = {"databases": {"demo": database}}
    checks = []

    def require(name, condition, details=None):
        if not condition:
            raise ValueError(f"执行器自检失败：{name}；{details}")
        checks.append({"name": name, "passed": True})

    def example(gold, expected, ordered=False):
        return {"id": "self_check", "database_id": "demo", "gold_sql": gold,
                "ordered": ordered, "expected_rows": {"base": expected}}

    case = example("SELECT label FROM items WHERE score >= 10 AND label IS NOT NULL", [["A"], ["A"], ["None"]])
    equivalents = [
        "SELECT label FROM items WHERE 10 <= score AND label IS NOT NULL",
        "SELECT label FROM items WHERE label IS NOT NULL AND score >= 10",
        "SELECT x.label FROM items AS x WHERE NOT (x.score < 10) AND x.label IS NOT NULL",
    ]
    for index, sql in enumerate(equivalents, 1):
        result = evaluate_prediction(suite, case, sql)
        require(f"equivalent_sql_{index}", result["all_match"], result)
    result = evaluate_prediction(suite, case, "SELECT label FROM items WHERE score >= 20 AND label IS NOT NULL")
    require("wrong_where_rejected", result["all_valid"] and not result["all_match"], result)
    result = evaluate_prediction(suite, case, "SELECT DISTINCT label FROM items WHERE score >= 10 AND label IS NOT NULL")
    require("duplicate_multiplicity_preserved", result["all_valid"] and not result["all_match"], result)

    distinct = example("SELECT DISTINCT label FROM items WHERE label IS NOT NULL", [["A"], ["B"], ["None"]])
    result = evaluate_prediction(suite, distinct, "SELECT label FROM items WHERE label IS NOT NULL")
    require("missing_distinct_rejected", result["all_valid"] and not result["all_match"], result)
    ordered = example("SELECT id FROM items WHERE score IS NOT NULL ORDER BY score, id", [[1], [4], [2], [5]], True)
    result = evaluate_prediction(suite, ordered, "SELECT id FROM items WHERE score IS NOT NULL ORDER BY score DESC, id DESC")
    require("reverse_order_rejected", result["all_valid"] and not result["all_match"], result)
    null_case = example("SELECT label FROM items WHERE id = 4", [[None]])
    result = evaluate_prediction(suite, null_case, "SELECT label FROM items WHERE id = 5")
    require("null_distinct_from_text", result["all_valid"] and not result["all_match"], result)
    columns = example("SELECT id, label FROM items WHERE id = 1", [[1, "A"]])
    result = evaluate_prediction(suite, columns, "SELECT label, id FROM items WHERE id = 1")
    require("column_order_preserved", result["all_valid"] and not result["all_match"], result)
    empty = example("SELECT label FROM items WHERE id < 0", [])
    result = evaluate_prediction(suite, empty, "SELECT id, label FROM items WHERE id < 0")
    require("empty_result_column_count_checked", result["all_valid"] and not result["all_match"], result)

    aggregates = example("SELECT COUNT(*), COUNT(score), SUM(score), AVG(score), MIN(score), MAX(score), ROUND(AVG(score),1) FROM items",
                         [[5, 4, 70, 17.5, 10, 30, 17.5]])
    result = evaluate_prediction(suite, aggregates, aggregates["gold_sql"])
    require("aggregate_functions_allowed", result["all_match"], result)
    functions = example("SELECT COALESCE(label, 'unknown'), IFNULL(score,-1), LOWER('ABC') FROM items WHERE id=4", [["unknown", 10, "abc"]])
    result = evaluate_prediction(suite, functions, functions["gold_sql"])
    require("scalar_functions_allowed", result["all_match"], result)
    literal = example("SELECT ';' FROM items WHERE id=1", [[";"]])
    result = evaluate_prediction(suite, literal, literal["gold_sql"])
    require("semicolon_inside_literal_preserved", result["all_match"], result)

    fixture = database["fixtures"][0]
    denied_queries = {
        "drop_blocked": "DROP TABLE items",
        "insert_blocked": "INSERT INTO items VALUES (6,'X',10)",
        "attach_blocked": "ATTACH DATABASE ':memory:' AS other",
        "pragma_blocked": "PRAGMA query_only=OFF",
        "catalog_read_blocked": "SELECT name FROM sqlite_master",
        "extension_function_blocked": "SELECT load_extension('not_a_library')",
    }
    for name, sql in denied_queries.items():
        result = _execute_fixture(database, fixture, sql)
        require(name, not result["valid"] and result["error_type"] == "query_rejected", result)
    result = _execute_fixture(database, fixture, "SELECT missing_column FROM items")
    require("invalid_sql_reported", not result["valid"] and result["error_type"] == "sqlite_error", result)
    result = _execute_fixture(database, fixture, "SELECT 1; SELECT 2")
    require("multiple_statements_rejected", not result["valid"], result)
    result = _execute_fixture(database, fixture, "SELECT id FROM items", max_rows=2)
    require("result_limit_enforced", not result["valid"] and result["error_type"] == "result_limit", result)
    heavy = {"schema": "CREATE TABLE numbers (value INTEGER);",
             "fixtures": [{"id": "base", "rows": {"numbers": [[i] for i in range(100)]}}]}
    result = _execute_fixture(heavy, heavy["fixtures"][0],
                              "SELECT COUNT(*) FROM numbers a, numbers b, numbers c, numbers d, numbers e, numbers f",
                              query_timeout_seconds=0.01)
    require("query_deadline_enforced", not result["valid"] and result["error_type"] == "query_timeout", result)
    result = _execute_fixture(database, fixture, "SELECT COUNT(*) FROM items", timeout_seconds=0.0001)
    require("parent_timeout_enforced", not result["valid"] and result["error_type"] == "parent_timeout", result)
    require("numeric_tolerance_accepts_small_difference", compare_rows([[1.0000001]], [[1]], False))
    require("numeric_tolerance_rejects_large_difference", not compare_rows([[1.001]], [[1]], False))
    require("large_integer_counts_compared_exactly", not compare_rows([[1000001]], [[1000000]], False))
    require("missing_round_not_hidden_by_relative_tolerance",
            not compare_rows([[8666.666666666666]], [[8666.67]], False))
    require("text_distinct_from_number", not compare_rows([["1"]], [[1]], False))
    require("unordered_rows_ignore_order", compare_rows([[2], [1]], [[1], [2]], False))
    require("ambiguous_tolerance_uses_complete_matching",
            compare_rows([[1.0], [1.0000015]], [[1.00000075], [0.99999925]], False))
    return {"passed_count": len(checks), "checks": checks,
            "numeric_abs_tol": ABS_TOL, "numeric_rel_tol": REL_TOL,
            "worker_query_timeout_seconds": 2, "parent_timeout_seconds": 5}
