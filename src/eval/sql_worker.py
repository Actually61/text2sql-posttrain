"""在独立进程中的内存 SQLite 上执行一条只读查询，仅依赖标准库。"""

import json
import math
import sqlite3
import sys
import time


ALLOWED_FUNCTIONS = {
    "count", "sum", "avg", "min", "max", "total", "round", "abs",
    "coalesce", "ifnull", "nullif", "iif", "like", "glob", "lower", "upper",
    "length", "substr", "substring", "trim", "ltrim", "rtrim", "replace",
    "instr", "typeof", "group_concat", "date", "time", "datetime", "strftime",
    "julianday", "unixepoch", "printf", "format",
}


def quoted_identifier(name):
    if not isinstance(name, str) or not name:
        raise ValueError("表名必须是非空字符串。")
    return '"' + name.replace('"', '""') + '"'


def execute(payload):
    started = time.perf_counter()
    connection = sqlite3.connect(":memory:")
    try:
        # schema 与 rows 是本地固定测试数据，不来自模型输出。
        connection.executescript(payload["schema"])
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )}
        for table, rows in payload["rows"].items():
            if table not in tables:
                raise ValueError(f"测试数据包含 schema 中不存在的表：{table}")
            column_count = len(connection.execute(f"PRAGMA table_info({quoted_identifier(table)})").fetchall())
            if any(not isinstance(row, list) or len(row) != column_count for row in rows):
                raise ValueError(f"表 {table} 的测试行列数不正确。")
            placeholders = ",".join("?" for _ in range(column_count))
            connection.executemany(f"INSERT INTO {quoted_identifier(table)} VALUES ({placeholders})", rows)
        connection.commit()
        connection.execute("PRAGMA query_only = ON")
        denied = []

        def authorize(action, arg1, arg2, database, source):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            # SQLite 对 COUNT(*) 或常量投影还可能发出 database=None、column='' 的表读取事件。
            is_main_read = database == "main" or (database is None and arg2 == "")
            if action == sqlite3.SQLITE_READ and is_main_read and arg1 in tables:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            if len(denied) < 3:
                denied.append({"action": action, "arg1": arg1, "arg2": arg2})
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        query_timeout = float(payload["query_timeout_seconds"])
        if not math.isfinite(query_timeout) or query_timeout <= 0:
            raise ValueError("查询超时必须为正数。")
        max_rows = int(payload["max_rows"])
        if max_rows < 1:
            raise ValueError("最大结果行数必须为正数。")
        sql = payload["sql"]
        if not isinstance(sql, str) or not sql.strip():
            return {"valid": False, "predicted_rows": None, "column_count": None,
                    "error_type": "empty_sql", "error": "SQL 为空。", "seconds": 0.0}
        query_started = time.perf_counter()
        deadline = query_started + query_timeout
        deadline_reached = False

        def check_deadline():
            nonlocal deadline_reached
            deadline_reached = time.perf_counter() >= deadline
            return int(deadline_reached)

        connection.set_progress_handler(check_deadline, 1000)
        try:
            cursor = connection.execute(sql)
            if cursor.description is None:
                return {"valid": False, "predicted_rows": None, "column_count": None,
                        "error_type": "not_a_query", "error": "没有返回查询列。",
                        "seconds": time.perf_counter() - query_started}
            column_count = len(cursor.description)
            rows = cursor.fetchmany(max_rows + 1)
            if len(rows) > max_rows:
                return {"valid": False, "predicted_rows": None, "column_count": column_count,
                        "error_type": "result_limit", "error": f"结果超过 {max_rows} 行上限。",
                        "seconds": time.perf_counter() - query_started}
            return {"valid": True, "predicted_rows": rows, "column_count": column_count,
                    "seconds": time.perf_counter() - query_started}
        except sqlite3.Error as error:
            kind = "query_timeout" if deadline_reached else ("query_rejected" if denied else "sqlite_error")
            return {"valid": False, "predicted_rows": None, "column_count": None,
                    "error_type": kind, "error": str(error),
                    "seconds": time.perf_counter() - query_started}
    except (KeyError, TypeError, ValueError, sqlite3.Error) as error:
        return {"valid": False, "predicted_rows": None, "column_count": None,
                "error_type": "fixture_error", "error": str(error),
                "seconds": time.perf_counter() - started}
    finally:
        connection.close()


def main():
    try:
        result = execute(json.load(sys.stdin))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        # 不把异常栈混入 stdout，父进程总能取得可判定的 JSON。
        print(json.dumps({"valid": False, "predicted_rows": None, "column_count": None,
                          "error_type": "worker_error", "error": f"{type(error).__name__}: {error}",
                          "seconds": 0.0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
