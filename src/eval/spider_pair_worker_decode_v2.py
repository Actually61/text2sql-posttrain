"""Read-only SQL execution with lossless TEXT decoding and the pinned comparator.

This process never rewrites SQL, substitutes values, or strips DISTINCT. Rows
remain local Python values; stdout contains only JSON-safe status metadata.
The parent MUST impose a hard process deadline: a progress handler cannot stop
every SQLite operation, and the official comparator is synchronous Python.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
from typing import Any

PROJECT_ROOT = Path(os.environ["TEXT2SQL_PROJECT_ROOT"]) if os.environ.get("TEXT2SQL_PROJECT_ROOT") else Path(__file__).resolve().parents[2]
if not PROJECT_ROOT.is_absolute():
    raise RuntimeError("TEXT2SQL_PROJECT_ROOT must be an absolute project path")
PROJECT_ROOT = PROJECT_ROOT.resolve()
VENDOR_ROOT = PROJECT_ROOT / "third_party/test-suite-sql-eval"
sys.path.insert(0, str(VENDOR_ROOT))
import exec_eval as official_exec_eval

if Path(official_exec_eval.__file__).resolve() != (VENDOR_ROOT / "exec_eval.py").resolve():
    raise RuntimeError("Imported result comparator does not come from the pinned vendor directory")


class QueryDeadlineReached(Exception):
    """Cooperative SQL deadline expired."""


def _positive_seconds(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return seconds


def _execute_query(path: Path, sql: str, timeout: float) -> tuple[dict, list[tuple]]:
    """Consume all rows while preserving SQLite values and read-only state."""
    started = time.perf_counter()
    metadata = {"status": "worker_error", "row_count": 0, "column_count": 0,
                "empty_result": None, "row_count_complete": False,
                "decode_mode": "utf-8/surrogateescape", "invalid_text_values": 0, "seconds": 0.0}
    rows: list[tuple] = []
    connection = None
    cursor = None
    denied = False
    expired = False
    query_started = False
    try:
        if not sql.strip():
            metadata.update(status="sql_error", error="SQL is empty")
            return metadata, rows
        if not path.is_file():
            metadata.update(status="database_error", error="Database file does not exist or is not a file")
            return metadata, rows
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")
        # Only SQLite TEXT passes through text_factory. BLOB remains bytes.
        # Invalid UTF-8 is represented reversibly, never ignored or replaced.
        def decode_text(raw):
            value = raw.decode("utf-8", errors="surrogateescape")
            if any(0xDC80 <= ord(character) <= 0xDCFF for character in value):
                metadata["invalid_text_values"] += 1
            return value
        connection.text_factory = decode_text
        read_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
        if hasattr(sqlite3, "SQLITE_RECURSIVE"):
            read_actions.add(sqlite3.SQLITE_RECURSIVE)

        def authorize(action, arg1, arg2, database, source):
            nonlocal denied
            if action in read_actions:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").casefold() not in {
                    "load_extension", "readfile", "writefile"}:
                return sqlite3.SQLITE_OK
            denied = True
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        deadline = time.perf_counter() + timeout

        def check_deadline():
            nonlocal expired
            if time.perf_counter() >= deadline:
                expired = True
                raise QueryDeadlineReached("Query exceeded its time budget")

        def progress():
            nonlocal expired
            if time.perf_counter() >= deadline:
                expired = True
                return 1
            return 0

        connection.set_progress_handler(progress, 1000)
        query_started = True
        check_deadline()
        cursor = connection.execute(sql)
        if cursor.description is None:
            metadata.update(status="query_rejected", error="Statement has no query result")
            return metadata, rows
        metadata["column_count"] = len(cursor.description)
        while True:
            check_deadline()
            batch = cursor.fetchmany(512)
            rows.extend(batch)
            metadata["row_count"] = len(rows)
            check_deadline()
            if not batch:
                break
        metadata.update(status="ok", row_count_complete=True, empty_result=len(rows) == 0)
    except UnicodeDecodeError as exc:
        metadata.update(status="decode_error", error=str(exc))
    except QueryDeadlineReached as exc:
        metadata.update(status="query_timeout", error=str(exc))
    except sqlite3.Error as exc:
        code = getattr(exc, "sqlite_errorcode", None)
        name = getattr(exc, "sqlite_errorname", None)
        if code is not None:
            metadata["sqlite_errorcode"] = code
        if name is not None:
            metadata["sqlite_errorname"] = name
        db_codes = {sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_CORRUPT,
                    sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_IOERR}
        if expired:
            status = "query_timeout"
        elif denied:
            status = "query_rejected"
        elif not query_started or (code is not None and (code & 255) in db_codes):
            status = "database_error"
        else:
            status = "sql_error"
        metadata.update(status=status, error=str(exc))
    except OSError as exc:
        metadata.update(status="database_error", error=str(exc))
    except Exception as exc:
        metadata.update(status="worker_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            if connection is not None:
                connection.close()
            metadata["seconds"] = round(time.perf_counter() - started, 6)
    return metadata, rows


def execute_request(request: Any) -> dict:
    """Score one gold/predicted SQL pair on exactly one existing database."""
    started = time.perf_counter()
    response = {"request_id": None, "status": "worker_error", "matched": None,
                "prediction_executable": False, "gold": None, "prediction": None,
                "order_matters": None, "comparison_seconds": None, "seconds": 0.0}
    try:
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        request_id = request.get("request_id")
        if request_id is not None and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
            raise ValueError("request_id must be a string, integer or null")
        response["request_id"] = request_id
        path_value = request.get("db_path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            raise ValueError("db_path must be an absolute path string")
        path = Path(path_value)
        gold_sql, predicted_sql = request.get("gold_sql"), request.get("predicted_sql")
        if not isinstance(gold_sql, str) or not isinstance(predicted_sql, str):
            raise ValueError("gold_sql and predicted_sql must be strings")
        query_budget = _positive_seconds(request.get("query_timeout_seconds", 2.0), "query_timeout_seconds")
        comparison_budget = _positive_seconds(request.get("comparison_timeout_seconds", 2.0), "comparison_timeout_seconds")
        seed = request.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        response["query_timeout_seconds"] = query_budget
        response["comparison_timeout_seconds"] = comparison_budget
        # Deliberately the same rule as the pinned upstream eval_exec_match.
        response["order_matters"] = "order by" in gold_sql.lower()
        gold, gold_rows = _execute_query(path, gold_sql, query_budget)
        predicted, predicted_rows = _execute_query(path, predicted_sql, query_budget)
        response["gold"], response["prediction"] = gold, predicted
        response["prediction_executable"] = predicted["status"] == "ok"
        if gold["status"] != "ok" or predicted["status"] != "ok":
            response["status"] = ("both_error" if gold["status"] != "ok" and predicted["status"] != "ok"
                                  else "gold_error" if gold["status"] != "ok" else "prediction_error")
            return response
        comparison_started = time.perf_counter()
        try:
            random.seed(seed)
            matched = official_exec_eval.result_eq(gold_rows, predicted_rows, response["order_matters"])
            elapsed = time.perf_counter() - comparison_started
            response["comparison_seconds"] = round(elapsed, 6)
            if elapsed > comparison_budget:
                # This is an observed overrun, NOT a claim of interruption.
                # The parent kills a still-running comparison at its hard deadline.
                response.update(status="comparison_error", error="Comparator returned after its time budget")
            else:
                response.update(status="scored", matched=bool(matched))
        except Exception as exc:
            response.update(status="comparison_error", error=f"{type(exc).__name__}: {exc}")
            response["comparison_seconds"] = round(time.perf_counter() - comparison_started, 6)
    except Exception as exc:
        response.update(status="worker_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        response["seconds"] = round(time.perf_counter() - started, 6)
    return response


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def main():
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    for line in sys.stdin:
        try:
            request = json.loads(line, parse_constant=_reject_nonfinite)
            response = execute_request(request)
        except Exception as exc:
            response = execute_request(None)
            response["error"] = f"Invalid JSONL request: {type(exc).__name__}: {exc}"
        sys.stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
