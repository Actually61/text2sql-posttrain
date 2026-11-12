"""Execute gold SQL in a persistent, read-only JSONL worker (stdlib only).

The parent process must impose a hard timeout as SQLite's progress handler
cannot interrupt every operation. Successful execution is not semantic proof.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


class QueryDeadlineReached(Exception):
    """The cooperative per-query deadline expired."""


def execute_request(request: Any) -> dict[str, Any]:
    """Open one existing database, execute one statement and consume all rows."""
    started = time.perf_counter()
    response: dict[str, Any] = {
        "request_id": None,
        "status": "worker_error",
        "row_count": 0,
        "column_count": 0,
        "empty_result": None,
        "row_count_complete": False,
        "seconds": 0.0,
        "decode_mode": "strict",
        "replacement_cell_count": 0,
    }
    connection: sqlite3.Connection | None = None
    cursor: sqlite3.Cursor | None = None
    denied_action = False
    deadline_reached = False
    query_started = False

    try:
        if not isinstance(request, dict):
            raise ValueError("Each request must be a JSON object")
        request_id = request.get("request_id")
        # Keep protocol IDs simple and guarantee standards-compliant JSON.
        if request_id is not None and (
            isinstance(request_id, bool) or not isinstance(request_id, (str, int))
        ):
            raise ValueError("request_id must be a string, integer or null")
        response["request_id"] = request_id
        mode = request.get("decode_mode", "strict")
        if mode not in ("strict", "replace"):
            raise ValueError("decode_mode must be strict or replace")
        response["decode_mode"] = mode
        timeout = request.get("timeout_seconds", 2.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)):
            raise ValueError("timeout_seconds must be a finite positive number")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        sql = request.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a nonempty string")
        path_value = request.get("db_path")
        if not isinstance(path_value, str):
            raise ValueError("db_path must be an absolute path string")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError("db_path must be absolute")
        if not path.is_file():
            response["status"] = "database_error"
            response["error"] = "Database file does not exist or is not a file"
            return response

        # Set protections before installing the authorizer (which denies PRAGMA).
        connection = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout
        )
        connection.enable_load_extension(False)
        connection.execute("PRAGMA query_only = ON")

        def decode_text(raw: bytes) -> str:
            try:
                return raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                if mode == "strict":
                    raise
                response["replacement_cell_count"] += 1
                return raw.decode("utf-8", errors="replace")

        connection.text_factory = decode_text
        read_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
        recursive_action = getattr(sqlite3, "SQLITE_RECURSIVE", None)
        if recursive_action is not None:
            read_actions.add(recursive_action)
        forbidden_functions = {"load_extension", "readfile", "writefile"}

        def authorize(action: int, arg1: str | None, arg2: str | None,
                      database: str | None, source: str | None) -> int:
            nonlocal denied_action
            if action in read_actions:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION:
                # No user-defined functions/extensions are registered in this
                # fresh connection. Preserve SQLite built-ins used by gold SQL.
                function_name = (arg2 or arg1 or "").casefold()
                if function_name not in forbidden_functions:
                    return sqlite3.SQLITE_OK
            denied_action = True
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        deadline = time.perf_counter() + timeout

        def check_deadline() -> None:
            nonlocal deadline_reached
            if time.perf_counter() >= deadline:
                deadline_reached = True
                raise QueryDeadlineReached("Query exceeded its time budget")

        def progress() -> int:
            nonlocal deadline_reached
            if time.perf_counter() >= deadline:
                deadline_reached = True
                return 1
            return 0

        connection.set_progress_handler(progress, 1000)
        query_started = True
        check_deadline()
        cursor = connection.execute(sql)
        if cursor.description is None:
            response["status"] = "query_rejected"
            response["error"] = "Statement does not produce a query result"
            return response
        response["column_count"] = len(cursor.description)
        while True:
            check_deadline()
            rows = cursor.fetchmany(512)
            response["row_count"] += len(rows)
            # Check after the batch as decoding/materialization also takes time.
            check_deadline()
            if not rows:
                break
        response["status"] = "ok"
        response["row_count_complete"] = True
        response["empty_result"] = response["row_count"] == 0
    except UnicodeDecodeError as exc:
        response["status"] = "decode_error"
        response["error"] = str(exc)
    except QueryDeadlineReached as exc:
        response["status"] = "query_timeout"
        response["error"] = str(exc)
    except sqlite3.Error as exc:
        code = getattr(exc, "sqlite_errorcode", None)
        name = getattr(exc, "sqlite_errorname", None)
        if code is not None:
            response["sqlite_errorcode"] = code
        if name is not None:
            response["sqlite_errorname"] = name
        database_error_codes = {
            sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_CORRUPT,
            sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_IOERR,
        }
        if deadline_reached:
            response["status"] = "query_timeout"
        elif denied_action:
            response["status"] = "query_rejected"
        elif not query_started or (code is not None and (code & 255) in database_error_codes):
            response["status"] = "database_error"
        else:
            response["status"] = "sql_error"
        response["error"] = str(exc)
    except (ValueError, TypeError, OverflowError) as exc:
        response["status"] = "worker_error"
        response["error"] = str(exc)
    except OSError as exc:
        response["status"] = "database_error"
        response["error"] = str(exc)
    except Exception as exc:
        response["status"] = "worker_error"
        response["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Always release handles, including failed fetches and interrupted SQL.
        try:
            if cursor is not None:
                cursor.close()
        finally:
            if connection is not None:
                connection.close()
            response["seconds"] = round(time.perf_counter() - started, 6)
    return response


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def main() -> None:
    # UTF-8 is explicit so protocol behavior is independent of Windows locale.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    for line in sys.stdin:
        try:
            request = json.loads(line, parse_constant=_reject_nonfinite)
            response = execute_request(request)
        except Exception as exc:
            response = execute_request(None)
            response["error"] = f"Invalid request or worker failure: {type(exc).__name__}: {exc}"
        sys.stdout.write(json.dumps(response, ensure_ascii=True, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
