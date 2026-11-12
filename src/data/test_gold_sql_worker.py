"""CPU-only synthetic SQLite tests; never open Spider databases or split files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from gold_sql_worker import execute_request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"


class GoldSQLWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        CACHE_ROOT.mkdir(exist_ok=True)
        cls.temp_directory = tempfile.TemporaryDirectory(prefix="gold_worker_test_", dir=CACHE_ROOT)
        cls.temp_root = Path(cls.temp_directory.name).resolve()
        if not cls.temp_root.is_relative_to(CACHE_ROOT.resolve()):
            raise RuntimeError("Test directory escaped the project cache")
        cls.db_path = cls.temp_root / "fixture.sqlite"
        connection = sqlite3.connect(cls.db_path)
        try:
            connection.execute("CREATE TABLE numbers (n INTEGER)")
            connection.executemany("INSERT INTO numbers VALUES (?)", ((n,) for n in range(1500)))
            connection.execute("CREATE TABLE duplicates (n INTEGER)")
            connection.executemany("INSERT INTO duplicates VALUES (?)", [(1,), (1,), (2,)])
            connection.execute("CREATE TABLE text_values (value TEXT)")
            connection.execute("INSERT INTO text_values VALUES (?)", ("already valid \ufffd",))
            connection.execute("INSERT INTO text_values VALUES (CAST(? AS TEXT))", (b"\x80\xff",))
            connection.execute("INSERT INTO text_values VALUES (CAST(? AS TEXT))", (b"valid\xc3",))
            connection.commit()
        finally:
            connection.close()
        cls.original_hash = hashlib.sha256(cls.db_path.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls) -> None:
        # Verify the resolved deletion target remains inside the known cache.
        if not cls.temp_root.is_relative_to(CACHE_ROOT.resolve()):
            raise RuntimeError("Refusing cleanup outside project cache")
        cls.temp_directory.cleanup()

    def tearDown(self) -> None:
        self.assertEqual(hashlib.sha256(self.db_path.read_bytes()).hexdigest(), self.original_hash)

    def request(self, sql: str, **overrides: object) -> dict:
        request = {"request_id": "test", "db_path": str(self.db_path), "sql": sql,
                   "timeout_seconds": 2.0, "decode_mode": "strict"}
        request.update(overrides)
        return execute_request(request)

    def assert_ok_rows(self, sql: str, rows: int, columns: int = 1) -> dict:
        result = self.request(sql)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["row_count"], rows)
        self.assertEqual(result["column_count"], columns)
        self.assertEqual(result["empty_result"], rows == 0)
        self.assertTrue(result["row_count_complete"])
        self.assertEqual(result["replacement_cell_count"], 0)
        return result

    def test_normal_select(self) -> None:
        self.assert_ok_rows("SELECT n, n * 2 FROM numbers WHERE n < 3", 3, 2)

    def test_empty_result(self) -> None:
        self.assert_ok_rows("SELECT n FROM numbers WHERE n < 0", 0)

    def test_duplicate_rows_are_preserved(self) -> None:
        self.assert_ok_rows("SELECT n FROM duplicates", 3)

    def test_full_fetch_exceeds_old_200_row_limit(self) -> None:
        self.assert_ok_rows("SELECT n FROM numbers", 1500)

    def test_aggregate_and_builtin_functions(self) -> None:
        self.assert_ok_rows("SELECT COUNT(*), AVG(n), COALESCE(MAX(n), 0) FROM numbers", 1, 3)

    def test_cte_union_and_recursive_reads(self) -> None:
        self.assert_ok_rows("WITH x AS (SELECT 1 AS n UNION SELECT 2) SELECT n FROM x", 2)
        self.assert_ok_rows("WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<5) SELECT n FROM x", 5)

    def test_sql_errors(self) -> None:
        for sql in ("SELECT missing_column FROM numbers", "SELEC n FROM numbers"):
            with self.subTest(sql=sql):
                result = self.request(sql)
                self.assertEqual(result["status"], "sql_error", result)
                self.assertIn("sqlite_errorcode", result)

    def test_write_attach_pragma_and_transaction_rejected(self) -> None:
        statements = (
            "DELETE FROM numbers", "UPDATE numbers SET n=0", "INSERT INTO numbers VALUES(9)",
            "CREATE TABLE forbidden(n)", "DROP TABLE numbers", "ATTACH ':memory:' AS other",
            "PRAGMA user_version", "BEGIN", "SELECT load_extension('missing')",
        )
        for sql in statements:
            with self.subTest(sql=sql):
                self.assertEqual(self.request(sql)["status"], "query_rejected")

    def test_multiple_statements_not_executed(self) -> None:
        self.assertEqual(self.request("SELECT 1; DELETE FROM numbers")["status"], "sql_error")
        self.assert_ok_rows("SELECT n FROM numbers", 1500)

    def test_query_timeout_and_next_query(self) -> None:
        result = self.request(
            "SELECT COUNT(*) FROM numbers a CROSS JOIN numbers b CROSS JOIN numbers c",
            timeout_seconds=0.01,
        )
        self.assertEqual(result["status"], "query_timeout", result)
        self.assertFalse(result["row_count_complete"])
        self.assertLess(result["seconds"], 2.0)
        self.assert_ok_rows("SELECT 1", 1)

    def test_invalid_utf8_strict(self) -> None:
        result = self.request("SELECT value FROM text_values ORDER BY rowid")
        self.assertEqual(result["status"], "decode_error", result)
        self.assertFalse(result["row_count_complete"])
        self.assertEqual(result["replacement_cell_count"], 0)

    def test_invalid_utf8_replace_counts_cells_not_characters(self) -> None:
        result = self.request("SELECT value FROM text_values ORDER BY rowid", decode_mode="replace")
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["replacement_cell_count"], 2)

    def test_missing_database_never_created(self) -> None:
        missing = self.temp_root / "missing.sqlite"
        result = self.request("SELECT 1", db_path=str(missing))
        self.assertEqual(result["status"], "database_error", result)
        self.assertFalse(missing.exists())

    def test_invalid_requests(self) -> None:
        for overrides in ({"timeout_seconds": float("nan")}, {"timeout_seconds": 0},
                          {"timeout_seconds": True}, {"decode_mode": "ignore"},
                          {"db_path": "relative.sqlite"}, {"sql": ""}):
            with self.subTest(overrides=overrides):
                result = self.request("SELECT 1", **overrides) if "sql" not in overrides else execute_request(overrides)
                self.assertEqual(result["status"], "worker_error", result)
                json.dumps(result, allow_nan=False)

    def test_jsonl_protocol_continues_after_bad_request(self) -> None:
        worker = Path(__file__).with_name("gold_sql_worker.py")
        request = {"request_id": "first", "db_path": str(self.db_path), "sql": "SELECT 1"}
        another = dict(request, request_id="second", sql="SELECT n FROM numbers")
        input_text = json.dumps(request) + "\n{invalid json\n" + json.dumps(another) + "\n"
        result = subprocess.run(
            [sys.executable, "-X", "utf8", str(worker)], input=input_text,
            capture_output=True, text=True, encoding="utf-8", timeout=15, check=True,
        )
        self.assertEqual(result.stderr, "")
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(responses), 3)
        self.assertEqual([item["status"] for item in responses], ["ok", "worker_error", "ok"])
        self.assertEqual(responses[0]["request_id"], "first")
        self.assertEqual(responses[2]["request_id"], "second")
        self.assertEqual(responses[2]["row_count"], 1500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
