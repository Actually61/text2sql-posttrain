"""Synthetic fixture tests for the official comparator and read-only execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from spider_pair_worker import execute_request

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache"
WORKER = Path(__file__).with_name("spider_pair_worker.py")


class SpiderPairWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CACHE.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="spider_pair_test_", dir=CACHE)
        cls.directory = Path(cls.temp.name).resolve()
        if not cls.directory.is_relative_to(CACHE.resolve()):
            raise RuntimeError("Temporary directory escaped cache")
        cls.db = cls.directory / "fixture.sqlite"
        conn = sqlite3.connect(cls.db)
        conn.executescript("""
            CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
            INSERT INTO people VALUES (1, 'A', 58), (2, 'B', 65), (3, 'C', 20);
            CREATE TABLE duplicates (n INTEGER);
            INSERT INTO duplicates VALUES (1), (1), (2);
            CREATE TABLE numbers (n INTEGER);
            CREATE TABLE nullable (n INTEGER, txt TEXT);
            INSERT INTO nullable VALUES (NULL, NULL), (1, 'a');
            CREATE TABLE values_data (value TEXT);
            INSERT INTO values_data VALUES ('> ='), ('value'), ('ok');
            CREATE TABLE invalid_text (value TEXT);
            INSERT INTO invalid_text VALUES (CAST(x'80ff' AS TEXT));
        """)
        conn.executemany("INSERT INTO numbers VALUES (?)", ((n,) for n in range(1500)))
        conn.commit()
        conn.close()
        cls.original_hash = hashlib.sha256(cls.db.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if not cls.directory.is_relative_to(CACHE.resolve()):
            raise RuntimeError("Refusing cleanup outside cache")
        cls.temp.cleanup()

    def tearDown(self):
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), self.original_hash)

    def request(self, gold, predicted=None, **overrides):
        request = {"request_id": "test", "db_path": str(self.db), "gold_sql": gold,
                   "predicted_sql": gold if predicted is None else predicted,
                   "query_timeout_seconds": 2.0, "comparison_timeout_seconds": 2.0, "seed": 42}
        request.update(overrides)
        return execute_request(request)

    def assert_match(self, gold, predicted=None, expected=True):
        result = self.request(gold, predicted)
        self.assertEqual(result["status"], "scored", result)
        self.assertIs(result["matched"], expected, result)
        self.assertTrue(result["prediction_executable"])
        self.assertTrue(result["gold"]["row_count_complete"])
        self.assertTrue(result["prediction"]["row_count_complete"])
        self.assertNotIn("rows", result["gold"])
        return result

    def test_three_equivalent_queries(self):
        for gold, predicted in [
            ("SELECT name FROM people WHERE age > 56", "SELECT p.name FROM people AS p WHERE p.age > 56"),
            ("SELECT COUNT(*) FROM people", "SELECT COUNT(id) FROM people"),
            ("SELECT name FROM people WHERE age BETWEEN 56 AND 65", "SELECT name FROM people WHERE age >= 56 AND age <= 65"),
        ]:
            with self.subTest(predicted=predicted):
                self.assert_match(gold, predicted)

    def test_wrong_value_is_not_plugged_from_gold(self):
        self.assert_match("SELECT COUNT(*) FROM people WHERE age > 56",
                          "SELECT COUNT(*) FROM people WHERE age > 60", expected=False)

    def test_duplicate_counts_and_distinct_are_preserved(self):
        self.assert_match("SELECT n FROM duplicates", "SELECT DISTINCT n FROM duplicates", expected=False)
        self.assert_match("SELECT n FROM duplicates", "SELECT n FROM duplicates ORDER BY n DESC")

    def test_order_by_in_gold_matters(self):
        result = self.assert_match("SELECT age FROM people ORDER BY age ASC", "SELECT age FROM people ORDER BY age DESC", expected=False)
        self.assertTrue(result["order_matters"])

    def test_order_heuristic_is_the_upstream_literal_rule(self):
        result = self.request("SELECT 'order by' AS value", "SELECT 'order by'")
        self.assertTrue(result["order_matters"])
        self.assertTrue(result["matched"])

    def test_null_values_and_bytes_stay_typed(self):
        self.assert_match("SELECT n, txt FROM nullable")
        self.assert_match("SELECT NULL", "SELECT ''", expected=False)
        self.assert_match("SELECT x'00ff', NULL, 1.5")
        self.assert_match("SELECT x'61'", "SELECT 'a'", expected=False)

    def test_global_column_permutation_is_allowed(self):
        self.assert_match("SELECT name, age FROM people", "SELECT age, name FROM people")

    def test_two_empty_results_are_upstream_match_even_different_columns(self):
        result = self.assert_match("SELECT name FROM people WHERE age > 200", "SELECT name, age FROM people WHERE age < 0")
        self.assertTrue(result["gold"]["empty_result"])
        self.assertEqual(result["gold"]["column_count"], 1)
        self.assertEqual(result["prediction"]["column_count"], 2)

    def test_sql_value_identifier_and_literal_not_rewritten(self):
        self.assert_match("SELECT value FROM values_data WHERE value = '> ='")
        self.assert_match("SELECT value FROM values_data WHERE value = 'value'")
        result = self.request("SELECT 1", "SELECT 2 > = 1")
        self.assertEqual(result["status"], "prediction_error")

    def test_full_results_are_not_truncated(self):
        result = self.assert_match("SELECT n FROM numbers")
        self.assertEqual(result["gold"]["row_count"], 1500)
        self.assert_match("SELECT n FROM numbers", "SELECT n FROM numbers WHERE n < 1499", expected=False)

    def test_writes_and_attachment_are_rejected(self):
        for sql in ("DELETE FROM people", "DROP TABLE people", "PRAGMA user_version=12",
                    "ATTACH DATABASE ':memory:' AS other", "BEGIN", "SELECT load_extension('x')"):
            with self.subTest(sql=sql):
                result = self.request("SELECT 1", sql)
                self.assertEqual(result["status"], "prediction_error", result)
                self.assertEqual(result["prediction"]["status"], "query_rejected", result)
                self.assertFalse(result["prediction_executable"])

    def test_gold_failure_does_not_skip_prediction(self):
        result = self.request("SELECT missing FROM people", "SELECT name FROM people")
        self.assertEqual(result["status"], "gold_error")
        self.assertIsNone(result["matched"])
        self.assertTrue(result["prediction_executable"])
        self.assertEqual(result["prediction"]["row_count"], 3)

    def test_both_failure_and_blank_prediction(self):
        result = self.request("SELECT missing FROM people", "SELECT absent FROM people")
        self.assertEqual(result["status"], "both_error")
        result = self.request("SELECT 1", "  ")
        self.assertEqual(result["status"], "prediction_error")
        self.assertEqual(result["prediction"]["status"], "sql_error")

    def test_strict_decoding_failure_is_not_ignored(self):
        result = self.request("SELECT 1", "SELECT value FROM invalid_text")
        self.assertEqual(result["status"], "prediction_error")
        self.assertEqual(result["prediction"]["status"], "decode_error")

    def test_query_timeout_and_next_call(self):
        result = self.request("SELECT 1", "SELECT SUM(a.n*b.n*c.n) FROM numbers a, numbers b, numbers c", query_timeout_seconds=0.01)
        self.assertEqual(result["status"], "prediction_error")
        self.assertEqual(result["prediction"]["status"], "query_timeout")
        self.assert_match("SELECT COUNT(*) FROM people")

    def test_missing_database_is_never_created(self):
        path = self.directory / "absent.sqlite"
        result = self.request("SELECT 1", db_path=str(path))
        self.assertEqual(result["status"], "both_error")
        self.assertEqual(result["gold"]["status"], "database_error")
        self.assertFalse(path.exists())

    def test_invalid_requests(self):
        for override in ({"query_timeout_seconds": 0}, {"comparison_timeout_seconds": float("inf")},
                         {"seed": True}, {"db_path": "relative.sqlite"}, {"gold_sql": None}):
            with self.subTest(override=override):
                self.assertEqual(self.request("SELECT 1", **override)["status"], "worker_error")

    def test_jsonl_protocol_continues_after_invalid_json(self):
        request = {"request_id": 7, "db_path": str(self.db), "gold_sql": "SELECT x'00ff'", "predicted_sql": "SELECT x'00ff'"}
        completed = subprocess.run([sys.executable, "-X", "utf8", str(WORKER)], input="not-json\n" + json.dumps(request) + "\n",
                                   text=True, encoding="utf-8", capture_output=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        lines = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["status"], "worker_error")
        self.assertEqual(lines[1]["request_id"], 7)
        self.assertTrue(lines[1]["matched"])

    def test_snapshot_worker_uses_explicit_project_root(self):
        snapshot = self.directory / "snapshot_worker.py"
        shutil.copyfile(WORKER, snapshot)
        environment = dict(os.environ, TEXT2SQL_PROJECT_ROOT=str(ROOT))
        request = {"request_id": "snapshot", "db_path": str(self.db), "gold_sql": "SELECT 1", "predicted_sql": "SELECT 1"}
        completed = subprocess.run([sys.executable, "-X", "utf8", str(snapshot)], input=json.dumps(request) + "\n",
                                   text=True, encoding="utf-8", capture_output=True, timeout=20, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "scored")
        self.assertTrue(result["matched"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
