"""Synthetic controller tests; no real Spider data, model or final set access."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

import evaluate_spider as evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"


def rows():
    return [{"source_index": index, "db_id": db, "query": f"SELECT {index}", "project_split": "validation"}
            for db, indices in (("db_a", (1, 2, 3, 4)), ("db_b", (10, 11, 12))) for index in indices]


def result(status="scored", matched=True, executable=True):
    return {"status": status, "matched": matched, "prediction_executable": executable,
            "gold": {"status": "ok", "empty_result": False},
            "prediction": {"status": "ok" if executable else "sql_error", "empty_result": False}}


class EvaluationControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CACHE_ROOT.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="eval_controller_test_", dir=CACHE_ROOT)
        cls.root = Path(cls.temporary.name).resolve()
        if not cls.root.is_relative_to(CACHE_ROOT.resolve()):
            raise RuntimeError("Temporary test path escaped project cache")
        cls.worker = cls.root / "fake_pair_worker.py"
        cls.worker.write_text(textwrap.dedent('''\
            import json
            import os
            import sys
            import time
            for line in sys.stdin:
                request = json.loads(line)
                action = request["predicted_sql"]
                if action == "hang":
                    time.sleep(30)
                if action == "exit":
                    sys.exit(3)
                if action == "bad_json":
                    print("not-json", flush=True)
                    continue
                if action == "non_dict":
                    print("[]", flush=True)
                    continue
                response = {"request_id": request["request_id"], "status": "scored", "matched": True,
                            "prediction_executable": True, "pid": os.getpid(), "python": sys.executable,
                            "gold": {"status": "ok", "empty_result": False},
                            "prediction": {"status": "ok", "empty_result": False}}
                if action == "dependencies":
                    import sqlparse
                    response["sqlparse_file"] = sqlparse.__file__
                if action == "missing_id":
                    del response["request_id"]
                if action == "wrong_id":
                    response["request_id"] = "wrong"
                if action == "missing_status":
                    del response["status"]
                if action == "bad_status":
                    response["status"] = []
                if action == "unknown_status":
                    response["status"] = "invented"
                if action == "matched_number":
                    response["matched"] = 1
                if action == "missing_executable":
                    del response["prediction_executable"]
                if action == "error_missing_matched":
                    response["status"] = "prediction_error"
                    response["prediction_executable"] = False
                    del response["matched"]
                if action == "error_matched_true":
                    response["status"] = "prediction_error"
                if action == "gold_missing_status":
                    response["gold"] = {}
                print(json.dumps(response), flush=True)
            '''), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        if not cls.root.is_relative_to(CACHE_ROOT.resolve()):
            raise RuntimeError("Refusing cleanup outside project cache")
        cls.temporary.cleanup()

    def client(self, suffix=""):
        return evaluator.PairClient(self.worker, self.root / f"{self._testMethodName}{suffix}.stderr.log", 0.5)

    def request(self, action="ok", index=1):
        return {"request_id": index, "predicted_sql": action, "query_timeout_seconds": 0.1,
                "comparison_timeout_seconds": 0.1}

    def test_prediction_order_does_not_change_id_alignment(self):
        expected = rows()
        predictions, _ = evaluator.make_control_predictions(expected, 42)
        aligned = evaluator.validate_predictions(list(reversed(predictions)), expected)
        self.assertEqual(set(aligned), {r["source_index"] for r in expected})
        for row in expected:
            self.assertEqual(aligned[row["source_index"]]["predicted_sql"], row["query"])

    def test_missing_duplicate_unknown_and_invalid_ids_are_rejected(self):
        expected = rows()
        good, _ = evaluator.make_control_predictions(expected, 42)
        cases = [good[:-1], good + [good[0]], good + [{"source_index": 999, "db_id": "db_a", "predicted_sql": "SELECT 1"}]]
        for index in (True, 1.0, None):
            modified = deepcopy(good)
            modified[0]["source_index"] = index
            cases.append(modified)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluator.validate_predictions(case, expected)

    def test_wrong_database_and_non_string_sql_are_rejected(self):
        expected = rows()
        good, _ = evaluator.make_control_predictions(expected, 42)
        for field, value in (("db_id", "db_b"), ("predicted_sql", None), ("predicted_sql", 123)):
            modified = deepcopy(good)
            modified[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluator.validate_predictions(modified, expected)
        # Empty output remains an attempted prediction to be scored as an error.
        good[0]["predicted_sql"] = ""
        self.assertEqual(evaluator.validate_predictions(good, expected)[1]["predicted_sql"], "")

    def test_same_database_shuffle_is_deterministic_and_has_no_fixed_indices(self):
        original = rows()
        copied = deepcopy(original)
        _, first = evaluator.make_control_predictions(original, 42)
        _, second = evaluator.make_control_predictions(list(reversed(original)), 42)
        self.assertEqual(original, copied)
        self.assertEqual(first, second)
        lookup = {r["source_index"]: r for r in original}
        self.assertEqual({p["donor_source_index"] for p in first}, set(lookup))
        for prediction in first:
            donor = lookup[prediction["donor_source_index"]]
            self.assertNotEqual(prediction["source_index"], prediction["donor_source_index"])
            self.assertEqual(prediction["db_id"], donor["db_id"])
            self.assertEqual(prediction["predicted_sql"], donor["query"])
        # The procedure guarantees different source IDs, not different SQL text.
        repeated_sql = deepcopy(original)
        for row in repeated_sql:
            row["query"] = "SELECT 1"
        _, shuffled = evaluator.make_control_predictions(repeated_sql, 42)
        self.assertTrue(all(p["predicted_sql"] == "SELECT 1" for p in shuffled))

    def test_shuffle_singleton_database_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluator.make_control_predictions(rows()[:1], 42)

    def test_errors_and_timeout_remain_in_full_denominator(self):
        scored = [result() for _ in range(623)]
        scored.append(result("prediction_error", None, False))
        timeout = result("parent_timeout", None, None)
        timeout["gold"] = timeout["prediction"] = None
        scored.append(timeout)
        summary = evaluator.aggregate(scored)
        self.assertEqual(summary["count"], 625)
        self.assertEqual(summary["execution_correct"], 623)
        self.assertEqual(summary["execution_accuracy"], 623/625)
        self.assertEqual(summary["executable_predictions"], 623)
        self.assertEqual(summary["prediction_executable_rate"], 623/625)
        self.assertEqual(summary["prediction_execution_unknown"], 1)
        self.assertEqual(summary["status_counts"]["parent_timeout"], 1)

    def test_persistent_worker_uses_real_python_and_venv_dependencies(self):
        client = self.client()
        try:
            first = client.call(self.request("dependencies"))
            second = client.call(self.request(index=2))
            self.assertEqual(first["status"], "scored", first)
            self.assertEqual(second["status"], "scored", second)
            self.assertEqual(first["pid"], second["pid"])
            # Store Python can report its AppData execution alias in
            # sys.executable. PID identity proves no intermediate venv launcher.
            self.assertEqual(first["pid"], client.process.pid)
            self.assertEqual(Path(client.process.args[0]).resolve(), Path(evaluator.WORKER_PYTHON).resolve())
            self.assertTrue(Path(first["sqlparse_file"]).resolve().is_relative_to(Path(evaluator.PURELIB).resolve()))
        finally:
            client.close()

    def test_hard_timeout_kills_actual_worker_and_next_request_recovers(self):
        client = self.client()
        try:
            ready = client.call(self.request())
            self.assertEqual(ready["status"], "scored", ready)
            old_process = client.process
            timed_out = client.call(self.request("hang", index=2))
            self.assertEqual(timed_out["status"], "parent_timeout", timed_out)
            self.assertIsNone(timed_out["matched"])
            self.assertIsNone(timed_out["prediction_executable"])
            self.assertIsNotNone(old_process.poll())
            self.assertIsNone(client.process)
            self.assertLess(timed_out["parent_wall_seconds"], 5)
            recovered = client.call(self.request(index=3))
            self.assertEqual(recovered["status"], "scored", recovered)
            self.assertNotEqual(recovered["pid"], ready["pid"])
        finally:
            client.close()

    def test_bad_protocol_and_worker_exit_are_errors_and_recover(self):
        for action in ("bad_json", "non_dict", "missing_id", "wrong_id", "missing_status", "bad_status",
                       "unknown_status", "matched_number", "missing_executable", "error_missing_matched",
                       "error_matched_true", "gold_missing_status", "exit"):
            with self.subTest(action=action):
                client = self.client(action)
                try:
                    bad = client.call(self.request(action))
                    self.assertEqual(bad["status"], "worker_error", bad)
                    self.assertIsNone(bad["matched"])
                    self.assertIsNone(bad["prediction_executable"])
                    recovered = client.call(self.request(index=2))
                    self.assertEqual(recovered["status"], "scored", recovered)
                finally:
                    client.close()

    def test_modified_frozen_source_hash_is_rejected(self):
        source = self.root / "synthetic_source.json"
        source.write_text('{"value":1}\n', encoding="utf-8")
        frozen = evaluator.digest(source)
        self.assertEqual(evaluator.expect_hash(source, frozen), frozen)
        source.write_text('{"value":2}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            evaluator.expect_hash(source, frozen)


if __name__ == "__main__":
    unittest.main(verbosity=2)
