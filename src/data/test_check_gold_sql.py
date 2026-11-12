"""Controller tests using temporary synthetic data only; no Spider inputs read."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import check_gold_sql as audit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache"


def sample(index, *, split="train", db="sample", question=None, sql="SELECT 1"):
    return {"source_index": index, "source_file": "synthetic.json", "project_split": split,
            "db_id": db, "question": question or f"Question {index}", "query": sql}


def checked(row, status="ok", complete=True):
    return {"source_index": row["source_index"], "status": status,
            "row_count_complete": complete, "attempts": [{"status": status}]}


class FakeClient:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.requests = []

    def call(self, request):
        self.requests.append(deepcopy(request))
        status = next(self.statuses)
        return {"request_id": request["request_id"], "status": status,
                "row_count": 1 if status == "ok" else None,
                "empty_result": False if status == "ok" else None,
                "row_count_complete": status == "ok"}


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CACHE_ROOT.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="gold_controller_test_", dir=CACHE_ROOT)
        cls.root = Path(cls.temporary.name).resolve()
        if not cls.root.is_relative_to(CACHE_ROOT.resolve()):
            raise RuntimeError("Temporary directory is outside project cache")
        cls.worker = cls.root / "fake_worker.py"
        cls.worker.write_text(textwrap.dedent('''\
            import json
            import os
            import sys
            import time
            for line in sys.stdin:
                request = json.loads(line)
                action = request["sql"]
                if action == "hang":
                    time.sleep(30)
                if action == "exit":
                    sys.exit(2)
                if action == "bad_json":
                    print("not JSON", flush=True)
                    continue
                if action == "missing_id":
                    print(json.dumps({"status": "ok"}), flush=True)
                    continue
                if action == "non_dict":
                    print("[]", flush=True)
                    continue
                response = {"request_id": request["request_id"], "status": "ok",
                            "row_count": 1, "column_count": 1, "row_count_complete": True,
                            "empty_result": False, "pid": os.getpid()}
                if action == "wrong_id":
                    response["request_id"] = "different"
                if action == "missing_status":
                    del response["status"]
                if action == "invalid_status":
                    response["status"] = "not_a_status"
                if action == "unhashable_status":
                    response["status"] = []
                print(json.dumps(response), flush=True)
            '''), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        if not cls.root.is_relative_to(CACHE_ROOT.resolve()):
            raise RuntimeError("Refusing cleanup outside project cache")
        cls.temporary.cleanup()

    def client(self, suffix=""):
        return audit.WorkerClient(self.worker, self.root / f"{self._testMethodName}{suffix}.stderr.log", 0.3)

    def request(self, action="ok", index=1):
        return {"request_id": index, "sql": action, "timeout_seconds": 0.05, "decode_mode": "strict"}

    def test_persistent_process_reused_for_normal_requests(self):
        client = self.client()
        try:
            first = client.call(self.request(index=1))
            second = client.call(self.request(index=2))
            self.assertEqual(first["status"], "ok", first)
            self.assertEqual(second["status"], "ok", second)
            self.assertEqual(first["pid"], second["pid"])
            self.assertEqual(second["request_id"], 2)
        finally:
            client.close()

    def test_hard_timeout_kills_process_and_next_call_restarts(self):
        client = self.client()
        try:
            ready = client.call(self.request())
            self.assertEqual(ready["status"], "ok", ready)
            original_process = client.process
            timed_out = client.call(self.request("hang", index=2))
            self.assertEqual(timed_out["status"], "parent_timeout", timed_out)
            self.assertFalse(timed_out["row_count_complete"])
            self.assertIsNotNone(original_process.poll())
            self.assertIsNone(client.process)
            self.assertLess(timed_out["parent_wall_seconds"], 5.0)
            recovered = client.call(self.request(index=3))
            self.assertEqual(recovered["status"], "ok", recovered)
            self.assertNotEqual(recovered["pid"], ready["pid"])
        finally:
            client.close()

    def test_bad_protocol_or_worker_exit_returns_error_and_recovers(self):
        actions = ("bad_json", "wrong_id", "missing_id", "non_dict",
                   "missing_status", "invalid_status", "unhashable_status", "exit")
        for action in actions:
            with self.subTest(action=action):
                client = self.client(action)
                try:
                    bad = client.call(self.request(action))
                    self.assertEqual(bad["status"], "worker_error", bad)
                    self.assertFalse(bad["row_count_complete"])
                    self.assertIsNone(client.process)
                    good = client.call(self.request(index=2))
                    self.assertEqual(good["status"], "ok", good)
                finally:
                    client.close()

    def test_each_timeout_gets_one_larger_budget_retry(self):
        config = {"initial_timeout_seconds": 2, "retry_timeout_seconds": 10}
        for status in ("query_timeout", "parent_timeout"):
            with self.subTest(status=status):
                client = FakeClient([status, "ok"])
                result = audit.audit_one(client, sample(1), config)
                self.assertEqual(result["status"], "ok")
                self.assertEqual([r["timeout_seconds"] for r in client.requests], [2, 10])
                self.assertEqual([a["status"] for a in result["attempts"]], [status, "ok"])

    def test_second_timeout_is_not_retried_forever(self):
        client = FakeClient(["parent_timeout", "query_timeout"])
        result = audit.audit_one(client, sample(1), {"initial_timeout_seconds": 2, "retry_timeout_seconds": 10})
        self.assertEqual(result["status"], "query_timeout")
        self.assertEqual(len(client.requests), 2)

    def test_decode_replacement_remains_quarantinable(self):
        row = sample(1)
        client = FakeClient(["decode_error", "ok"])
        result = audit.audit_one(client, row, {"initial_timeout_seconds": 2, "retry_timeout_seconds": 10})
        self.assertEqual(result["status"], "decode_error")
        self.assertFalse(result["row_count_complete"])
        self.assertEqual([r["decode_mode"] for r in client.requests], ["strict", "replace"])
        keep, quarantine, duplicates = audit.derive_training([row], [result])
        self.assertEqual(keep, [])
        self.assertEqual(len(quarantine), 1)
        self.assertEqual(duplicates, [])

    def test_sql_error_has_no_budget_retry(self):
        client = FakeClient(["sql_error"])
        result = audit.audit_one(client, sample(1), {"initial_timeout_seconds": 2, "retry_timeout_seconds": 10})
        self.assertEqual(result["status"], "sql_error")
        self.assertEqual(len(client.requests), 1)

    def test_exact_dedup_quarantine_and_validation_preservation(self):
        rows = [sample(0, question="same"), sample(1, question="same"),
                sample(2, sql="SELECT missing"), sample(3),
                sample(4, question="same", sql="SELECT 2"),
                sample(5, question="same", db="another"),
                sample(6, question=" same "),
                sample(7, split="validation", db="validation", question="repeated"),
                sample(8, split="validation", db="validation", question="repeated"),
                sample(9, split="validation", db="validation", sql="SELECT missing")]
        before = deepcopy(rows)
        checks = [checked(row) for row in rows]
        checks[2] = checked(rows[2], "sql_error", False)
        checks[3] = checked(rows[3], "ok", False)
        checks[9] = checked(rows[9], "sql_error", False)
        keep, quarantine, duplicates = audit.derive_training(rows, checks)
        self.assertEqual([r["source_index"] for r in keep], [0, 4, 5, 6])
        self.assertEqual([r["source_index"] for r in quarantine], [2, 3])
        self.assertEqual([r["source_index"] for r in duplicates], [1])
        self.assertEqual(duplicates[0]["duplicate_of_source_index"], 0)
        self.assertEqual(rows, before)
        self.assertEqual(sum(len(part) for part in (keep, quarantine, duplicates)), 7)
        self.assertTrue(all(r["project_split"] == "train" for part in (keep, quarantine, duplicates) for r in part))
        self.assertEqual([r for r in rows if r["project_split"] == "validation"], before[7:])

    def test_duplicate_uses_lowest_source_index_conservatively(self):
        first = sample(10, question="same")
        later = sample(20, question="same")
        keep, quarantine, duplicates = audit.derive_training(
            [later, first], [checked(later), checked(first, "query_timeout", False)]
        )
        self.assertEqual(keep, [])
        self.assertEqual([r["source_index"] for r in quarantine], [10])
        self.assertEqual(duplicates[0]["duplicate_of_source_index"], 10)

    def test_input_loader_needs_only_synthetic_train_validation_files(self):
        root = self.root / "synthetic_project"
        split_dir = root / "data/processed/spider_v1"
        split_dir.mkdir(parents=True)
        (root / "reports").mkdir()
        rows = [sample(i, split="train" if i < 6375 else "validation",
                       db="train_db" if i < 6375 else "validation_db") for i in range(7000)]
        for split in ("train", "validation"):
            audit.write_jsonl(split_dir / f"{split}.jsonl", [r for r in rows if r["project_split"] == split])
        source_hashes, download_files = {}, []
        for db_id in ("train_db", "validation_db"):
            path = root / "data/raw/spider/database" / db_id / f"{db_id}.sqlite"
            path.parent.mkdir(parents=True)
            # load_inputs verifies bytes/hashes only; it never executes these fixtures.
            path.write_bytes(b"synthetic database identity only")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source_hashes[path.relative_to(root).as_posix()] = digest
            download_files.append({"path": f"database/{db_id}/{db_id}.sqlite", "sha256": digest, "size": path.stat().st_size})
        download_path = root / "download_manifest.json"
        audit.write_json(download_path, {"files": download_files})
        manifest_path = split_dir / "split_manifest.json"
        audit.write_json(manifest_path, {
            "download_provenance": {"manifest_path": "download_manifest.json", "manifest_sha256": audit.sha256(download_path)},
            "source_sha256": source_hashes,
            "splits": {"train": {"db_ids": ["train_db"]}, "validation": {"db_ids": ["validation_db"]},
                       "final_eval": {"db_ids": ["absent_protected_database"]}},
        })
        audit.write_json(root / "reports/spider_split_reproducibility.json", {
            "outputs_sha256": {p.name: audit.sha256(p) for p in split_dir.iterdir()}
        })
        original_hashes = {p: audit.sha256(p) for p in root.rglob("*") if p.is_file()}
        with patch.object(audit, "ROOT", root), patch.object(audit, "SPLIT_DIR", split_dir):
            loaded, hashes = audit.load_inputs()
        self.assertEqual(len(loaded), 7000)
        self.assertEqual(loaded, rows)
        self.assertFalse(any("absent_protected_database" in name for name in hashes))
        self.assertEqual({p: audit.sha256(p) for p in original_hashes}, original_hashes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
