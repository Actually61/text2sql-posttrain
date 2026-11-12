"""小型合成文件验证划分和来源检查，不接触正式Spider原始文件。"""

import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import check_spider as checker


class DataChecks(unittest.TestCase):
    def setUp(self):
        cache = checker.PROJECT_ROOT / ".cache" / "data_check_tests"
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fixture_", dir=cache)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(cache.resolve())
        self.addCleanup(self.temp.cleanup)
        self.raw = self.root / "data/raw/spider"
        self.raw.mkdir(parents=True)
        schemas = []
        for db_id in ["a", "b", "c", "d", "final_db"]:
            folder = self.raw / "database" / db_id
            folder.mkdir(parents=True)
            connection = sqlite3.connect(folder / f"{db_id}.sqlite")
            try:
                connection.execute("CREATE TABLE item(id INTEGER PRIMARY KEY, name TEXT)")
            finally:
                connection.close()
            columns = [[-1, "*"], [0, "id"], [0, "name"]]
            schemas.append({"db_id": db_id, "table_names_original": ["item"], "table_names": ["item"],
                "column_names_original": columns, "column_names": columns, "column_types": ["text", "number", "text"],
                "primary_keys": [1], "foreign_keys": []})
        def rows(ids):
            return [{"db_id": db, "question": f"List item {i}", "query": f"SELECT id FROM item WHERE id={i}",
                     "sql": {"select": []}} for db in ids for i in (1, 2)]
        for name, value in [("train_spider.json", rows(["a", "b", "c", "d"])),
                            ("dev.json", rows(["final_db"])), ("tables.json", schemas)]:
            (self.raw / name).write_text(json.dumps(value), encoding="utf-8")
        self.config = {"raw_dir": "data/raw/spider", "download_manifest": "reports/download.json",
            "train_file": "train_spider.json", "final_file": "dev.json", "schema_file": "tables.json",
            "database_dir": "database", "expected_train_examples": 8, "expected_final_examples": 2,
            "split_seed": 42, "validation_db_fraction": 0.25, "processed_dir": "data/processed/v1"}
        manifest = {"stage": "extracted", "raw_dir": self.config["raw_dir"], "archive_path": "synthetic.zip",
            "archive_sha256": "synthetic_fixture_not_a_real_download", "source_url": "synthetic_fixture",
            "files": [{"path": p.relative_to(self.raw).as_posix(), "size": p.stat().st_size, "sha256": checker.sha256(p)}
                      for p in sorted(self.raw.rglob("*")) if p.is_file()]}
        (self.root / "reports").mkdir()
        (self.root / self.config["download_manifest"]).write_text(json.dumps(manifest), encoding="utf-8")

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return checker.check_data(self.config, self.root)

    def test_disjoint_reproducible_and_final_reference_only(self):
        report, data = self.check()
        self.assertTrue(report["passed"])
        self.assertEqual(report["download_provenance"]["matched_file_count"], 8)
        artifacts, manifest = checker.build_artifacts(self.config, report, data, self.root)
        again, _ = checker.build_artifacts(self.config, report, data, self.root)
        self.assertEqual(artifacts, again)
        self.assertEqual(manifest["splits"]["train"]["example_count"], 6)
        self.assertEqual(manifest["splits"]["validation"]["example_count"], 2)
        self.assertNotIn("final_eval.jsonl", artifacts)
        final = json.loads(artifacts["final_eval_manifest.json"])
        self.assertFalse(final["labels_copied_to_processed"])
        self.assertTrue(all(not value["db_id_overlap"] for value in manifest["cross_split_checks"].values()))
        self.assertGreater(manifest["cross_split_checks"]["train_vs_validation"]["normalized_question_overlap_groups"], 0)

    def test_post_extraction_label_change_is_rejected(self):
        path = self.raw / "train_spider.json"
        rows = json.loads(path.read_text())
        rows[0]["question"] = "Changed but structurally valid question"
        path.write_text(json.dumps(rows), encoding="utf-8")
        report, data = self.check()
        self.assertFalse(report["passed"])
        self.assertIsNone(data)
        self.assertTrue(any("ZIP解压清单不一致" in error for error in report["fatal_errors"]))

    def test_sqlite_content_change_is_rejected(self):
        path = self.raw / "database/a/a.sqlite"
        connection = sqlite3.connect(path)
        try:
            connection.execute("INSERT INTO item VALUES(1, 'changed')")
            connection.commit()
        finally:
            connection.close()
        report, _ = self.check()
        self.assertFalse(report["passed"])
        self.assertTrue(any("ZIP解压清单不一致" in error for error in report["fatal_errors"]))

    def test_invalid_primary_key_index_is_rejected(self):
        path = self.raw / "tables.json"
        schemas = json.loads(path.read_text())
        schemas[0]["primary_keys"] = [0]
        path.write_text(json.dumps(schemas), encoding="utf-8")
        report, _ = self.check()
        self.assertFalse(report["passed"])
        self.assertTrue(any("primary_keys" in error for error in report["fatal_errors"]))

    def test_wrong_expected_count_is_rejected(self):
        self.config["expected_train_examples"] = 9
        report, _ = self.check()
        self.assertFalse(report["passed"])

    def test_existing_different_split_is_not_overwritten(self):
        target = self.root / "processed"
        checker.publish_immutable({"one.json": "first"}, target)
        checker.publish_immutable({"one.json": "first"}, target)
        with self.assertRaises(ValueError):
            checker.publish_immutable({"new.json": "new", "one.json": "changed"}, target)
        self.assertEqual((target / "one.json").read_text(), "first")
        self.assertFalse((target / "new.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
