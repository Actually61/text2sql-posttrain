"""Small CPU-only input/protocol tests; no weights, generation, SQL or CUDA calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import baseline_full as full


def fixture():
    db_ids = [f"db_{i:02d}" for i in range(14)]
    config = {"expected_examples": 625, "expected_database_count": 14, "expected_database_ids": db_ids}
    manifest = {"validation_examples": 625}
    rows = [{"source_index": i * 2, "source_file": "data/raw/spider/train_spider.json",
             "db_id": db_ids[i % 14], "project_split": "validation", "messages": [
                 {"role": "system", "content": "Return one SQL query."},
                 {"role": "user", "content": "Table t has column n. Count its rows."}]} for i in range(625)]
    return config, manifest, rows


class FullBaselineTests(unittest.TestCase):
    def test_valid_complete_inputs_are_preserved(self):
        config, manifest, rows = fixture()
        original = deepcopy(rows)
        indices, db_ids = full.validate_full_inputs(config, manifest, rows)
        self.assertEqual(indices, list(range(0, 1250, 2)))
        self.assertEqual(db_ids, config["expected_database_ids"])
        self.assertEqual(rows, original)

    def test_count_order_and_duplicate_ids_rejected(self):
        config, manifest, rows = fixture()
        invalid_sets = [rows[:-1], rows[::-1]]
        duplicate = deepcopy(rows)
        duplicate[1]["source_index"] = duplicate[0]["source_index"]
        invalid_sets.append(duplicate)
        for changed in invalid_sets:
            with self.subTest(count=len(changed)), self.assertRaises(ValueError):
                full.validate_full_inputs(config, manifest, changed)
        with self.assertRaises(ValueError):
            full.validate_full_inputs(config, {"validation_examples": 624}, rows)

    def test_wrong_split_source_database_and_gold_fields_rejected(self):
        for field, value in (("project_split", "final"), ("source_file", "data/raw/spider/dev.json"),
                             ("db_id", "unexpected_db"), ("source_index", True),
                             ("gold_sql", "SELECT 1"), ("completion", [{"role": "assistant", "content": "SELECT 1"}])):
            config, manifest, rows = fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                full.validate_full_inputs(config, manifest, rows)

    def test_assistant_or_extra_message_fields_rejected(self):
        for change in ("assistant", "gold_field", "empty_content", "third_message"):
            config, manifest, rows = fixture()
            if change == "assistant":
                rows[0]["messages"][1]["role"] = "assistant"
            elif change == "gold_field":
                rows[0]["messages"][1]["query"] = "SELECT 1"
            elif change == "empty_content":
                rows[0]["messages"][1]["content"] = "   "
            else:
                rows[0]["messages"].append({"role": "assistant", "content": "SELECT 1"})
            with self.subTest(change=change), self.assertRaises(ValueError):
                full.validate_full_inputs(config, manifest, rows)

    def test_effective_generation_config_matches_recorded_pilot(self):
        # Instantiate only metadata classes on a plain holder, never a model.
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin
        config = full.read_json(full.CONFIG)
        pilot = full.read_json(full.project_path(config["pilot_generation_report"]))
        defaults = GenerationConfig.from_pretrained(
            full.ROOT / "models/qwen2.5-coder-1.5b-instruct", local_files_only=True)
        pad = pilot["generation_config"]["pad_token_id"]
        requested = full.make_generation_config(config["generation"], defaults, pad)
        effective, extra = GenerationMixin._prepare_generation_config(
            SimpleNamespace(generation_config=defaults), generation_config=requested)
        full.verify_effective_config(effective, pilot)
        self.assertEqual(extra, {})
        self.assertEqual(effective.num_return_sequences, 1)
        self.assertFalse(effective.do_sample)
        self.assertEqual(effective.max_new_tokens, 256)
        self.assertEqual(effective.max_time, 30.0)
        changed = deepcopy(effective)
        changed.max_new_tokens = 257
        with self.assertRaises(ValueError):
            full.verify_effective_config(changed, pilot)

    def test_changed_full_vs_pilot_protocol_rejected_before_data_or_model(self):
        config = full.read_json(full.CONFIG)
        pilot = full.read_json(full.project_path(config["pilot_generation_report"]))
        changes = [("seed", 43), ("batch_size", 2), ("retries_per_example", 1),
                   ("warmup_generations", 1), ("attention_implementation", "eager")]
        settings = deepcopy(config["generation"])
        settings["num_return_sequences"] = 2
        changes.append(("generation", settings))
        for field, value in changes:
            changed = deepcopy(config)
            changed[field] = value
            # The early guard must fail before package/data/tokenizer/weight reads.
            with self.subTest(field=field), patch.object(full, "read_json", side_effect=[changed, pilot]), \
                    patch.object(full, "verify_hashes"), \
                    patch.object(full, "version", side_effect=AssertionError("Must fail before runtime checks")), \
                    self.assertRaisesRegex(ValueError, "changed the pilot inference protocol"):
                full.preflight()

    def test_failure_records_keep_all_identities_and_no_fake_predictions(self):
        _, _, sources = fixture()
        failures = [full.record_failure(row, "controlled budget exhaustion", "skipped_budget") for row in sources]
        self.assertEqual(len(failures), 625)
        self.assertEqual([r["source_index"] for r in failures], [r["source_index"] for r in sources])
        self.assertTrue(all(r["predicted_sql"] == "" and r["generation_ok"] is False for r in failures))
        self.assertTrue(all(r["generated_token_count"] == 0 and r["new_token_ids"] == [] for r in failures))
        self.assertTrue(all(r["peak_allocated_mib"] is None for r in failures))

    def test_protected_evidence_hash_change_rejected(self):
        cache = full.ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="baseline_full_test_", dir=cache)
        directory = Path(temporary.name).resolve()
        if not directory.is_relative_to(cache.resolve()):
            raise RuntimeError("Test directory escaped project cache")
        try:
            path = directory / "synthetic_evidence.txt"
            path.write_bytes(b"fixed evidence")
            expected = {full.relative(path): hashlib.sha256(b"fixed evidence").hexdigest()}
            full.verify_hashes(expected, "Protected evidence")
            path.write_bytes(b"changed evidence")
            with self.assertRaisesRegex(ValueError, "Protected evidence changed"):
                full.verify_hashes(expected, "Protected evidence")
        finally:
            if not directory.is_relative_to(cache.resolve()):
                raise RuntimeError("Refusing cleanup outside project cache")
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
