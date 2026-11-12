"""CPU-only invariants for pilot inputs and text handling; never loads weights."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

from baseline_pilot import extract_sql, failed_prediction, validate_pilot_inputs


def fixture():
    rows = [
        {"source_index": i, "source_file": "synthetic.json", "db_id": f"db_{i % 4}",
         "project_split": "validation", "messages": [
             {"role": "system", "content": "Return one SQL query."},
             {"role": "user", "content": "Count rows in table t."}]}
        for i in range(32)
    ]
    manifest = {"example_count": 32, "maximum_generation_examples": 32,
                "selected_source_indices": list(range(32)),
                "selected_examples": [{"source_index": r["source_index"], "db_id": r["db_id"]} for r in rows]}
    return manifest, rows


class ExtractionTests(unittest.TestCase):
    def test_literal_semicolon_preserved(self):
        self.assertEqual(extract_sql("  SELECT 'a;b' AS text;  "), "SELECT 'a;b' AS text;")

    def test_complete_fence_only(self):
        for tag in ("sql", "SQL", ""):
            self.assertEqual(extract_sql(f"```{tag}\nSELECT 'a;b';\n```"), "SELECT 'a;b';")
        text = "Here is the SQL:\n```sql\nSELECT 1;\n```"
        self.assertEqual(extract_sql(text), text)
        self.assertEqual(extract_sql("```sql\nSELECT 1;"), "```sql\nSELECT 1;")

    def test_multiple_statements_not_silently_repaired(self):
        sql = "SELECT 1; SELECT 2;"
        self.assertEqual(extract_sql(sql), sql)


class FailureTests(unittest.TestCase):
    def test_failed_prediction_keeps_identity_and_empty_sql(self):
        source = {"source_index": 712, "db_id": "db_a"}
        original = dict(source)
        result = failed_prediction(source, "budget exceeded", "skipped_budget")
        self.assertEqual(source, original)
        self.assertEqual((result["source_index"], result["db_id"]), (712, "db_a"))
        self.assertEqual(result["predicted_sql"], "")
        self.assertFalse(result["generation_ok"])
        self.assertEqual(result["generation_status"], "skipped_budget")
        self.assertEqual(result["new_token_ids"], [])
        self.assertEqual(result["generated_token_count"], 0)


class InputValidationTests(unittest.TestCase):
    def test_valid_32_inputs(self):
        validate_pilot_inputs(*fixture())

    def test_row_and_manifest_count(self):
        manifest, rows = fixture()
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows[:-1])
        for field in ("example_count", "maximum_generation_examples"):
            changed = deepcopy(manifest)
            changed[field] = 33
            with self.assertRaises(ValueError):
                validate_pilot_inputs(changed, rows)

    def test_reordered_or_duplicate_indices_rejected(self):
        manifest, rows = fixture()
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows[::-1])
        rows[1]["source_index"] = rows[0]["source_index"]
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows)

    def test_gold_or_answer_fields_rejected(self):
        for field in ("gold_sql", "query", "completion", "answer"):
            manifest, rows = fixture()
            rows[0][field] = "SELECT 1"
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pilot_inputs(manifest, rows)

    def test_assistant_role_and_extra_message_rejected(self):
        manifest, rows = fixture()
        rows[0]["messages"][1]["role"] = "assistant"
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows)
        manifest, rows = fixture()
        rows[0]["messages"].append({"role": "assistant", "content": "SELECT 1"})
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows)

    def test_message_extra_fields_and_empty_content_rejected(self):
        manifest, rows = fixture()
        rows[0]["messages"][0]["gold_sql"] = "SELECT 1"
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows)
        manifest, rows = fixture()
        rows[0]["messages"][1]["content"] = ""
        with self.assertRaises(ValueError):
            validate_pilot_inputs(manifest, rows)

    def test_wrong_database_and_split_rejected(self):
        for field, value in (("db_id", "other_db"), ("project_split", "final")):
            manifest, rows = fixture()
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pilot_inputs(manifest, rows)


class InstalledGenerationConfigTests(unittest.TestCase):
    def test_current_private_api_and_explicit_greedy_override(self):
        # Exercise installed resolver on a plain object, not a model or tensor.
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin
        holder = SimpleNamespace(generation_config=GenerationConfig(
            do_sample=True, repetition_penalty=1.1, temperature=0.7,
            top_p=0.8, top_k=20, eos_token_id=[151645, 151643]))
        requested = GenerationConfig(do_sample=False, num_beams=1,
            num_return_sequences=1, max_new_tokens=256, repetition_penalty=1.0,
            temperature=1.0, top_p=1.0, top_k=50, max_time=30.0, use_cache=True)
        effective, extra = GenerationMixin._prepare_generation_config(holder, generation_config=requested)
        self.assertFalse(effective.do_sample)
        self.assertEqual(effective.repetition_penalty, 1.0)
        self.assertEqual(effective.max_new_tokens, 256)
        self.assertEqual(effective.max_time, 30.0)
        self.assertEqual(effective.eos_token_id, [151645, 151643])
        self.assertEqual(extra, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
