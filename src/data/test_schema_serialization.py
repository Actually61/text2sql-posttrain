"""Synthetic schema tests; never load Spider datasets or databases."""
from __future__ import annotations

import copy
import unittest

from schema_serialization import serialize_schema, validate_schema


def example_schema():
    return {
        "db_id": "teaching_example",
        "table_names_original": ["Department", "Employee"],
        "table_names": ["department display name", "employee display name"],
        "column_names_original": [
            [-1, "*"], [0, "Dept_ID"], [0, "Dept_Name"],
            [1, "Employee_ID"], [1, "Department_ID"], [1, "Manager_ID"],
        ],
        "column_names": [[-1, "*"], [0, "display id"]],
        "column_types": ["text", "number", "text", "number", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1], [5, 3]],
    }


class SchemaSerializationTests(unittest.TestCase):
    def test_original_names_order_and_metadata_types(self):
        schema = example_schema()
        rendered = serialize_schema(schema)
        self.assertTrue(rendered.startswith("Database: teaching_example\n"))
        self.assertIn('Table "Department":', rendered)
        self.assertIn('"Dept_Name" [text]', rendered)
        self.assertIn('"Employee_ID" [number]', rendered)
        self.assertLess(rendered.index('Table "Department":'), rendered.index('Table "Employee":'))
        self.assertLess(rendered.index('"Dept_ID" [number]'), rendered.index('"Dept_Name" [text]'))
        self.assertNotIn("department display name", rendered)
        self.assertNotIn("display id", rendered)
        self.assertIn("not exact SQLite DDL types", rendered)

    def test_placeholder_not_rendered_and_indices_not_shifted(self):
        rendered = serialize_schema(example_schema())
        self.assertNotIn('"*"', rendered)
        self.assertIn('Primary key columns: "Dept_ID"', rendered)
        self.assertIn('Primary key columns: "Employee_ID"', rendered)
        self.assertIn('"Department_ID" -> "Department"."Dept_ID"', rendered)
        self.assertNotIn('"Manager_ID" -> "Department"."Dept_Name"', rendered)

    def test_multiple_primary_columns_and_self_reference(self):
        schema = example_schema()
        schema["primary_keys"] = [1, 3, 4]
        rendered = serialize_schema(schema)
        self.assertIn('Primary key columns: "Employee_ID", "Department_ID"', rendered)
        self.assertIn('"Manager_ID" -> "Employee"."Employee_ID"', rendered)
        self.assertEqual(rendered.count(" -> "), 2)
        self.assertNotIn("UNIQUE", rendered)
        self.assertNotIn("CREATE TABLE", rendered)

    def test_missing_keys_are_explicit_not_guessed(self):
        schema = example_schema()
        schema["primary_keys"] = []
        schema["foreign_keys"] = []
        rendered = serialize_schema(schema)
        self.assertEqual(rendered.count("Primary key columns: (none recorded)"), 2)
        self.assertEqual(rendered.count("Foreign keys: (none recorded)"), 2)

    def test_embedded_quotes_spaces_and_non_ascii_names(self):
        schema = example_schema()
        schema["table_names_original"][0] = 'Research "部门"'
        schema["column_names_original"][1][1] = 'Dept "Key"'
        rendered = serialize_schema(schema)
        self.assertIn('Table "Research ""部门""":', rendered)
        self.assertIn('"Dept ""Key""" [number]', rendered)
        self.assertIn('"Department_ID" -> "Research ""部门"""."Dept ""Key"""', rendered)

    def test_only_schema_fields_and_no_mutation(self):
        schema = example_schema()
        schema["query"] = "SELECT SECRET_GOLD_MARKER FROM Employee"
        schema["question"] = "SECRET_QUESTION_MARKER"
        schema["rows"] = [["SECRET_DATABASE_VALUE_MARKER"]]
        before = copy.deepcopy(schema)
        rendered = serialize_schema(schema)
        self.assertNotIn("SECRET_", rendered)
        self.assertEqual(schema, before)
        self.assertEqual(rendered, serialize_schema(copy.deepcopy(schema)))

    def test_invalid_required_fields(self):
        for bad in (None, [], {}, {"db_id": "test"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_schema(bad)
        for field, value in (("db_id", ""), ("table_names_original", []),
                             ("table_names_original", [None]), ("column_names_original", []),
                             ("primary_keys", None), ("foreign_keys", None)):
            schema = example_schema()
            schema[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                serialize_schema(schema)

    def test_invalid_column_shape_owner_and_placeholder(self):
        for index, bad_column in ((0, [0, "*"]), (0, [-1, "id"]),
                                  (1, [-1, "*"]), (1, [2, "id"]),
                                  (1, [True, "id"]), (1, [0]), (1, [0, None])):
            schema = example_schema()
            schema["column_names_original"][index] = bad_column
            with self.subTest(index=index, bad=bad_column), self.assertRaises(ValueError):
                serialize_schema(schema)

    def test_invalid_primary_key_indices(self):
        for bad_index in (0, -1, 6, True, 1.0, "1"):
            schema = example_schema()
            schema["primary_keys"] = [bad_index]
            with self.subTest(index=bad_index), self.assertRaises(ValueError):
                serialize_schema(schema)

    def test_invalid_foreign_key_indices_and_pairs(self):
        for bad_pair in ([0, 1], [4, 0], [-1, 1], [4, 6], [True, 1], [4, "1"], [4], None):
            schema = example_schema()
            schema["foreign_keys"] = [bad_pair]
            with self.subTest(pair=bad_pair), self.assertRaises(ValueError):
                serialize_schema(schema)

    def test_type_alignment_and_no_invented_type_mapping(self):
        schema = example_schema()
        schema["column_types"][2] = "others"
        rendered = serialize_schema(schema)
        self.assertIn('"Dept_Name" [others]', rendered)
        self.assertNotIn("BLOB", rendered)
        for bad_types in (["number"], None, [None] * 6, [""] * 6):
            schema["column_types"] = bad_types
            with self.subTest(types=bad_types), self.assertRaises(ValueError):
                serialize_schema(schema)


if __name__ == "__main__":
    unittest.main(verbosity=2)
