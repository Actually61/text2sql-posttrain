"""Synthetic tests: fixed selection, subset-only scoring and immutable outputs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_baseline_pilot as pilot

sys.path.insert(0, str(pilot.ROOT / "src/eval"))
import evaluate_baseline_pilot as wrapper
import evaluate_spider as full


def synthetic_population():
    population = {"easy": 138, "medium": 247, "hard": 152, "extra": 88}
    rows, labels = [], {}
    for level, count in population.items():
        for _ in range(count):
            index = len(rows)
            rows.append({"source_index": index, "db_id": f"fixture_{index % 14}", "project_split": "validation"})
            labels[index] = level
    return rows, labels


class BaselinePilotTests(unittest.TestCase):
    def test_largest_remainder_matches_population_proportions(self):
        self.assertEqual(pilot.allocate_counts({"easy": 138, "medium": 247, "hard": 152, "extra": 88}, 32),
                         {"easy": 7, "medium": 13, "hard": 8, "extra": 4})
        self.assertEqual(pilot.allocate_counts(dict.fromkeys(pilot.LEVELS, 1), 2),
                         {"easy": 1, "medium": 1, "hard": 0, "extra": 0})
        for size in (0, 626, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                pilot.allocate_counts({"easy": 138, "medium": 247, "hard": 152, "extra": 88}, size)

    def test_selection_is_seeded_order_independent_and_unique(self):
        rows, labels = synthetic_population()
        original = deepcopy(rows)
        chosen, population, quotas = pilot.choose_rows(rows, labels)
        reordered = rows[:]
        random.Random(81).shuffle(reordered)
        self.assertEqual(chosen, pilot.choose_rows(reordered, labels)[0])
        self.assertEqual(rows, original)
        self.assertEqual(len(chosen), 32)
        self.assertEqual(len({r["source_index"] for r in chosen}), 32)
        self.assertEqual([r["source_index"] for r in chosen], sorted(r["source_index"] for r in chosen))
        self.assertEqual({h: sum(labels[r["source_index"]] == h for r in chosen) for h in pilot.LEVELS}, quotas)
        self.assertEqual(sum(population.values()), 625)
        self.assertNotEqual(chosen, pilot.choose_rows(rows, labels, seed=43)[0])

    def test_repeated_sources_or_incomplete_hardness_rejected(self):
        rows, labels = synthetic_population()
        with self.assertRaises(ValueError):
            pilot.choose_rows(rows + [rows[0]], labels)
        del labels[0]
        with self.assertRaises(ValueError):
            pilot.choose_rows(rows, labels)

    def test_subset_config_does_not_mutate_full_validation_protocol(self):
        original = {"split": "validation", "expected_examples": 625, "denominator_policy": "all 625",
                    "query_timeout_seconds": 2.0, "comparison_timeout_seconds": 2.0,
                    "keep_distinct": True, "plug_value": False, "rewrite_sql": False}
        before = deepcopy(original)
        derived = wrapper.pilot_config(original, {"example_count": 32, "maximum_evaluation_examples": 32})
        self.assertEqual(original, before)
        self.assertEqual(derived["expected_examples"], 32)
        self.assertEqual(derived["split"], "validation_pilot")
        for key in ("query_timeout_seconds", "comparison_timeout_seconds", "keep_distinct", "plug_value", "rewrite_sql"):
            self.assertEqual(derived[key], original[key])
        with self.assertRaises(ValueError):
            wrapper.pilot_config(original, {"example_count": 625, "maximum_evaluation_examples": 625})

    def test_prediction_coverage_is_exactly_the_selected_subset(self):
        rows, labels = synthetic_population()
        selected = pilot.choose_rows(rows, labels)[0]
        predictions = [{"source_index": r["source_index"], "db_id": r["db_id"], "predicted_sql": "SELECT 1"} for r in selected]
        self.assertEqual(len(full.validate_predictions(list(reversed(predictions)), selected)), 32)
        all_predictions = [{"source_index": r["source_index"], "db_id": r["db_id"], "predicted_sql": "SELECT 1"} for r in rows]
        wrong_db = deepcopy(predictions)
        wrong_db[0]["db_id"] = "not_the_database"
        for bad in (predictions[:-1], predictions + [predictions[0]], all_predictions, wrong_db):
            with self.subTest(count=len(bad)), self.assertRaises(ValueError):
                full.validate_predictions(bad, selected)

    def test_published_selection_rejects_changed_manifest_inputs_or_report(self):
        cache = pilot.ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pilot_protocol_test_", dir=cache) as name:
            directory = Path(name).resolve()
            self.assertTrue(directory.is_relative_to(cache.resolve()))
            report = directory / "report.json"
            payload = b'{"source_index":1,"messages":[]}\n'
            expected = {"example_count": 32, "selected_source_indices": list(range(32))}
            original_manifest = full.json_bytes(expected)
            (directory / "manifest.json").write_bytes(original_manifest)
            (directory / "inputs.jsonl").write_bytes(payload)
            report.write_bytes(original_manifest)
            frozen = (expected, payload, [], {"expected_examples": 625}, {})
            with patch.object(pilot, "DIRECTORY", directory), patch.object(pilot, "REPORT", report), patch.object(pilot, "build_expected", return_value=frozen):
                result = pilot.validate_published()
                self.assertEqual(result[0], expected)
                for path in (directory / "manifest.json", directory / "inputs.jsonl", report):
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    with self.subTest(path=path.name), self.assertRaises(ValueError):
                        pilot.validate_published()
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
