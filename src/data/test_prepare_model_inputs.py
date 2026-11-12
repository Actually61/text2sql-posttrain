"""Small synthetic checks for prompt separation, length accounting and freezing.

No Spider examples, final data, network, or model weights are read here.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_model_inputs as module


class FakeTokenizer:
    """Character tokens plus explicit chat boundaries, sufficient for invariants."""
    eos_token_id = 900001

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        text = "".join(f"<start>{row['role']}\n{row['content']}<end>\n" for row in messages)
        if add_generation_prompt:
            text += "<start>assistant\n"
        if not tokenize:
            return text
        # Match text encoding for preview's independent tokenization check while
        # retaining one distinguishable EOS ID in each completed message.
        return self.encode(text, add_special_tokens=False)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("The caller must not add special tokens twice")
        parts = text.split("<end>")
        ids = []
        for index, part in enumerate(parts):
            if index:
                ids.append(self.eos_token_id)
            ids.extend(ord(char) for char in part)
        return ids


PROMPT = {"system": "Return a SQLite query.", "user_template": "Schema:\n{schema}\nQuestion:\n{question}"}


def schema(db_id):
    return {"db_id": db_id, "table_names_original": ["Records"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "column_types": ["text", "number", "text"], "primary_keys": [1], "foreign_keys": []}


def row(index, split, db_id, query):
    return {"source_index": index, "source_file": "synthetic_only.json", "db_id": db_id,
            "project_split": split, "question": f"Synthetic question {index}", "query": query}


def length_row(index, prompt, full, sql=2, suffix=4):
    return {"source_index": index, "db_id": "synthetic", "schema_tokens": 2,
            "question_tokens": 3, "prompt_tokens": prompt, "full_reference_tokens": full,
            "sql_text_tokens": sql, "assistant_suffix_tokens": suffix}


class PrepareInputsTests(unittest.TestCase):
    def test_prompt_independent_of_target_and_keeps_question_text(self):
        original = row(0, "train", "synthetic", "SELECT secret_A")
        changed = {**original, "query": "SELECT secret_B"}
        first = module.build_messages("table structure", original["question"], PROMPT)
        second = module.build_messages("table structure", changed["question"], PROMPT)
        self.assertEqual(first, second)
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertNotIn("secret_A", json.dumps(first))
        question = 'Keep {braces}, "quotes", and\nline breaks'
        self.assertIn(question, module.build_messages("schema", question, PROMPT)[1]["content"])
        with self.assertRaises(ValueError):
            module.build_messages("schema", "  ", PROMPT)

    def test_full_suffix_counts_sql_eos_and_trailing_newline(self):
        tokenizer = FakeTokenizer()
        messages = module.build_messages("schema", "question", PROMPT)
        result = module.token_lengths(tokenizer, messages, "SELECT 1")
        self.assertEqual(result["sql_text_tokens"], len("SELECT 1"))
        self.assertEqual(result["assistant_suffix_tokens"], len("SELECT 1") + 2)
        self.assertEqual(result["full_reference_tokens"], result["prompt_tokens"] + result["assistant_suffix_tokens"])

    def test_prefix_mismatch_is_rejected(self):
        class Misaligned(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                if messages[-1]["role"] == "assistant":
                    ids[0] = -1
                return ids
        with self.assertRaisesRegex(ValueError, "prefix"):
            module.token_lengths(Misaligned(), module.build_messages("s", "q", PROMPT), "SELECT 1")

    def test_missing_assistant_eos_is_rejected(self):
        class MissingEOS(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                if messages[-1]["role"] == "assistant":
                    ids[-2] = 42
                return ids
        with self.assertRaisesRegex(ValueError, "EOS"):
            module.token_lengths(MissingEOS(), module.build_messages("s", "q", PROMPT), "SELECT 1")

    def test_nearest_rank_percentiles_and_singleton(self):
        values = module.distribution(list(range(1, 21)))
        self.assertEqual([values[k] for k in ("p50", "p90", "p95", "p99")], [10, 18, 19, 20])
        self.assertEqual(module.distribution([7])["p99"], 7)
        with self.assertRaises(ValueError):
            module.distribution([])

    def test_budget_equal_is_covered_and_reserve_is_separate(self):
        config = {"length_budgets": [10], "generation_reserve_tokens": 3}
        rows = [length_row(1, 7, 10), length_row(2, 8, 11), length_row(3, 10, 10)]
        summary = module.summarize(rows, config)
        budget = summary["budgets"][0]
        self.assertEqual(budget["prompt_over_source_indices"], [])
        self.assertEqual(budget["prompt_plus_reserve_over_source_indices"], [2, 3])
        self.assertEqual(budget["full_reference_over_source_indices"], [2])
        self.assertEqual(budget["prompt_plus_reserve_coverage_percent"], 33.333)
        self.assertEqual(budget["full_reference_coverage_percent"], 66.667)

    def test_artifacts_keep_all_rows_and_separate_targets(self):
        config = {"version": "synthetic", "schema_format": "test", "length_budgets": [1, 4096],
                  "generation_reserve_tokens": 256}
        rows = {"train": [row(0, "train", "a", "SELECT secret_A"), row(1, "train", "a", "SELECT secret_B"),
                          row(2, "train", "b", "SELECT secret_C")],
                "validation": [row(3, "validation", "c", "SELECT secret_D"), row(4, "validation", "c", "SELECT secret_E")]}
        before = copy.deepcopy(rows)
        with redirect_stdout(io.StringIO()):
            payloads, stats, _ = module.build_artifacts(config, rows, {name: schema(name) for name in "abc"},
                {"model_id": "synthetic", "revision": "test"}, PROMPT, {}, FakeTokenizer())
        self.assertEqual(rows, before)
        self.assertNotIn("validation_sft.jsonl", payloads)
        self.assertEqual(stats["schema_database_count"], 3)
        for split, expected in (("train", 3), ("validation", 2)):
            inputs = [json.loads(line) for line in payloads[f"{split}_inputs.jsonl"].decode().splitlines()]
            refs = [json.loads(line) for line in payloads[f"{split}_references.jsonl"].decode().splitlines()]
            lengths = [json.loads(line) for line in payloads[f"{split}_lengths.jsonl"].decode().splitlines()]
            self.assertEqual((len(inputs), len(refs), len(lengths)), (expected, expected, expected))
            self.assertEqual(stats["splits"][split]["count"], expected)
            for source, item, ref in zip(rows[split], inputs, refs):
                self.assertEqual(set(item), {"source_index", "source_file", "db_id", "project_split", "messages"})
                self.assertEqual([m["role"] for m in item["messages"]], ["system", "user"])
                self.assertNotIn("secret_", json.dumps(item))
                self.assertEqual(ref["gold_sql"], source["query"])
                self.assertEqual(ref["source_index"], item["source_index"])
        sft = [json.loads(line) for line in payloads["train_sft.jsonl"].decode().splitlines()]
        self.assertEqual(len(sft), 3)
        self.assertEqual(sft[0]["completion"], [{"role": "assistant", "content": "SELECT secret_A"}])

    def test_changed_source_hash_and_changed_output_are_rejected(self):
        cache = module.ROOT / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="schema-input-tests-", dir=cache) as temp:
            root = Path(temp).resolve()
            self.assertTrue(root.is_relative_to(cache.resolve()))
            source = root / "source.json"
            source.write_text("first", encoding="utf-8")
            digest = module.sha256(source)
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                module.require_hash(source, digest)
            output = root / "output"
            output.mkdir()
            frozen = output / "train_inputs.jsonl"
            frozen.write_bytes(b"original\n")
            config = {"output_dir": "output"}
            with patch.object(module, "ROOT", root), patch.object(module, "load_sources", return_value=(config, {}, {}, {}, {}, {})), \
                    patch.object(module, "load_tokenizer", return_value=FakeTokenizer()), \
                    patch.object(module, "build_artifacts", return_value=({"train_inputs.jsonl": b"new\n"}, {}, [])):
                with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                    module.main("build")
            self.assertEqual(frozen.read_bytes(), b"original\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
