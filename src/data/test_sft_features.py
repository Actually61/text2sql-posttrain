"""Synthetic SFT masking/padding checks and a tiny CPU causal-loss example.

No training examples, validation/final data, model weights, model forward,
backward or optimizer are used. The optional local-tokenizer check uses only
tokenizer files and a synthetic SELECT query, with networking disabled.
"""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

from sft_features import encode_sft_example, collate_sft_batch


EOS = 151645
PAD = 151643
IGNORE = -100


class FakeTokenizer:
    """Character tokens and one distinguishable EOS; retains template newline."""
    eos_token_id = EOS
    pad_token_id = PAD
    eos_token = "<|im_end|>"

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Do not add special tokens twice")
        ids = []
        for index, part in enumerate(text.split(self.eos_token)):
            if index:
                ids.append(EOS)
            ids.extend(ord(char) + 10 for char in part)
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Boundary verification must preserve exact token text")
        return "".join(self.eos_token if token == EOS else chr(token - 10) for token in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=True):
        # Default deliberately matches the installed Transformers behavior.
        if return_dict:
            raise AssertionError("Caller must explicitly request return_dict=False")
        text = "".join(f"<start>{message['role']}\n{message['content']}{self.eos_token}\n"
                       for message in messages)
        if add_generation_prompt:
            text += "<start>assistant\n"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class BrokenPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        ids = super().apply_chat_template(messages, **kwargs)
        if messages[-1]["role"] == "assistant":
            ids[0] += 1
        return ids


def example(sql="SELECT 1;", index=0):
    return {
        "source_index": index, "source_file": "synthetic_only.json", "db_id": "synthetic",
        "project_split": "train",
        "prompt": [{"role": "system", "content": "Return SQLite SQL."},
                   {"role": "user", "content": "Return the integer one."}],
        "completion": [{"role": "assistant", "content": sql}],
    }


class SFTFeatureTests(unittest.TestCase):
    def test_prompt_mask_sql_and_eos_supervised_tail_newline_ignored_without_shift(self):
        tokenizer = FakeTokenizer()
        source = example("SELECT 1;\n")
        original = copy.deepcopy(source)
        feature = encode_sft_example(source, tokenizer)
        prompt = tokenizer.apply_chat_template(source["prompt"], tokenize=True,
                                               add_generation_prompt=True, return_dict=False)
        sql = tokenizer.encode(source["completion"][0]["content"], add_special_tokens=False)
        expected_ids = prompt + sql + [EOS] + tokenizer.encode("\n", add_special_tokens=False)
        self.assertEqual(feature["input_ids"], expected_ids)
        self.assertEqual(feature["labels"], [IGNORE] * len(prompt) + sql + [EOS, IGNORE])
        self.assertEqual(feature["attention_mask"], [1] * len(expected_ids))
        self.assertEqual(feature["prompt_length"], len(prompt))
        self.assertEqual(feature["supervised_token_count"], len(sql) + 1)
        self.assertEqual(feature["trailing_template_token_count"], 1)
        # System/user messages contain EOS too, but their labels remain masked.
        self.assertGreater(prompt.count(EOS), 0)
        for position, token in enumerate(prompt):
            if token == EOS:
                self.assertEqual(feature["labels"][position], IGNORE)
        # Labels stay at input positions; model loss is responsible for shifting.
        first_sql = len(prompt)
        self.assertEqual(feature["labels"][first_sql], feature["input_ids"][first_sql])
        self.assertNotEqual(feature["labels"][first_sql], feature["input_ids"][first_sql + 1])
        self.assertEqual(source, original)

    def test_right_padding_and_pad_equal_eos_preserve_real_eos_and_inputs(self):
        import torch

        tokenizer = FakeTokenizer()
        features = [encode_sft_example(example("SELECT 1;", 1), tokenizer),
                    encode_sft_example(example("SELECT 123456789;", 2), tokenizer)]
        original = copy.deepcopy(features)
        short_length, long_length = map(lambda item: len(item["input_ids"]), features)
        self.assertLess(short_length, long_length)
        for padding_token in (PAD, EOS):
            with self.subTest(pad_token_id=padding_token):
                batch = collate_sft_batch(features, padding_token)
                self.assertEqual(set(batch), {"input_ids", "attention_mask", "labels"})
                for tensor in batch.values():
                    self.assertEqual(tensor.shape, (2, long_length))
                    self.assertEqual(tensor.dtype, torch.long)
                    self.assertEqual(tensor.device.type, "cpu")
                self.assertEqual(batch["input_ids"][0, :short_length].tolist(), features[0]["input_ids"])
                self.assertEqual(batch["labels"][0, :short_length].tolist(), features[0]["labels"])
                self.assertEqual(batch["input_ids"][0, short_length:].tolist(),
                                 [padding_token] * (long_length - short_length))
                self.assertEqual(batch["attention_mask"][0].tolist(),
                                 [1] * short_length + [0] * (long_length - short_length))
                self.assertEqual(batch["labels"][0, short_length:].tolist(),
                                 [IGNORE] * (long_length - short_length))
                self.assertEqual(batch["labels"][1].tolist(), features[1]["labels"])
                self.assertEqual(int(batch["labels"][0, features[0]["eos_position"]]), EOS)
                self.assertEqual(int(batch["labels"][1, features[1]["eos_position"]]), EOS)
        self.assertEqual(features, original)

    def test_reject_prefix_mismatch_wrong_split_empty_answer_and_overbudget(self):
        tokenizer = FakeTokenizer()
        source = example()
        with self.assertRaises(ValueError):
            encode_sft_example(source, BrokenPrefixTokenizer())
        for split in ("validation", "final_eval", ""):
            with self.subTest(split=split), self.assertRaises(ValueError):
                encode_sft_example({**source, "project_split": split}, tokenizer)
        for answer in ("", "  \n\t"):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                encode_sft_example(example(answer), tokenizer)
        length = len(encode_sft_example(source, tokenizer)["input_ids"])
        self.assertEqual(len(encode_sft_example(source, tokenizer, max_length=length)["input_ids"]), length)
        with self.assertRaises(ValueError):
            encode_sft_example(source, tokenizer, max_length=length - 1)
        with self.assertRaises(ValueError):
            collate_sft_batch([encode_sft_example(source, tokenizer)], PAD, max_length=length - 1)

    def test_collator_rejects_pre_shifted_labels_and_masked_real_eos(self):
        feature = encode_sft_example(example(), FakeTokenizer())
        shifted = copy.deepcopy(feature)
        shifted["labels"] = shifted["labels"][1:] + [IGNORE]
        masked_eos = copy.deepcopy(feature)
        masked_eos["labels"][masked_eos["eos_position"]] = IGNORE
        for invalid in (shifted, masked_eos):
            with self.subTest(invalid=invalid["labels"]), self.assertRaises(ValueError):
                collate_sft_batch([invalid], PAD)

    def test_local_qwen_tokenizer_on_synthetic_sql(self):
        tokenizer_dir = Path(__file__).resolve().parents[2] / "models/qwen2.5-coder-1.5b-instruct"
        if not (tokenizer_dir / "tokenizer.json").is_file():
            self.skipTest("Local tokenizer is absent; no download is attempted")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        source = example("SELECT 1;\n")
        original = copy.deepcopy(source)
        feature = encode_sft_example(source, tokenizer)
        start, end = feature["prompt_length"], feature["eos_position"]
        decoded = tokenizer.decode(feature["input_ids"][start:end], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        self.assertEqual(decoded, source["completion"][0]["content"])
        self.assertTrue(all(label == IGNORE for label in feature["labels"][:start]))
        self.assertEqual(feature["labels"][start:end + 1], feature["input_ids"][start:end + 1])
        self.assertEqual(feature["labels"][end], tokenizer.eos_token_id)
        self.assertTrue(all(label == IGNORE for label in feature["labels"][end + 1:]))
        self.assertEqual(source, original)


class CausalLossTests(unittest.TestCase):
    def test_installed_loss_matches_manual_shift_and_ignores_only_shifted_mask(self):
        import torch
        import torch.nn.functional as functional
        from transformers.loss.loss_utils import ForCausalLMLoss

        # Two short CPU sequences: prompt, prompt, SQL..., EOS, optional padding.
        # EOS=2 is deliberately tiny here; these logits are independent of Qwen.
        labels = torch.tensor([[IGNORE, IGNORE, 3, 4, 2, IGNORE],
                               [IGNORE, IGNORE, 5, 2, IGNORE, IGNORE]], dtype=torch.long)
        vocab_size = 7
        logits = torch.arange(2 * 6 * vocab_size, dtype=torch.float32).reshape(2, 6, vocab_size) / 17
        logits_before, labels_before = logits.clone(), labels.clone()
        installed = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=vocab_size)
        manual = functional.cross_entropy(logits[:, :-1].contiguous().reshape(-1, vocab_size),
                                          labels[:, 1:].contiguous().reshape(-1), ignore_index=IGNORE)
        torch.testing.assert_close(installed, manual, rtol=1e-6, atol=1e-7)
        self.assertFalse(installed.requires_grad)

        # A logit at position t predicts label t+1. The last logit has no target.
        shifted_targets = functional.pad(labels, (0, 1), value=IGNORE)[:, 1:]
        ignored_logit_positions = shifted_targets == IGNORE
        altered_ignored = logits.clone()
        altered_ignored[ignored_logit_positions] = torch.arange(vocab_size, dtype=torch.float32) * 100 - 300
        ignored_loss = ForCausalLMLoss(logits=altered_ignored, labels=labels, vocab_size=vocab_size)
        torch.testing.assert_close(installed, ignored_loss, rtol=0, atol=0)

        # labels[0,1] is masked, but logits[0,1] predicts the first SQL at [0,2].
        # Masking this last-prompt logit would incorrectly remove first-SQL loss.
        self.assertEqual(int(labels[0, 1]), IGNORE)
        self.assertFalse(bool(ignored_logit_positions[0, 1]))
        changed_first_sql = logits.clone()
        changed_first_sql[0, 1, int(labels[0, 2])] += 10
        changed_loss = ForCausalLMLoss(logits=changed_first_sql, labels=labels, vocab_size=vocab_size)
        self.assertLess(float(changed_loss), float(installed))
        torch.testing.assert_close(logits, logits_before, rtol=0, atol=0)
        torch.testing.assert_close(labels, labels_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
