"""固定32条完整训练题的压力试跑排程；只返回索引和统计，不修改token。"""
from __future__ import annotations

from collections import Counter
import random


BUCKET_UPPER_BOUNDS = (512, 1024, 2048, 4096)


def bucket_index(length):
    if not isinstance(length, int) or isinstance(length, bool) or length <= 1:
        raise ValueError("Sequence length must be an integer greater than one")
    for index, bound in enumerate(BUCKET_UPPER_BOUNDS):
        if length <= bound:
            return index
    raise ValueError("Complete training row exceeds 4096; do not truncate")


def select_pilot(rows, seed=42):
    """每桶8条、每组各桶2条；最长输入固定在第三组的最后一个microbatch。"""
    if not isinstance(seed, int) or isinstance(seed, bool) or not rows:
        raise ValueError("Require a nonempty training source and integer seed")
    buckets = [[] for _ in BUCKET_UPPER_BOUNDS]
    by_id = {}
    for row in rows:
        identifier = row.get("source_index")
        length, targets = row.get("sequence_length"), row.get("supervised_token_count")
        if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier < 0 or identifier in by_id:
            raise ValueError("Training source IDs must be nonnegative, unique integers")
        if row.get("project_split") != "train":
            raise ValueError("Pilot may only select training examples")
        slot = bucket_index(length)
        if not isinstance(targets, int) or isinstance(targets, bool) or not 0 < targets < length:
            raise ValueError("Every example must have valid next-token targets")
        for field in ("input_ids", "attention_mask", "labels"):
            if field in row and len(row[field]) != length:
                raise ValueError("Stored full arrays do not match sequence_length")
        metadata = {"source_index": identifier, "sequence_length": length,
                    "supervised_token_count": targets, "bucket_index": slot}
        buckets[slot].append(metadata)
        by_id[identifier] = metadata
    if any(len(bucket) < 8 for bucket in buckets):
        raise ValueError("Require at least eight complete examples in every length bucket")

    # Tie-breaking by source ID is fixed; selection is independent of input-file row order.
    longest = min(by_id.values(), key=lambda r: (-r["sequence_length"], r["source_index"]))
    most_targets = min(by_id.values(), key=lambda r: (-r["supervised_token_count"], r["source_index"]))
    required_ids = {longest["source_index"], most_targets["source_index"]}
    rng = random.Random(seed)
    selected_buckets = []
    for bucket in buckets:
        ordered = sorted(bucket, key=lambda r: r["source_index"])
        forced = [r for r in ordered if r["source_index"] in required_ids]
        remaining = [r for r in ordered if r["source_index"] not in required_ids]
        chosen = forced + rng.sample(remaining, 8 - len(forced))
        rng.shuffle(chosen)
        selected_buckets.append(chosen)

    # Index 4 is assigned to group index 2: after the first process has saved and exited.
    long_bucket = selected_buckets[longest["bucket_index"]]
    long_offset = next(i for i, r in enumerate(long_bucket) if r["source_index"] == longest["source_index"])
    long_bucket[4], long_bucket[long_offset] = long_bucket[long_offset], long_bucket[4]
    groups = []
    required_examples = {}
    for group_index in range(4):
        group = [r for bucket in selected_buckets for r in bucket[2 * group_index:2 * group_index + 2]]
        rng.shuffle(group)
        if group_index == 2:
            offset = next(i for i, r in enumerate(group) if r["source_index"] == longest["source_index"])
            group[-1], group[offset] = group[offset], group[-1]
        counts = Counter(str(BUCKET_UPPER_BOUNDS[r["bucket_index"]]) for r in group)
        if sorted(counts.values()) != [2, 2, 2, 2]:
            raise AssertionError("Each accumulation group must contain two examples from every bucket")
        record = {"group_index": group_index, "update": group_index + 1,
                  "source_indices": [r["source_index"] for r in group],
                  "sequence_lengths": [r["sequence_length"] for r in group],
                  "target_counts": [r["supervised_token_count"] for r in group],
                  "group_target_tokens": sum(r["supervised_token_count"] for r in group),
                  "group_input_tokens": sum(r["sequence_length"] for r in group),
                  "maximum_sequence_length": max(r["sequence_length"] for r in group),
                  "maximum_target_tokens": max(r["supervised_token_count"] for r in group),
                  "bucket_counts": dict(counts)}
        groups.append(record)
        for role, required in (("longest_input", longest), ("longest_answer", most_targets)):
            for micro_index, r in enumerate(group):
                if r["source_index"] == required["source_index"]:
                    required_examples[role] = {**r, "group_index": group_index, "microbatch_index": micro_index}
    selected_ids = [i for group in groups for i in group["source_indices"]]
    if len(selected_ids) != 32 or len(set(selected_ids)) != 32 or not required_ids <= set(selected_ids):
        raise AssertionError("Pilot selection lost/repeated required training examples")
    return {"seed": seed, "training_source_count": len(rows),
            "selected_source_indices": selected_ids, "groups": groups,
            "bucket_counts": {str(bound): 8 for bound in BUCKET_UPPER_BOUNDS},
            "candidate_bucket_counts": {str(bound): len(bucket) for bound, bucket in zip(BUCKET_UPPER_BOUNDS, buckets)},
            "required_examples": required_examples,
            "input_tokens": sum(g["group_input_tokens"] for g in groups),
            "target_tokens": sum(g["group_target_tokens"] for g in groups),
            "selected_count": 32, "optimizer_updates": 4, "micro_batch_size": 1,
            "gradient_accumulation_steps": 8, "input_arrays_modified": False,
            "scope": "Balanced length stress pilot, not the natural training distribution; do not directly extrapolate mean throughput"}
