"""短训练子集、固定排程与有效token加权loss；不改写正式数据。"""
from contextlib import nullcontext
import math
import random
import time

import torch
from src.data.sft_features import collate_sft_batch


def select_short_subset(rows, seed=42, count=32, max_length=512):
    ids = [r["source_index"] for r in rows]
    if len(set(ids)) != len(ids) or any(r["project_split"] != "train" for r in rows):
        raise ValueError("Selection requires unique training records")
    eligible = sorted((r for r in rows if r["sequence_length"] <= max_length), key=lambda r: r["source_index"])
    if count <= 0 or len(eligible) < count:
        raise ValueError("Not enough eligible short training examples")
    chosen = random.Random(seed).sample(eligible, count)
    return sorted(chosen, key=lambda r: r["source_index"]), len(eligible)


def build_schedule(source_ids, seed=42, batch_size=2, steps=32):
    if (len(set(source_ids)) != len(source_ids) or batch_size <= 0
            or len(source_ids) % batch_size or not 0 < steps <= 32):
        raise ValueError("Invalid unique IDs, batch size or step budget")
    if not source_ids:
        raise ValueError("Empty training subset")
    schedule, epoch = [], 1
    while len(schedule) < steps:
        order = sorted(source_ids)
        random.Random(seed + epoch - 1).shuffle(order)
        for start in range(0, len(order), batch_size):
            schedule.append({"step": len(schedule) + 1, "epoch": epoch,
                             "source_indices": order[start:start + batch_size]})
            if len(schedule) == steps:
                break
        epoch += 1
    return schedule


def weighted_loss_summary(records):
    if not records:
        raise ValueError("No loss batches")
    total_tokens, total_nll = 0, 0.0
    for record in records:
        count, loss = record["supervised_tokens"], record["loss"]
        if type(count) is not int or count <= 0 or not math.isfinite(loss) or loss < 0:
            raise ValueError("Require positive token counts and finite nonnegative loss")
        total_tokens += count
        total_nll += float(loss) * count
    return {"weighted_loss": total_nll / total_tokens, "total_negative_log_likelihood": total_nll,
            "supervised_tokens": total_tokens, "batch_count": len(records)}


def measure_subset_loss(model, rows, device, batch_size=2, max_length=512, pad_token_id=151643):
    """相同顺序、相同batch边界的teacher-forcing训练子集loss，不计算EX。"""
    was_training = model.training
    model.eval()
    records = []
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                group = rows[start:start + batch_size]
                batch = collate_sft_batch(group, pad_token_id, max_length=max_length)
                # 对齐模型内部一次shift后的目标；首位置label本来就是-100。
                count = int((batch["labels"][:, 1:] != -100).sum().item())
                if count != sum(r["supervised_token_count"] for r in group) or count == 0:
                    raise ValueError("Shifted target count differs from assistant-only metadata")
                batch = {k: v.to(device) for k, v in batch.items()}
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
                with amp:
                    loss = model(**batch, use_cache=False).loss
                value = float(loss.item())
                if not math.isfinite(value) or loss.requires_grad:
                    raise ValueError("Invalid no-grad subset loss")
                records.append({"source_indices": [r["source_index"] for r in group],
                                "loss": value, "supervised_tokens": count})
                del loss, batch  # 仅保留标量；不保存整批logits，防止显存累积。
    finally:
        model.train(was_training)
    summary = weighted_loss_summary(records)
    return {**summary, "batches": records, "seconds": time.perf_counter() - started,
            "gradient_updates": 0, "source": "fixed training subset, not validation or EX"}
