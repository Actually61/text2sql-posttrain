"""把冻结的训练消息转成SFT特征，并在CPU上组成动态右填充batch。

本模块不加载模型、不训练、不连接数据库。真实来源哈希与训练成员归属
由上层准备脚本核验；这里检查单条记录、模板边界和监督标签的结构。
"""
from __future__ import annotations

from typing import Any


IGNORE_INDEX = -100
QWEN_EOS_TOKEN_ID = 151645
QWEN_PAD_TOKEN_ID = 151643


def _integer(value: Any) -> bool:
    # Python的bool也是int子类，但True不能当作来源索引或token编号。
    return isinstance(value, int) and not isinstance(value, bool)


def _check_budget(max_length: int) -> None:
    if not _integer(max_length) or max_length <= 0:
        raise ValueError("max_length必须是正整数")


def _token_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not all(_integer(token) and token >= 0 for token in value):
        raise ValueError(f"{name}必须是非负整数token组成的list")
    return list(value)


def _message_copy(message: Any, role: str) -> dict[str, str]:
    if (not isinstance(message, dict) or set(message) != {"role", "content"}
            or message.get("role") != role or not isinstance(message.get("content"), str)
            or not message["content"].strip()):
        raise ValueError(f"消息必须仅有role/content，role={role}，且content为非空文本")
    # 新建消息字典；传给tokenizer时也不让调用者原列表被就地改动。
    return {"role": role, "content": message["content"]}


def _validate_feature(feature: Any, max_length: int) -> tuple[list[int], list[int], list[int]]:
    """验证未padding的完整特征，保证后面只补齐、不改变监督位置。"""
    if not isinstance(feature, dict) or feature.get("project_split") != "train":
        raise ValueError("batch只能包含train特征")
    required = {"input_ids", "attention_mask", "labels", "prompt_length", "eos_position",
                "sequence_length", "supervised_token_count", "trailing_template_token_count"}
    if not required <= set(feature):
        raise ValueError("特征缺少token或边界元信息")
    ids = _token_list(feature["input_ids"], "input_ids")
    attention = _token_list(feature["attention_mask"], "attention_mask")
    labels = feature["labels"]
    if (not isinstance(labels, list) or not all(_integer(label) and label >= IGNORE_INDEX for label in labels)
            or any(label < 0 and label != IGNORE_INDEX for label in labels)):
        raise ValueError("labels只能包含token编号或-100")
    labels = list(labels)
    length = len(ids)
    if length == 0 or length > max_length or len(attention) != length or len(labels) != length:
        raise ValueError("特征不能为空、不能超长，且三组序列长度必须相同")
    if any(value != 1 for value in attention):
        raise ValueError("collate前的特征不应已经padding；attention_mask必须全为1")
    for name in ("prompt_length", "eos_position", "sequence_length", "supervised_token_count", "trailing_template_token_count"):
        if not _integer(feature[name]):
            raise ValueError(f"{name}必须是整数")
    start, end = feature["prompt_length"], feature["eos_position"]
    # 至少有1个prompt token、1个SQL token和一个EOS。
    if not 0 < start < end < length:
        raise ValueError("prompt/SQL/EOS边界无效或没有SQL监督token")
    if (feature["sequence_length"] != length or feature["supervised_token_count"] != end - start + 1
            or feature["trailing_template_token_count"] != length - end - 1):
        raise ValueError("特征记录的长度或监督token数与实际序列不一致")
    if ids[end] != QWEN_EOS_TOKEN_ID or ids[start:].count(QWEN_EOS_TOKEN_ID) != 1:
        raise ValueError("答案后缀必须恰好包含一个固定Qwen EOS")
    for position, (token, label) in enumerate(zip(ids, labels)):
        expected = token if start <= position <= end else IGNORE_INDEX
        if label != expected:
            raise ValueError("labels只能监督原位置上的SQL和EOS；prompt及模板尾部应为-100")
    return ids, attention, labels


def encode_sft_example(example: dict, tokenizer: Any, max_length: int = 4096) -> dict:
    """编码一条train记录；监督SQL+EOS，不裁剪，也不提前平移labels。"""
    _check_budget(max_length)
    if not isinstance(example, dict) or example.get("project_split") != "train":
        raise ValueError("encode_sft_example只接受train记录")
    if not _integer(example.get("source_index")) or example["source_index"] < 0:
        raise ValueError("source_index必须是非负整数")
    for key in ("db_id", "source_file"):
        if not isinstance(example.get(key), str) or not example[key].strip():
            raise ValueError(f"{key}必须是非空文本")
    if not isinstance(example.get("prompt"), list) or len(example["prompt"]) != 2:
        raise ValueError("prompt必须恰好包含system和user两条消息")
    if not isinstance(example.get("completion"), list) or len(example["completion"]) != 1:
        raise ValueError("completion必须恰好包含一条assistant SQL")
    prompt = [_message_copy(example["prompt"][0], "system"), _message_copy(example["prompt"][1], "user")]
    completion = _message_copy(example["completion"][0], "assistant")
    if (tokenizer.eos_token_id != QWEN_EOS_TOKEN_ID or tokenizer.pad_token_id != QWEN_PAD_TOKEN_ID):
        raise ValueError("分词器EOS/PAD与冻结的Qwen版本不一致")

    # return_dict=False必须显式传入，避免把BatchEncoding字典的键数量当token数。
    prompt_ids = _token_list(tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=False), "prompt_ids")
    input_ids = _token_list(tokenizer.apply_chat_template(
        prompt + [completion], tokenize=True, add_generation_prompt=False, return_dict=False), "input_ids")
    if not prompt_ids or input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("完整训练序列不是生成prompt token序列的延续")
    if len(input_ids) > max_length:
        raise ValueError(f"完整序列长度{len(input_ids)}超过max_length={max_length}；本版拒绝截断")

    prompt_length = len(prompt_ids)
    suffix = input_ids[prompt_length:]
    if suffix.count(QWEN_EOS_TOKEN_ID) != 1:
        raise ValueError("assistant后缀必须恰好有一个EOS")
    eos_position = prompt_length + suffix.index(QWEN_EOS_TOKEN_ID)
    sql_ids = input_ids[prompt_length:eos_position]
    if not sql_ids:
        raise ValueError("EOS前没有SQL token")
    # 逐字核对原SQL，不能把strip、空白压缩或模板变化悄悄当作同一训练目标。
    decoded_sql = tokenizer.decode(sql_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded_sql != completion["content"]:
        raise ValueError("EOS前token解码与原始assistant SQL不一致")
    trailing_ids = input_ids[eos_position + 1:]
    trailing_text = tokenizer.decode(trailing_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if trailing_text not in ("", "\n") or (not trailing_text and trailing_ids):
        raise ValueError("EOS后只允许没有token或一个模板换行，不接受其他内容")

    # 因果模型在计算loss时会做一次错位对齐；这里labels保持原token位置。
    # 首个SQL标签位于prompt_length，将由前一位置的logits负责预测。
    labels = [IGNORE_INDEX] * len(input_ids)
    labels[prompt_length:eos_position + 1] = input_ids[prompt_length:eos_position + 1]
    feature = {key: example[key] for key in ("source_index", "source_file", "db_id", "project_split")}
    feature.update(input_ids=input_ids, attention_mask=[1] * len(input_ids), labels=labels,
                   prompt_length=prompt_length, eos_position=eos_position,
                   sequence_length=len(input_ids), supervised_token_count=len(sql_ids) + 1,
                   trailing_template_token_count=len(trailing_ids))
    _validate_feature(feature, max_length)
    return feature


def collate_sft_batch(features: list[dict], pad_token_id: int, max_length: int = 4096) -> dict:
    """在CPU上右padding到当前batch最长序列，不按token值删除真实EOS。"""
    _check_budget(max_length)
    if not _integer(pad_token_id) or pad_token_id < 0:
        raise ValueError("pad_token_id必须是非负整数")
    if not isinstance(features, list) or not features:
        raise ValueError("features必须是非空list")
    checked = [_validate_feature(feature, max_length) for feature in features]
    longest = max(len(ids) for ids, _, _ in checked)
    padded_ids, padded_attention, padded_labels = [], [], []
    for ids, attention, labels in checked:
        padding_length = longest - len(ids)
        # 按新增位置补齐三组序列。即使pad与EOS编号相同，也不会mask真实EOS。
        padded_ids.append(ids + [pad_token_id] * padding_length)
        padded_attention.append(attention + [0] * padding_length)
        padded_labels.append(labels + [IGNORE_INDEX] * padding_length)
    import torch
    return {"input_ids": torch.tensor(padded_ids, dtype=torch.long, device="cpu"),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long, device="cpu"),
            "labels": torch.tensor(padded_labels, dtype=torch.long, device="cpu")}
