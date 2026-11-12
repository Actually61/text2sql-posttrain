"""真实模型的一次无梯度前向，定位首次字节审计差异；不执行训练或保存adapter。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from src.data.sft_features import collate_sft_batch
from src.infer.baseline_pilot import read_json, save_json, sha256


def records(model, canonical_bias_names=()):
    result = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        value = p.detach()
        if name in canonical_bias_names:
            value = value.to(torch.bfloat16)
        raw = value.cpu().contiguous().view(torch.uint8).numpy()
        result[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                        "sha256": hashlib.sha256(memoryview(raw).cast("B")).hexdigest()}
    return result


started = time.perf_counter()
torch.manual_seed(42)
config = read_json(ROOT / "configs/qlora_smoke.json")
manifest_path = ROOT / "data/processed/spider_sft_tokens_v1/manifest.json"
assert sha256(manifest_path) == config["sft_manifest_sha256"]
manifest = read_json(manifest_path)
tokens = manifest_path.parent / "train_tokens.jsonl"
assert sha256(tokens) == manifest["output_sha256"][tokens.name]
with tokens.open(encoding="utf-8") as stream:
    rows = [json.loads(next(stream)), json.loads(next(stream))]
assert [r["source_index"] for r in rows] == [0, 1]
batch = collate_sft_batch(rows, 151643, max_length=512)
weight = ROOT / config["model_dir"] / "model.safetensors"
expected_weight_hash = "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"
assert sha256(weight) == expected_weight_hash
model = AutoModelForCausalLM.from_pretrained(ROOT / config["model_dir"], local_files_only=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
    dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
    revision=config["revision"], **config["lora"]))
model.eval()
biases = {name + ".bias": module.bias.detach().cpu().clone()
          for name, module in model.named_modules()
          if isinstance(module, bnb.nn.Linear4bit) and module.bias is not None}
assert len(biases) == 84 and all(v.dtype == torch.float32 for v in biases.values())
before = records(model)
expected = dict(before)
for name, value in biases.items():
    cast = value.to(torch.bfloat16)
    assert torch.equal(value, cast.float())
    expected[name] = {"dtype": str(cast.dtype), "shape": list(cast.shape),
                      "sha256": hashlib.sha256(memoryview(cast.view(torch.uint8).numpy()).cast("B")).hexdigest()}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(**{k: v.to("cuda") for k, v in batch.items()}, use_cache=False)
after = records(model)
changed = [name for name in before if before[name] != after[name]]
assert set(changed) == set(biases), changed
assert expected == after, "Unexpected changes beyond the exact bias casts"
named = dict(model.named_parameters())
assert all(torch.equal(value, named[name].detach().cpu().float()) for name, value in biases.items())
assert all(p.grad is None for p in model.parameters())
assert sha256(weight) == expected_weight_hash
report = {"status": "pass", "model_revision": config["revision"],
    "diagnostic_top_level_forward_calls": 1, "backward_calls": 0, "optimizer_steps": 0,
    "frozen_parameter_tensors": len(before), "changed_tensors": len(changed),
    "changed_names": changed, "changes_only_84_biases_fp32_to_bf16": True,
    "all_bias_values_exactly_equal_as_fp32": True,
    "all_other_frozen_parameter_bytes_unchanged": True,
    "matches_expected_after_numerically_lossless_bias_preparation": True,
    "base_weight_file_unchanged": True, "forward_loss": float(output.loss),
    "seconds_excluding_initial_imports": time.perf_counter() - started,
    "validation_examples_loaded": False, "final_examples_read": False,
    "scope": "Independent reproduction of the same first-forward path; cannot recover the exited v1 adapter",
    "before_records": before, "after_records": after}
save_json(ROOT / "reports/qlora_smoke_full_bias_diagnostic.json", report)
print(json.dumps({k: v for k, v in report.items() if k not in ("before_records", "after_records", "changed_names")}, indent=2))
