"""复现bitsandbytes首次forward的bias类型转换；微型层，无Qwen/Spider/反传/优化器。"""
from pathlib import Path
import json
import torch
from bitsandbytes.nn import Linear4bit

ROOT = Path(__file__).resolve().parents[2]
torch.manual_seed(42)
layer = Linear4bit(32, 32, bias=True, compute_dtype=torch.bfloat16,
                   compress_statistics=True, quant_type="nf4").to(dtype=torch.bfloat16, device="cuda")
layer.requires_grad_(False)
original_bias = layer.bias.detach().cpu().clone()
# 复现prepare_model_for_kbit_training对非量化参数的上转。
layer.bias.data = layer.bias.data.float()
before = layer.bias.detach().cpu().clone()
weight_before = layer.weight.detach().cpu().clone()
values = torch.ones((2, 32), device="cuda", dtype=torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    result = layer(values)
after = layer.bias.detach().cpu().clone()
assert before.dtype == torch.float32 and after.dtype == torch.bfloat16
assert torch.equal(before, after.float()) and torch.equal(original_bias, after)
assert torch.equal(weight_before, layer.weight.detach().cpu())
assert all(p.grad is None for p in layer.parameters())
report = {"status": "pass", "before_bias_dtype": str(before.dtype), "after_bias_dtype": str(after.dtype),
          "bias_numeric_values_exactly_equal_when_compared_as_fp32": True,
          "quantized_weight_bytes_unchanged": True, "finite_output": bool(torch.isfinite(result).all()),
          "qwen_model_loaded": False, "spider_data_loaded": False, "backward_calls": 0, "optimizer_steps": 0,
          "cause_source": ".venv/Lib/site-packages/bitsandbytes/nn/modules.py:629-634",
          "scope": "Minimal reproduction establishes library behavior; not a per-parameter trace of the exited Qwen process"}
(ROOT / "reports/qlora_smoke_bias_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
