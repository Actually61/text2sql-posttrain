"""离线检查分词、模型保存加载、GPU 生成和 4-bit 量化计算。"""

import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# 只在本进程内启用离线模式，所有测试文件都在本地生成。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import accelerate
import bitsandbytes as bnb
import huggingface_hub
import safetensors
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请先运行 env_check.py 定位基础环境问题。")

    torch.manual_seed(42)
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / ".cache" / "library_smoke" / "random_qwen2"
    model_dir.mkdir(parents=True, exist_ok=True)
    package_names = [
        "torch", "numpy", "transformers", "accelerate", "bitsandbytes",
        "huggingface-hub", "tokenizers", "safetensors",
    ]
    package_versions = {name: version(name) for name in package_names}
    for name, installed_version in package_versions.items():
        print(f"{name}: {installed_version}")

    # 1. 自建一个简单词表，检查分词器的保存和重新加载。
    # 它仅用于环境检查，不是正式 Qwen 模型的分词器。
    vocab = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3,
             "SELECT": 4, "name": 5, "FROM": 6, "students": 7}
    vocab.update({f"unused_{index}": index for index in range(8, 64)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)

    # 2. 随机初始化极小的 Qwen2 结构，保存成 safetensors 权重文件。
    # 正式的 1.5B 预训练权重将在后续步骤下载；这里没有训练模型。
    config = Qwen2Config(
        # 使用 64 的倍数，使测试线性层满足 NF4 快内核的块对齐要求。
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=2, eos_token_id=3,
        pad_token_id=1,
    )
    original_model = AutoModelForCausalLM.from_config(config)
    parameter_count = sum(parameter.numel() for parameter in original_model.parameters())
    original_model.save_pretrained(model_dir)
    del original_model
    if not (model_dir / "model.safetensors").is_file():
        raise RuntimeError("未生成预期的 safetensors 权重文件。")

    # 3. 从本地重新加载。device_map 使用 Accelerate 的设备分配能力。
    # 自建 WordLevel 分词器要显式使用对应的通用类。
    # 正式模型会使用它自己的官方分词器文件和 AutoTokenizer。
    loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, device_map={"": "cuda:0"},
        dtype=torch.float32, attn_implementation="eager",
    ).eval()
    inputs = loaded_tokenizer(
        "SELECT name FROM students", return_tensors="pt", return_token_type_ids=False,
    ).to("cuda:0")
    if inputs["input_ids"].tolist() != [[4, 5, 6, 7]]:
        raise RuntimeError("分词结果与自建词表不一致。")
    if not all(parameter.device.type == "cuda" for parameter in model.parameters()):
        raise RuntimeError("微型模型未完整加载到 GPU。")
    with torch.inference_mode():
        logits = model(**inputs).logits
        generated = model.generate(**inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False)
    if tuple(logits.shape) != (1, 4, 64) or not torch.isfinite(logits).all().item():
        raise RuntimeError("微型模型前向计算结果异常。")
    if tuple(generated.shape) != (1, 6):
        raise RuntimeError("微型模型未按预期生成两个新 token。")
    print("本地分词器保存与加载: PASS")
    print("本地 safetensors 模型加载、GPU 前向与生成: PASS")

    # 4. 检查 Windows 上的 4-bit GPU 量化内核，而不只检查模块能否导入。
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    layer = bnb.nn.Linear4bit(
        64, 32, bias=False, compute_dtype=compute_dtype, quant_type="nf4",
    ).to("cuda:0")
    with torch.inference_mode():
        quantized_output = layer(torch.randn(2, 64, device="cuda:0", dtype=compute_dtype))
    torch.cuda.synchronize()
    if layer.weight.quant_state is None or layer.weight.dtype != torch.uint8:
        raise RuntimeError("线性层未生成预期的 4-bit 量化状态。")
    if quantized_output.device.type != "cuda" or tuple(quantized_output.shape) != (2, 32):
        raise RuntimeError("4-bit 量化层设备或输出形状异常。")
    if not torch.isfinite(quantized_output).all().item():
        raise RuntimeError("4-bit 量化层输出包含非有限数值。")
    print(f"NF4 量化层 GPU 计算: PASS，计算类型为 {compute_dtype}")

    # 5. 检查 Transformers 与 bitsandbytes 的完整量化加载接口。
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, device_map={"": "cuda:0"},
        dtype=compute_dtype, attn_implementation="eager",
        quantization_config=quantization_config,
    ).eval()
    quantized_layer_count = sum(
        isinstance(module, bnb.nn.Linear4bit) for module in quantized_model.modules()
    )
    with torch.inference_mode():
        quantized_logits = quantized_model(**inputs).logits
    torch.cuda.synchronize()
    if quantized_layer_count == 0 or not torch.isfinite(quantized_logits).all().item():
        raise RuntimeError("Transformers 的 4-bit 模型加载或前向计算异常。")
    if tuple(quantized_logits.shape) != (1, 4, 64):
        raise RuntimeError("4-bit 微型模型的输出形状异常。")
    print(f"Transformers 4-bit 模型加载与前向计算: PASS，量化线性层 {quantized_layer_count} 个")

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": package_versions,
        "offline": True,
        "smoke_model_origin": "random_initialization",
        "smoke_model_parameter_count": parameter_count,
        "smoke_model_path": str(model_dir),
        "tokenizer_roundtrip": "passed",
        "tokenizer_class": type(loaded_tokenizer).__name__,
        "tokenizer_origin": "custom_wordlevel_for_smoke_test",
        "safetensors_model_load": "passed",
        "model_device": str(next(model.parameters()).device),
        "logits_shape": list(logits.shape),
        "generated_shape": list(generated.shape),
        "nf4_gpu_forward": "passed",
        "nf4_compute_dtype": str(compute_dtype),
        "nf4_output_shape": list(quantized_output.shape),
        "nf4_packed_weight_dtype": str(layer.weight.dtype),
        "transformers_nf4_model_load": "passed",
        "transformers_nf4_layer_count": quantized_layer_count,
        "formal_model_validation": "pending",
    }
    report_path = project_root / "reports" / "model_libraries_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MODEL_LIBRARIES_CHECK: PASS")
    print(f"检查结果已保存: {report_path}")


if __name__ == "__main__":
    main()
