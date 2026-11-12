"""检查项目 Python、PyTorch 和实际 GPU 前向/反向计算。"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main():
    # 第一关：确认 PyTorch 能发现可用的 CUDA 设备。
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python 路径: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch CUDA 构建版本: {torch.version.cuda}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请先检查安装版本和驱动，再继续项目。")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"GPU: {properties.name}")
    print(f"总显存: {properties.total_memory / 1024**2:.1f} MiB")
    print(f"bf16 能力检查: {bf16_supported}")

    # 第二关：显式把数据放到 GPU，实际完成一次矩阵乘法。
    # requires_grad=True 表示需要计算损失对 w 的梯度。
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    w = torch.tensor([[1.0], [2.0]], device=device, requires_grad=True)
    y = x @ w
    loss = y.square().mean()

    # 第三关：反向传播。此处只检查梯度计算，没有更新参数。
    loss.backward()
    torch.cuda.synchronize(device)
    y_cpu = y.detach().cpu()
    gradient_cpu = w.grad.detach().cpu()
    torch.testing.assert_close(y_cpu, torch.tensor([[5.0], [11.0]]))
    torch.testing.assert_close(loss.detach().cpu(), torch.tensor(73.0))
    torch.testing.assert_close(gradient_cpu, torch.tensor([[38.0], [54.0]]))
    # 将结果移回 CPU 后，再检查张量与 NumPy 数组的转换。
    numpy_output = y_cpu.numpy()
    np.testing.assert_array_equal(numpy_output, np.array([[5.0], [11.0]]))
    torch.testing.assert_close(torch.from_numpy(numpy_output), y_cpu)

    # 这些是 PyTorch 分配器的统计，不包括全部驱动上下文占用。
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_cuda_build_version": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": properties.name,
        "gpu_total_memory_mib": properties.total_memory / 1024**2,
        "compute_capability": [properties.major, properties.minor],
        "bf16_capability_check": bf16_supported,
        "smoke_test": {
            "status": "passed",
            "device": str(y.device),
            "dtype": str(y.dtype),
            "matrix_output": y_cpu.tolist(),
            "loss": loss.item(),
            "weight_gradient": gradient_cpu.tolist(),
            "numpy_roundtrip": "passed",
        },
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }
    output_path = Path(__file__).resolve().parents[1] / "reports" / "env_check.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"GPU 计算结果: {y_cpu.tolist()}")
    print(f"损失: {loss.item()}；梯度: {gradient_cpu.tolist()}")
    print(f"PyTorch 峰值显存: allocated={peak_allocated:.3f} MiB, reserved={peak_reserved:.3f} MiB")
    print("GPU 前向与反向计算: PASS")
    print("PyTorch 与 NumPy 转换: PASS")
    print(f"检查结果已保存: {output_path}")


if __name__ == "__main__":
    main()
