"""
演示 ONNX 前端：torch.onnx 导出 → 加载/打印 GraphProto → 遍历节点 → shape inference → ONNX Runtime 推理。

模型是一个 Linear + ReLU 小网络，导出后能看到真实的 Gemm / Relu 节点、权重 initializer，
以及 shape inference 如何补全中间张量的形状。

运行: python onnx_export_inspect.py
"""

import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from onnx import shape_inference


class TinyMLP(nn.Module):
    """Linear(4->3) + ReLU：导出为 Gemm + Relu。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _print_block(title: str, hint: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")
    if hint:
        print(f"提示: {hint}\n")


def main() -> None:
    model = TinyMLP().eval()
    x = torch.randn(2, 4)
    onnx_path = (Path(__file__).resolve().parent / "../../build/tiny_mlp.onnx").resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    _print_block(
        "1) 导出 ONNX",
        "关闭 external data，只生成单个 .onnx 文件；输入命名为 x 便于后续 runtime 喂数据。",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        torch.onnx.export(
            model,
            x,
            onnx_path,
            opset_version=18,
            input_names=["x"],
            output_names=["y"],
            external_data=False,
        )
    print(f"导出完成: {onnx_path}")

    _print_block(
        "2) 加载并打印 ONNX Graph",
        "看图的文本表示，确认输入、输出、initializer（权重/bias）和算子链是否符合预期。",
    )
    model_proto = onnx.load(onnx_path)
    print(onnx.printer.to_text(model_proto.graph))

    _print_block(
        "3) 遍历节点与权重",
        "逐个打印 op_type、输入、输出；initializer 是被折进图里的常量（Gemm 的 weight/bias）。",
    )
    for node in model_proto.graph.node:
        print(f"  {node.op_type:8} in={list(node.input)} out={list(node.output)}")
    for init in model_proto.graph.initializer:
        print(f"  initializer {init.name}: dims={list(init.dims)}")

    _print_block(
        "4) Shape Inference 前",
        "查看中间值 ValueInfo；导出后中间张量的 shape 往往尚未补全。",
    )
    print("Intermediate Value Info:", list(model_proto.graph.value_info))

    _print_block(
        "5) 执行 Shape Inference",
        "让 ONNX 依据算子语义（Gemm/Relu）推断并补全中间张量形状。",
    )
    model_proto = shape_inference.infer_shapes(model_proto)
    print("Shape inference complete.")

    _print_block(
        "6) Shape Inference 后",
        "再次查看 ValueInfo，观察中间张量 shape 是否被补上。",
    )
    print("Intermediate Value Info:", list(model_proto.graph.value_info))
    onnx.save(model_proto, onnx_path)
    print(f"已保存带 shape 信息的模型: {onnx_path}")

    _print_block(
        "7) ONNX Runtime 推理 + 与 PyTorch 对齐",
        "加载模型执行一次推理，并与 PyTorch eager 输出对比，验证导出数值一致。",
    )
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    input_data = x.numpy().astype(np.float32)
    (ort_out,) = session.run(None, {input_name: input_data})
    torch_out = model(x).detach().numpy()
    print(f"Runtime input name: {input_name}")
    print("Runtime output:\n", ort_out)
    print(f"max abs diff vs PyTorch: {np.abs(ort_out - torch_out).max():.3e}")


if __name__ == "__main__":
    main()
