"""
演示 torch.fx：symbolic_trace 得到 GraphModule → 遍历节点 → 代数化简（消除恒等算子）→ recompile。

模型故意混入两个恒等算子（`+ 0`、`* 1`），再叠加真实的 `* 3`、`+ 2`、`relu`，
这样图改写前后对比清晰：改写会删掉 add(_, 0) 与 mul(_, 1)，保留真正的计算。

运行: python fx_graph_rewrite.py
"""

from __future__ import annotations

import operator

import torch
from torch import fx


def _print_block(title: str, hint: str) -> None:
    """统一打印区块标题与说明，便于区分不同输出。"""
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")
    if hint:
        print(f"提示: {hint}\n")


class AffineReLU(torch.nn.Module):
    """(x + 0) * 1 * 3 + 2 -> relu：前两步是可被消除的恒等算子。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0        # 恒等：加 0，可消除
        x = x * 1        # 恒等：乘 1，可消除
        x = x * 3 + 2    # 真实的 scale + shift
        return torch.relu(x)


def _is_literal(value: object, target: float) -> bool:
    """判断 FX 节点实参是否为指定字面量常数（排除 Node/Tensor）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == target


def _fold_identity(node: fx.Node) -> bool:
    """把 add(x, 0) / mul(x, 1)（含反向操作数顺序）替换为其非常量输入。"""
    if node.op != "call_function" or len(node.args) < 2:
        return False
    lhs, rhs = node.args[0], node.args[1]
    if node.target in (torch.add, operator.add) and _is_literal(rhs, 0):
        node.replace_all_uses_with(lhs)
    elif node.target in (torch.add, operator.add) and _is_literal(lhs, 0):
        node.replace_all_uses_with(rhs)
    elif node.target in (torch.mul, operator.mul) and _is_literal(rhs, 1):
        node.replace_all_uses_with(lhs)
    elif node.target in (torch.mul, operator.mul) and _is_literal(lhs, 1):
        node.replace_all_uses_with(rhs)
    else:
        return False
    node.graph.erase_node(node)
    return True


def main() -> None:
    model = AffineReLU()
    traced = fx.symbolic_trace(model)

    _print_block(
        "1) FX Graph（IR 的文本表示）",
        "每个节点对应一次计算；边由 def/use 隐含在 args 里。",
    )
    print(traced.graph)

    _print_block(
        "2) 节点列表（op / target）",
        "op: placeholder / get_attr / call_function / call_method / output 等；"
        "target 对 call_* 为具体函数或方法。",
    )
    for node in traced.graph.nodes:
        print(f"  {node.op:16} {node.target}")

    _print_block(
        "3) 图改写：消除恒等算子",
        "遍历 call_function，若命中 add(_, 0) 或 mul(_, 1) 则用其非常量输入替换并擦除节点。",
    )
    changed = sum(_fold_identity(node) for node in list(traced.graph.nodes))
    print(f"  移除的恒等节点数: {changed}")

    traced.graph.lint()
    traced.recompile()

    _print_block(
        "4) 改写后的 FX Graph + Python 代码",
        "recompile() 根据当前 graph 重新生成可执行的 forward；恒等算子已消失，只剩 mul/add/relu。",
    )
    print(traced.graph)
    print(traced.code)

    _print_block(
        "5) 数值自检",
        "改写不应改变语义：对同一输入，改写前后输出应完全一致。",
    )
    x = torch.randn(2, 4)
    expect = model(x)
    actual = traced(x)
    print(f"  max abs diff: {(expect - actual).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
