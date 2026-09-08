# GPU Matmul Codegen

本目录演示两条基于 MLIR Transform Dialect 的 GPU Matmul 路径：

1. **`tiling`**：从 256³ tensor `linalg.matmul` 出发，完成 CTA/K/warp tiling、vectorize、One-Shot Bufferize（OSB）和 GPU 映射，产出 `gpu.launch + vector.contract`；
2. **`mma-sync`**：从已经缩小到单条 MMA 指令形状、且已经放入单 warp `gpu.launch` 的 memref `linalg.matmul` 出发，将其改写为 `nvgpu.mma.sync`，并继续 Lower 到 PTX 和 cubin。

两条路径用于验证不同阶段，不能视为同一条完整大矩阵 Tensor Core 流水线。当前可稳定生成 Tensor Core cubin 的是 `mma-sync`；`tiling` 主要展示大矩阵的层级切分与 GPU 映射。

## 核心结论

### `rewrite_matmul_as_mma_sync` 实际处理什么

`transform.nvgpu.rewrite_matmul_as_mma_sync` 不负责：

- CTA、K 或 warp tiling；
- `vectorize_children`；
- 创建 `gpu.launch`；
- 将任意尺寸 Matmul 自动切分为多条 MMA 指令。

它要求 payload 已满足单条 MMA 指令的形状和执行布局。本仓的
[`matmul_mma_sync.mlir`](matmul_mma_sync.mlir) 已预先给定：

```text
grid  = 1 × 1 × 1 CTA
block = 32 × 1 × 1 threads（一个 warp）
A     = 16×16xf16
B     = 16×8xf16
C     = 16×8xf16
MMA   = m16n8k16
```

payload 此时仍包含 `linalg.matmul`，只是该算子已经位于 `gpu.launch` 内。Transform 根据 `gpu.thread_id x` 计算 lane ID，为每个 lane 插入符合 NVIDIA fragment 布局的 load/store，并将 Matmul 替换为一次 `nvgpu.mma.sync`。

因此，这条路径验证的是：

> 已完成 tiling 与 GPU 映射的 warp 级 Matmul，如何 tensorize 并 Lower 为真实 `mma.sync` 指令。

它不是通用大矩阵调度器。

### 为什么 Transform 可以出现在 OSB 之后

Transform Dialect 并不限定只能在 OSB 之前运行；具体时序由 transform op 的输入约束决定。

- tiling、fusion 和 tensor-level vectorize 通常放在 OSB 之前，便于利用 tensor 语义；
- 部分 GPU mapping、memref rewrite 和后端 tensorize 必须或可以在 OSB 之后执行；
- `rewrite_matmul_as_mma_sync` 接收的是符合约束的 memref `linalg.matmul`，因此本例无需再次执行 OSB。

本目录中的时序如下：

```text
tiling mode:
tensor linalg.matmul
  → CTA/K/warp tiling
  → vectorize
  → OSB
  → gpu.launch / thread mapping

mma-sync mode:
memref linalg.matmul inside gpu.launch
  → rewrite_matmul_as_mma_sync
  → nvgpu.mma.sync
  → NVVM / LLVM IR
  → PTX
  → cubin
```

`mma-sync` payload 是手工构造的 post-bufferization 形态，并已固定问题分块和 GPU mapping。它用于隔离验证 tensorize 与后端 Codegen，而不是证明前置 tiling/OSB 流水线已经自动产生了该 IR。

## 两种模式

| 模式 | 输入 | Transform 执行内容 | 输出与验收范围 |
|---|---|---|---|
| `tiling` | 256×256×256 tensor Matmul | CTA/K/warp tiling、vectorize、OSB、block/thread mapping | `gpu.launch` + `vector.contract` 或 `nvgpu.*` 中间 IR |
| `mma-sync` | 1 CTA × 1 warp × m16n8k16 memref Matmul | `rewrite_matmul_as_mma_sync` | `nvgpu.mma.sync` → LLVM NVVM intrinsic → PTX `mma.sync` → cubin |

### `tiling`：大矩阵层级调度

[`matmul_tiling.mlir`](matmul_tiling.mlir) 从 256³ FP16×FP16→FP32 Matmul 出发：

```text
tile_using_forall [128,128]
  → 2×2 CTA grid，每个 CTA 负责 128×128 输出

tile_using_for [0,0,64]
  → K 维按 64 分块

tile_using_forall num_threads [4,4]
  → 每个 CTA 使用 4×4 warps

vectorize_children_and_apply_patterns
  → linalg 计算转换为 vector.transfer_* + vector.contract

one_shot_bufferize
  → tensor 转为 memref

map_forall_to_blocks / map_nested_forall_to_threads
  → 生成 gpu.launch 并映射 blockIdx/threadIdx
```

非线性 `#gpu.warp<y>, #gpu.warp<x>` 映射使用
`block_dims = [128, 4, 1]`：X 方向包含 4 个 warp 的 128 个线程，Y 方向包含 4 个 warp。

脚本随后尝试：

```text
unroll_vectors_subgroup_mma [16,16,16]
convert-vector-to-gpu{use-nvgpu=true}
```

但当前内存布局可能不满足 warp-matrix 转换约束，因此 `vector.contract` 不一定改写为 `nvgpu.mma.sync`。该模式当前只保证生成 GPU 映射后的中间 IR，不保证生成 Tensor Core cubin。

### `mma-sync`：warp 级 Tensor Core Codegen

[`matmul_mma_sync.mlir`](matmul_mma_sync.mlir) 的 payload 已经是单 CTA、单 warp 和 MMA 原生形状。Transform 仅执行：

```mlir
transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
```

改写后的每个 lane 持有一部分 A/B/C fragment，32 个 lane 共同执行一次 `m16n8k16`：

```text
groupID         = laneId >> 2
threadIDInGroup = laneId % 4

A fragment：每个 lane 读取 8 个 f16
B fragment：每个 lane 读取 4 个 f16
C fragment：每个 lane 读取并写回 4 个 f16
```

形状、dtype、layout 或执行线程数不满足该 op 的约束时，改写会失败。256³ Matmul 必须先完成层级 tiling，不能直接交给此 op。

## Codegen 流水线

`mma-sync` 模式执行以下步骤：

```text
linalg.matmul inside gpu.launch
  → Transform Interpreter
nvgpu.mma.sync
  → gpu-kernel-outlining
  → convert-nvgpu-to-nvvm / convert-gpu-to-nvvm
NVVM / LLVM dialect
  → mlir-translate --mlir-to-llvmir
LLVM IR（llvm.nvvm.mma intrinsic）
  → llc -march=nvptx64 -mcpu=sm_XX
PTX（mma.sync）
  → ptxas -arch=sm_XX
cubin（CUDA ELF）
```

这条路径只生成 device cubin，不生成 host launcher，也不使用 `pipe-demo --jit`。

## 测试范围

当前测试是 **静态 Codegen smoke test**，不启动 GPU kernel，也不做数值正确性或性能测试。

`scripts/run_gpu_matmul_codegen.sh mma-sync` 依次检查：

1. Transform 后 IR 包含 `nvgpu.mma.sync`；
2. LLVM IR 包含 `llvm.nvvm.mma` intrinsic；
3. PTX 包含 `mma.sync`；
4. `ptxas` 成功生成 cubin；
5. `file` 将 cubin 识别为 ELF。

这些检查证明 MLIR 改写和 NVPTX 后端确实生成了 Tensor Core 指令，但不能证明：

- Matmul 数值结果正确；
- kernel 可以通过 CUDA Driver 正常启动；
- 内存访问没有越界；
- 性能优于 SIMT 实现。

`tiling` 模式只检查 Transform 后 IR 中存在 `gpu.launch`，以及
`vector.contract` 或 `nvgpu.mma`，不继续生成 cubin。

## 运行方式

```bash
# Tensor Core 静态 Codegen：生成 PTX 和 cubin
./scripts/run_gpu_matmul_codegen.sh mma-sync

# 大矩阵 CTA/K/warp tiling：生成 GPU 中间 IR
./scripts/run_gpu_matmul_codegen.sh tiling

# 等价的 CMake targets
cmake --build build --target test_gpu_matmul_codegen
cmake --build build --target test_gpu_matmul_tiling
```

默认目标为 `sm_80`，可通过环境变量修改：

```bash
CUDA_SM=sm_80 ./scripts/run_gpu_matmul_codegen.sh mma-sync
```

## 输出文件

默认输出目录为 `output/gpu_codegen/<mode>/`。

| 文件 | 内容 |
|---|---|
| `01_after_transform.mlir` | Transform 后的 `nvgpu.mma.sync` 或 `gpu.launch` IR |
| `02_llvm_dialect.mlir` | GPU/NVGPU Lower 后的 NVVM/LLVM dialect |
| `03_kernel_only.mlir` | 从 `gpu.module` 提取的 device module |
| `04_kernel.ll` | `mlir-translate` 生成的 LLVM IR |
| `04_kernel.sanitized.ll` | 兼容旧版 NVPTX `llc` 的 LLVM IR |
| `05_kernel.ptx` | NVPTX 后端生成的 PTX |
| `06_kernel.cubin` | `ptxas` 生成的 CUDA ELF |

`tiling` 模式当前只生成 `01_after_transform.mlir`。

## 工具链要求

| 工具 | 用途 |
|---|---|
| `mlir-opt` | 执行 Transform Interpreter 与 GPU/NVGPU Lowering |
| `mlir-translate` | MLIR LLVM dialect → LLVM IR |
| 带 NVPTX target 的 `llc` | LLVM IR → PTX |
| `ptxas` | PTX → cubin |

脚本按以下顺序查找带 NVPTX 后端的 `llc`：

1. `MLIR_PASS_LLC`；
2. `LLC`；
3. `PATH` 中的 `llc`；
4. `third_party/llvm-18/bin/llc`；
5. `/tmp/llvm_nvptx/extracted/usr/lib/llvm-18/bin/llc`。

若系统 LLVM 未包含 NVPTX，建议重新构建：

```bash
cmake ... -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;RISCV"
```

当 `mlir-translate` 与回退 `llc` 版本不一致时，
[`sanitize_llvm_ir_for_nvptx.py`](../scripts/sanitize_llvm_ir_for_nvptx.py)
会移除旧版 parser 不支持的部分新语法。该脚本只用于文本 IR 兼容，不改变 Matmul 调度或 MMA 语义。

## 与项目其他路径的关系

- **`pipe-demo`**：StableHLO → Linalg → CPU loops/vector → LLVM Dialect → ExecutionEngine JIT；
- **本目录**：Linalg + Transform → GPU/NVGPU → NVVM/LLVM IR → PTX → device cubin；
- **`mlir_compiler` P11**：GPU/PTX 概念模拟；本目录使用真实 `mlir-opt`、`llc` 和 `ptxas`。

当前目录重点验证 GPU 调度与后端 Codegen。它不是完整 GPU 应用：尚未包含 host runtime、数值回归和性能 benchmark。
