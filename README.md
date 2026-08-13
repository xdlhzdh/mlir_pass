# MLIR AI Compiler Pipeline Demo

用官方 MLIR C++ API 把 StableHLO IR 逐步降到 LLVM Dialect 并 JIT 执行的教学工程。

本仓提供：

- **`pipe-demo`**：唯一的可执行文件。读入一份 `.mlir`（StableHLO），按固定 pipeline 依次跑 fusion → linalg → bufferize → loops/affine/vector → LLVM，输出最终 IR 或 JIT 执行结果。
- **28 个自定义 teaching pass**（`lib/Transforms/`）：19 个 fusion（Conv+BN、Softmax、Attention 等子图识别与标注）+ 9 个中后端（linalg 折叠、buffer 提升、loop tiling、vector 简化、LLVM 清理）。用官方 `PassManager` 编排，源码在 `lib/Pipeline/`。
- **`AICompilerPlugin`**：把 fusion pass 打包成 `mlir-opt` 可加载的 plugin（`tools/mlir-opt-plugin/`）。
- **测试体系**：Shell regression（粗粒度 grep）、LIT/FileCheck（IR 结构断言）、JIT 数值校验（编译执行后和 NumPy 比对）、跨仓 e2e（多数为兄弟仓 P4 导出 → 本仓 fusion）。

输入来源：`test/` 下的手写 `.mlir`，或兄弟仓 P4 导出的标准 StableHLO 文本。交接说明见 [mlir_compiler README · 与 mlir_pass 的接口](../mlir_compiler/README.md#mlir-pass-interface)。

---

## 环境与构建

### 依赖

| 组件 | 要求 |
|------|------|
| LLVM / MLIR | `/usr/local`（或 `CMAKE_PREFIX_PATH`），含 `MLIRConfig.cmake`、`libMLIR*.a` |
| StableHLO | 与 MLIR 版本匹配的 headers / 静态库 |
| Ninja | 推荐 |
| LIT + FileCheck | 可选（仅 `test_lit_filecheck`）。`apt install llvm-22-tools` 提供 |

`llvm-*-dev` 通常不含可链接的 `libMLIR*.a`，不能用 `/usr/lib/llvm-*` 作为 `CMAKE_PREFIX_PATH`。`lit` / `FileCheck` 与自编译 MLIR 无需同版本。

### 编译

```bash
export CC=gcc CXX=g++
cmake -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DSTABLEHLO_LIB_DIR=/usr/local/lib
ninja -C build
```

产物：`build/tools/pipe-demo/pipe-demo`。

> WSL 注意：若未指定编译器，CMake 可能误选 `clang-cl`。显式设 `CC`/`CXX` 或删 `build/` 重新配置。

### 快速验证

```bash
ninja -C build test_shell_regression   # 17 项 Shell 回归
./build/tools/pipe-demo/pipe-demo --input=test/mini_model.mlir --loop-mode=scf-seq
```

---

## 运行 `pipe-demo`

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--input=<file>` | StableHLO `.mlir`（必填） |
| `--loop-mode=` | `scf-seq` / `scf-par`（默认） / `affine` / `vector`——四条互斥的 lowering 路径 |
| `--pipeline-stop-after=` | `fusion` / `linalg` / `bufferize` / `loops` / `affine` / `vector` / `llvm` / `all`（默认） |
| `--dump-ir` | 每个 pass 和 stage 后把 IR 打到 stderr；不再把最终 IR 打到 stdout |
| `--jit` | 跑完 pipeline 后 JIT 执行，打印 `JIT result: …`（见下文） |
| `--entry-func=` | JIT 入口函数名（默认 `inference`） |
| `--list-passes` | 打印 pipeline 路径与自定义 pass 的对应关系 |

### 典型用法

```bash
# 看最终 LLVM IR（stdout）
pipe-demo --input=test/matmul_add.mlir --loop-mode=scf-seq > after-llvm.mlir

# 只到 fusion：看 Conv+BN 融合后的 StableHLO
pipe-demo --input=test/conv_bn_relu.mlir --pipeline-stop-after=fusion

# 观察每个 pass 的 IR 变化（stderr）
pipe-demo --input=test/matmul_add.mlir --dump-ir --loop-mode=affine 2>&1 | less

# JIT 执行并打印数值
pipe-demo --input=test/matmul_add.mlir --jit --loop-mode=scf-seq
```

### `--jit` 工作原理

不加 `--jit`：pipeline 跑完后把最终 IR 打到 stdout。加了 `--jit`：在本进程里用 MLIR `ExecutionEngine`（底层 LLVM ORC JIT）把 LLVM Dialect 编译成本机代码，调用入口函数，打印返回的 f32 数组。

`runJit`（`tools/pipe-demo/main.cpp`）四步：

**1. 读入口签名（lowering 之前）。** 必须在降到 LLVM 之前查 `func.func`，因为此时返回值还是 `tensor<…xf32>`，可以拿到 rank 和元素个数。教学图把常量嵌在 IR 里，入口必须无参数、返回 f32 ranked tensor（rank ≤ 4）。

```cpp
auto func = module.lookupSymbol<func::FuncOp>(funcName);  // "inference"
auto ranked = dyn_cast<RankedTensorType>(func.getFunctionType().getResult(0));
```

**2. 跑完整 pipeline，降到 LLVM Dialect。**

```cpp
runAICompilerPipeline(module, opts);
```

**3. 创建 JIT 引擎。** `ExecutionEngine::create` 把 LLVM Dialect 译成 LLVM IR，再编译成机器码。还需链上 runner `.so` 并注册 `malloc`/`free`（bufferize 后的 alloc 会调它们）。

```cpp
ExecutionEngineOptions engineOptions;
engineOptions.sharedLibPaths = {
    "/usr/local/lib/libmlir_c_runner_utils.so",
    "/usr/local/lib/libmlir_runner_utils.so",
};
auto engine = ExecutionEngine::create(module, engineOptions);
engine->registerSymbols(…);  // malloc / free
```

**4. 调用并打印。** `invokePacked` 按函数名找到 JIT 符号，结果写进 `StridedMemRefType<float, Rank>`。

```cpp
StridedMemRefType<float, Rank> result{};
void *args[] = {&result};
engine.invokePacked(funcName, args);
// → JIT result (4 elements): 1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00
```

以 `test/matmul_add.mlir` 为例——`[[1,2],[3,4]] × I + 0.5` → `[[1.5, 2.5], [3.5, 4.5]]`：

```mlir
func.func @inference() -> tensor<2x2xf32> {
  %a   = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %b   = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
  %bias = stablehlo.constant dense<[[0.5, 0.5], [0.5, 0.5]]> : tensor<2x2xf32>
  %mm  = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0],
           precision = [DEFAULT, DEFAULT]
           : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %out = stablehlo.add %mm, %bias : tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}
```

---

## Pipeline 与 Pass

编排入口：`lib/Pipeline/Pipeline.cpp`。每个 stage 用独立 `PassManager`。

### Pipeline 总览

```text
StableHLO → Linalg → bufferize(memref)
  → scf-seq / scf-par / affine / vector  (--loop-mode)
  → CF → LLVM Dialect → JIT
```

| Stage | `--pipeline-stop-after` | 做什么 | 自定义 pass |
|-------|-------------------------|--------|-------------|
| 1 fusion | `fusion` | StableHLO 图优化：Conv+BN 融合、ReLU→clamp、子图标注（Softmax/Attention/RMSNorm/GELU/SwiGLU/QDQ/Layout/KVCache…）、常量折叠 | 19 个（见下表） |
| 2 linalg | `linalg` | StableHLO → Linalg tensor | `custom-linalg-opt` |
| 3 bufferize | `bufferize` | tensor → memref + buffer 管理 | `custom-buffer-opt` |
| 4 loops | `loops` / `affine` / `vector` | memref → SCF / Affine / Vector（按 `--loop-mode`） | `custom-loop-tiling` / `custom-affine-opt` / `custom-vector-opt` |
| 5 llvm | `llvm` / `all` | → LLVM Dialect | `custom-llvm-cleanup` |

### 自定义 Pass（28 个）

**Fusion（19 个，`lib/Transforms/`）**

| Pass | 源文件 | 做什么 |
|------|--------|--------|
| `conv-bn-fusion` | `ConvBNFusion.cpp` | BN 折叠进 Conv 权重和 bias |
| `conv-bn-relu-fusion` | `ConvBNReluFusion.cpp` | `maximum(x,0)` → `clamp(0,x,+inf)` |
| `stablehlo-constant-fold` | `StablehloConstantFold.cpp` | 双常量 `add`/`multiply` 折叠 |
| `softmax-legalize` | `SoftmaxLegalize.cpp` | `exp/reduce_sum(exp)` 分解 → 标注 `aicom.softmax_canonicalized` |
| `rmsnorm-legalize` | `RMSNormLegalize.cpp` | RMSNorm 分解链 → `aicom.rmsnorm_canonicalized` |
| `attention-legalize` | `AttentionLegalize.cpp` | 两连 `dot_general` → `aicom.scaled_dot_product_attention` |
| `rope-legalize` | `RoPELegalize.cpp` | RoPE 分解链 → `aicom.rope_canonicalized` |
| `layernorm-legalize` | `LayerNormLegalize.cpp` | LayerNorm → `aicom.layernorm_canonicalized` |
| `gelu-legalize` | `GeluLegalize.cpp` | GELU 分解链 → `aicom.gelu_canonicalized` |
| `swiglu-legalize` | `SwiGLULegalize.cpp` | `silu(gate)*up` → `aicom.swiglu_canonicalized` |
| `qdq-legalize` | `QdqLegalize.cpp` | 双端 dequant + MatMul → `aicom.qdq_matmul_canonicalized` |
| `matmul-bias-fusion` | `MatMulBiasFusion.cpp` | `dot_general` + 常量 bias → `aicom.matmul_bias_fused` |
| `horizontal-gemm-fusion` | `HorizontalGemmFusion.cpp` | 共享 LHS 双 GEMM + concat → `aicom.horizontal_gemm_fused` |
| `elementwise-chain-legalize` | `ElementwiseChainLegalize.cpp` | add/mul → ReLU 链 → `aicom.elementwise_chain_fused` |
| `producer-consumer-legalize` | `ProducerConsumerLegalize.cpp` | GEMM → softmax 链 → `aicom.producer_consumer_fused` |
| `layout-bridge-legalize` | `LayoutBridgeLegalize.cpp` | Conv + NCHW→NHWC transpose → `aicom.layout_folded` |
| `kvcache-legalize` | `KVCacheLegalize.cpp` | decode 函数 K/V → `aicom.kvcache_boundary` |
| `graph-partition-legalize` | — | 分区边界标注 → `aicom.partition_boundary` |
| `conv-bn-const-fold` | — | Conv+BN 常量 weight 编译期计算 |

**中后端（9 个）**

| Pass | Stage | 做什么 |
|------|-------|--------|
| `custom-linalg-opt` | linalg | 常量 elementwise `linalg.generic` 编译期折叠 |
| `custom-buffer-opt` | bufferize | 小 `alloc` → `alloca`（跳过 return buffer） |
| `custom-loop-tiling` | loops/`scf-seq` | `scf.for` strip-mining |
| `custom-linalg-to-parallel-loops` | loops/`scf-par` | elementwise + 2D matmul → `scf.parallel` |
| `custom-affine-opt` | affine | 最外层 `affine.for` strip-mining |
| `custom-vector-opt` | vector | 静态 shape `transfer_*` → `vector.load/store` |
| `custom-llvm-cleanup` | llvm | LLVM dialect 上死 store / 死 op 清理 |

Pass 注册在 `include/AICompiler/Passes.td` 和 `lib/Transforms/RegisterPasses.cpp`。`pipe-demo` 以固定 pipeline 驱动，不把它们作为单独 CLI 参数开放；用 `--list-passes` 查看。

### Loop 路径（`--loop-mode`）

| 路径 | IR 轨迹 | 自定义 pass |
|------|---------|-------------|
| `scf-seq` | memref → `scf.for` → CF → LLVM | `custom-loop-tiling` |
| `scf-par` | memref → `scf.parallel` → CF → LLVM | `custom-linalg-to-parallel-loops` |
| `affine` | memref → `affine.for` → SCF → CF → LLVM | `custom-affine-opt` |
| `vector` | memref → Affine → `vector.*` → SCF → CF → LLVM | `custom-vector-opt` |

### mlir-opt Plugin

`AICompilerPlugin` 把 fusion pass 打包成可加载 `.so`，可独立于 `pipe-demo` 使用：

```bash
mlir-opt input.mlir \
  --load-dialect-plugin=./libAICompilerPlugin.so \
  --load-pass-plugin=./libAICompilerPlugin.so \
  '--pass-pipeline=builtin.module(aicom-fusion)' \
  -o output.mlir
```

---

## 测试体系

本仓有四类测试。多数测试最终调用 `pipe-demo`，区别是谁读输出、怎么判断对错（`test_mlir_opt_plugin` 走 `mlir-opt`）。

### 总览

| 类型 | Ninja target | 本仓 C++ 做什么 | 输入 | 判定方式 | 验证什么 |
|------|-------------|----------------|------|----------|----------|
| **Shell regression** | `test_shell_regression` | `pipe-demo` 跑各种 `.mlir`，输出 IR 或 JIT 结果到 stdout | `test/*.mlir` + `test/lit/*.mlir` | Bash `grep` 有/无某字符串 | pipeline 各路径不崩、fusion 标注存在、IR 里出现预期 op |
| **LIT/FileCheck** | `test_lit_filecheck` | 同上 | `test/lit/*.mlir`（14 个） | FileCheck 逐行匹配 `.mlir` 文件里的 `CHECK` 行 | 更精确的 IR 结构：某 op 的属性、标注位置 |
| **JIT 数值校验** | `test_jit_golden` | `pipe-demo --jit`：完整 pipeline → LLVM → JIT 执行，stdout 打印浮点数组 | 6 个 `.mlir` | Python 脚本 `run_jit_golden.py` 解析输出，和 NumPy 参考实现 `allclose` | 编译器整条链的数值正确性（本仓唯一执行生成代码的测试） |
| **跨仓 e2e** | `test_*_e2e` / `test_partition_smoke` | `pipe-demo --pipeline-stop-after=fusion`（或 linalg） | P4 导出或本仓手写 `.mlir` | 脚本 `grep` fusion 后 IR 里的标注 | 对应子图能被 fusion pass 识别 |

等价脚本在 `scripts/`：`test_shell_regression.sh`、`test_lit_filecheck.sh`、`run_jit_golden.sh`、`run_*_e2e.sh` 等。

### Shell Regression（17 项）

`ninja -C build test_shell_regression`

脚本 `scripts/test_shell_regression.sh` 依次用不同参数调 `pipe-demo`，对 stdout 做 `grep` 断言。不逐元素比对浮点结果。

覆盖三类场景：
- **全 pipeline**（#1–#6）：`matmul_add` / `conv_bn_relu` / `mini_model` 分别走 `scf-seq` / `scf-par` / `affine` / `vector` 到 LLVM，断言输出含 `llvm.func @inference`
- **Fusion 标注**（#7–#12）：Conv+BN 消除 `batch_norm`、ReLU→clamp、Softmax/RMSNorm/Attention 标注、常量折叠
- **Stop-after / JIT smoke / 兼容**（#13–#17）：停在 fusion 有 `stablehlo.convolution` 无 `llvm.func`；停在 affine 有 `affine.for`；JIT 输出含 `1.5`

### LIT/FileCheck（14 项，可选）

`ninja -C build test_lit_filecheck`

需要 `lit` + `FileCheck`（`apt install llvm-22-tools`）。`test/lit/*.mlir` 每个文件自带 `// CHECK` 行，FileCheck 逐行匹配 `pipe-demo` 输出的 IR。比 Shell regression 更精确——能检查 op 属性、标注位置、类型。

LIT 配置：`test/lit/lit.cfg.py` + CMake 生成的 `build/test/lit/lit.site.cfg.py`。产物在 `build/test/lit/Output/`（已 gitignore）。

### JIT 数值校验（6 项）

`ninja -C build test_jit_golden`

Python 脚本 `scripts/run_jit_golden.py` 对每个 case：调 `pipe-demo --jit`，用正则从 stdout 抽出 `JIT result (N elements): …`，和脚本里的 NumPy 参考实现做 `np.allclose`。

| Case | 输入 | 比什么 |
|------|------|--------|
| `matmul_add` | `test/matmul_add.mlir` | MatMul + Bias → `[1.5, 2.5, 3.5, 4.5]` |
| `jit_scale` | `test/jit_scale.mlir` | 常量 ×2 |
| `jit_gelu` | `test/jit_gelu.mlir` | GELU |
| `jit_gelu_p4` | `test/fixtures/gelu_p4_jit.mlir` | P4 导出的 GELU（`atol=1e-3`） |
| `jit_swiglu` | `test/jit_swiglu.mlir` | SwiGLU |
| `jit_swiglu_p4` | `test/fixtures/swiglu_p4_jit.mlir` | P4 导出的 SwiGLU |

这是本仓唯一「编译生成代码 → 执行 → 逐元素比对」的测试。兄弟仓的 `run_onnx_golden` 是 Python 用 ORT 查 `.onnx` fixture，不跑 JIT。

<a id="cross-repo-e2e"></a>

### 跨仓 e2e（10 项）

多数 e2e：兄弟仓交出 `.mlir` 后，本仓做两件事：

1. `parseSourceFile`：官方 StableHLO dialect 解析（文本合法才能过）。
2. `pipe-demo --pipeline-stop-after=fusion`：跑 `buildFusionStage`（`PipelineStages.cpp`），即 Conv+BN / Softmax / RMSNorm / Attention / RoPE / LayerNorm / GELU / SwiGLU / QDQ / MatMul+Bias / Horizontal GEMM / Layout / KVCache 等 **legalize/fusion pass**。这些 pass 匹配到子图后打上 `aicom.*` 属性，一般不改计算。

`test_broadcast_e2e` / `test_dynamic_e2e` 还会再跑 `--pipeline-stop-after=linalg`（官方 StableHLO→Linalg）。**不跑** bufferize / LLVM / JIT。判定就是对 stdout 做 `grep`：对应 pass 有没有打上标注。

`ninja test_e2e` 一次跑完。前置：兄弟仓 `cmake --build build --target run_lowering_l3 gen_lowering_models gen_quant_models`。

P4 导出 = `run_lowering_l3 --mlir-only <file.onnx>`。下表「P4：`foo.onnx`」都指这一条。不读兄弟仓 P12–P14 的自定义 IR。

| Target | 输入 | 做什么 | 过线条件 |
|--------|------|--------|----------|
| `test_attention_e2e` | P4：`lowering_attention.onnx` | fusion / `attention-legalize` | `aicom.scaled_dot_product_attention`、`aicom.softmax_canonicalized` |
| `test_transformer_e2e` | P4：多个 `lowering_*.onnx`（softmax / attention / rmsnorm / …） | fusion，按子图验对应 pass | 各子图对应的 `aicom.*`（见 `run_transformer_e2e.sh`） |
| `test_quant_e2e` | ① 本仓 `test/lit/qdq_legalize.mlir` ② P4：`lowering_qdq_matmul.onnx` ③ P4：`quant_qdq_matmul.onnx` | fusion / `qdq-legalize` | `aicom.qdq_matmul_canonicalized` |
| `test_layout_e2e` | P4：`lowering_layout_conv.onnx` | fusion / `layout-bridge-legalize` | 有 `aicom.layout_folded`，无 `stablehlo.transpose` |
| `test_kvcache_e2e` | ① 本仓 `test/lit/kvcache_legalize.mlir` ② P4：`lowering_decode_step.onnx`；若已构建再跑 P13 `run_memory_planning`（只看 stdout） | fusion / `kvcache-legalize` | `aicom.kvcache_boundary`；P13 stdout 含 `decode step` |
| `test_broadcast_e2e` | P4：`lowering_broadcast.onnx` | fusion，再 linalg | fusion 有 `broadcast_in_dim`；linalg 有 `linalg.` |
| `test_dynamic_e2e` | P4：`lowering_dynamic.onnx` 等 | fusion，再 linalg | `tensor<?x`、`stablehlo.dot_general`、`linalg.matmul` |
| `test_partition_smoke` | ① 兄弟仓 `run_graph_partition`（图在 P14 源码里）② 本仓 `test/lit/graph_partition_smoke.mlir` | 两步独立冒烟，中间不传文件 | ① stdout 有 `boundary tensors`；② fusion 后 IR 有 `aicom.partition_boundary` |
| `test_torch_e2e` | 本仓 `test/fixtures/conv_bn_torch.mlir`（或现场 torch-mlir 导出） | fusion / `conv-bn-fusion` | 有 `convolution`，无 `batch_norm` |
| `test_mlir_opt_plugin` | 本仓 `test/lit/constant_fold.mlir` | `mlir-opt` 加载 `aicom-fusion` | pipeline 跑通 |

### 全量测试与 IR 落盘

```bash
ninja -C build test_all             # Shell regression + LIT（不含 e2e）
ninja -C build test_e2e             # 全部跨仓 e2e（需兄弟仓 P4 二进制 + .onnx）
ninja -C build run_pipeline_demo    # 把各 stage IR 写到 output/pipeline-dumps/latest/（无断言）
```

---

## 与 mlir_compiler 的接口

交接方式见 [mlir_compiler README · 与 mlir_pass 的接口](../mlir_compiler/README.md#mlir-pass-interface)。本仓各 e2e target 见 [跨仓 e2e](#cross-repo-e2e)。

---

## 示例输入

| 文件 | 用途 |
|------|------|
| `test/mini_model.mlir` | Conv→BN→ReLU→MatMul→Add 全图 |
| `test/conv_bn_relu.mlir` | Conv+BN+ReLU fusion |
| `test/matmul_add.mlir` | MatMul+Add，JIT |
| `test/lit/softmax_legalize.mlir` | Softmax 分解标注 |
| `test/lit/constant_fold.mlir` | 双常量编译期折叠 |
| `test/lit/dynamic_batch.mlir` | 动态 batch smoke |
| `test/fixtures/attention_p4.mlir` | P4 导出的 Attention（跨仓 e2e） |
| `test/fixtures/rmsnorm_p4.mlir` | P4 导出的 RMSNorm |

---

## 相关文档

- [Affine/Vector lowering 设计](./docs/superpowers/specs/2026-06-16-affine-vector-lowering-design.md)
- [设计规格](./docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md)
- [实现计划](./docs/superpowers/plans/2026-05-23-mlir-ai-compiler-demo.md)
- 能力边界：[编译器能力映射.md](../mlir_compiler/src/mlir/gpu/docs/编译器能力映射.md)
- 学习路径：[两仓库学习路径与代码导读.md](../mlir_compiler/src/mlir/gpu/docs/两仓库学习路径与代码导读.md)
- CPU 侧 `mlir-opt` 命令链：[cpu README §2.5](../mlir_compiler/src/mlir/cpu/README.md)
