# MLIR AI Compiler Pipeline Demo

将 StableHLO 子图（Conv + BN + ReLU + MatMul + Add）经 **Linalg → Bufferize → Loops → LLVM** 降至 LLVM Dialect 的 C++ 演示工程：官方 MLIR Pass + **6 个自定义 teaching pass**，支持 JIT、分阶段 IR 导出与回归测试。

- 概念对齐：[`mlir_compiler`](../mlir_compiler/src/mlir/README.md) P5–P9  
- 环境安装：[`mlir_compiler` cpu README](../mlir_compiler/src/mlir/cpu/README.md) §1.2–§1.5  
- 设计细节：[相关文档](#相关文档)

---

## 环境准备

### 依赖

| 组件 | 要求 |
|------|------|
| LLVM / MLIR | 安装到 `/usr/local`（或设置 `CMAKE_PREFIX_PATH`），含 `MLIRConfig.cmake`、`libMLIR*.a` |
| StableHLO | 与 MLIR 版本匹配的 headers / 静态库 |
| Ninja | 推荐 |
| LIT + FileCheck | 可选；仅 LIT 回归（`test_lit_filecheck`）需要 |

```bash
mlir-opt --version
ls /usr/local/lib/cmake/mlir/MLIRConfig.cmake
```

### 工具链说明

| 用途 | 典型路径 | 说明 |
|------|----------|------|
| 编译链接 MLIR | `/usr/local` | 源码 `cmake install`（`mlir_compiler` §1.3） |
| LIT 工具 | `/usr/lib/llvm-21/bin` 等 | `apt install llvm-21-tools` 提供 `lit`、`FileCheck` |

`llvm-21-dev` 通常 **不含** 可链接的 `libMLIR*.a`，**不能** 用 `/usr/lib/llvm-21` 作为 `CMAKE_PREFIX_PATH`。`lit`/`FileCheck` 与 MLIR **无需同版本**。

启用 LIT 示例：

```bash
sudo apt install llvm-21-tools
export PATH="/usr/lib/llvm-21/bin:$PATH"
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/usr/local -DSTABLEHLO_LIB_DIR=/usr/local/lib
```

LIT 配置由两层组成：`test/lit/lit.cfg.py` 定义测试套件规则；CMake 生成的 `build/test/lit/lit.site.cfg.py` 注入构建目录中的 `pipe-demo` 路径，并把测试里的 `%pipe-demo` 替换成该路径。LIT 产物在 `build/test/lit/Output/`（勿提交）。

---

## 编译与构建

```bash
cmake -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DSTABLEHLO_LIB_DIR=/usr/local/lib
ninja -C build
```

- `build/compile_commands.json`：IDE / clangd（`CMAKE_EXPORT_COMPILE_COMMANDS=ON`）
- `-fno-rtti`：匹配 MLIR ABI（CMake 已设置）
- CMake 目标：`test_pipeline_demo`（Shell 回归）、`run_pipeline_demo`（IR 落盘）；若找到 lit/FileCheck，另有 `test_lit_filecheck`、`test_all`。

### 命令速查

按自然执行顺序列出本工程常用命令。`test_pipeline_demo` 与 `test_lit_filecheck` 是 **不同** 的 Ninja target：前者跑 Shell 回归，后者只跑 LIT/FileCheck。

| 阶段 | Bash / 可执行文件 | Ninja | 作用 |
|------|-------------------|-------|------|
| 配置 | `cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/usr/local -DSTABLEHLO_LIB_DIR=/usr/local/lib` | — | 生成构建目录；检测 lit/FileCheck；生成 `compile_commands.json` |
| 编译全部 | — | `ninja -C build` | 编译静态库与 `pipe-demo` |
| 编译驱动 | — | `ninja -C build pipe-demo` | 只构建驱动程序及其依赖 |
| 运行 pipeline | `./build/tools/ai-compiler-demo/pipe-demo --input=test/mini_model.mlir --no-vectorize` | — | 直接运行完整 pipeline |
| Pipeline 回归 | `bash scripts/test_pipeline_demo.sh` | `ninja -C build test_pipeline_demo` | 用 Bash + `grep` 断言 pipeline、fusion、stop-after、JIT 正确 |
| LIT/FileCheck | `bash scripts/test_lit_filecheck.sh` | `ninja -C build test_lit_filecheck` | 只执行 `test/lit/*.mlir` 的 FileCheck 用例 |
| 全量测试 | `bash scripts/test_all.sh` | `ninja -C build test_all` | 先 Pipeline 回归，再 LIT/FileCheck |
| IR 落盘 Demo | `bash scripts/run_pipeline_demo.sh` | `ninja -C build run_pipeline_demo` | 生成各 stage IR 与 pass trace 到 `output/pipeline-dumps/latest/` |

最小工作流：

```bash
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/usr/local -DSTABLEHLO_LIB_DIR=/usr/local/lib
ninja -C build
ninja -C build test_pipeline_demo
./build/tools/ai-compiler-demo/pipe-demo --input=test/mini_model.mlir --no-vectorize
```

完整验证与演示工作流：

```bash
ninja -C build test_pipeline_demo   # Pipeline 回归
ninja -C build test_lit_filecheck   # 仅 LIT/FileCheck（需要 lit + FileCheck）
ninja -C build test_all             # Pipeline 回归 + LIT/FileCheck
ninja -C build run_pipeline_demo    # IR 落盘到 output/pipeline-dumps/latest/
```

对应脚本入口：

```bash
bash scripts/test_pipeline_demo.sh
bash scripts/test_lit_filecheck.sh
bash scripts/test_all.sh
bash scripts/run_pipeline_demo.sh
```

---

## 运行 `pipe-demo`

路径：`./build/tools/ai-compiler-demo/pipe-demo`

### 示例

```bash
# 完整 pipeline（推荐加 --no-vectorize）
./build/tools/ai-compiler-demo/pipe-demo \
  --input=test/mini_model.mlir --no-vectorize

# 打印每个 pass 后的 IR
./build/tools/ai-compiler-demo/pipe-demo \
  --input=test/conv_bn_relu.mlir --dump-ir --no-vectorize 2>&1 | less

# 在指定 stage 停止
./build/tools/ai-compiler-demo/pipe-demo \
  --input=test/mini_model.mlir --pipeline-stop-after=bufferize --no-vectorize

# JIT（matmul_add，检查数值）
./build/tools/ai-compiler-demo/pipe-demo \
  --input=test/matmul_add.mlir --jit --no-vectorize
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--input=<file>` | StableHLO `.mlir`（必填） |
| `--dump-ir` | 阶段间 + pass 间 IR 输出到 stderr |
| `--pipeline-stop-after=` | `fusion` \| `linalg` \| `bufferize` \| `loops` \| `llvm` \| `all` |
| `--jit` | JIT 执行 `--entry-func` |
| `--no-vectorize` | loops 用顺序 `scf.for`（非 Vector 方言；见 [Lowering 路径摘要](#lowering-路径摘要)） |
| `--entry-func=` | JIT 入口（默认 `inference`） |

### 示例输入

| 文件 | 用途 |
|------|------|
| `test/mini_model.mlir` | 全图 Conv→BN→ReLU→MatMul→Add |
| `test/conv_bn_relu.mlir` | Conv+BN fusion |
| `test/matmul_add.mlir` | MatMul+Add，JIT |

---

## Pipeline 与 Pass

编排入口是 `lib/Pipeline/Pipeline.cpp`。每个 stage 使用独立 `PassManager`，`--pipeline-stop-after` 控制停在哪个 stage，`--dump-ir` 用来观察 stage 和 pass 之间的 IR。

### Pipeline 总览

| Stage | 停止参数 | 主要 IR 变化 | 自定义 pass |
|-------|----------|--------------|-------------|
| 1 fusion | `--pipeline-stop-after=fusion` | StableHLO 图优化，Conv+BN 融合后仍是 StableHLO | `conv-bn-fusion` |
| 2 linalg | `--pipeline-stop-after=linalg` | StableHLO tensor ops → Linalg tensor ops | `custom-linalg-opt` |
| 3 bufferize | `--pipeline-stop-after=bufferize` | tensor → memref，插入 buffer 管理 | `custom-buffer-opt` |
| 4 loops | `--pipeline-stop-after=loops` | Linalg/memref → SCF，再降到 CF | `custom-loop-tiling` 或 `custom-linalg-to-parallel-loops` |
| 5 llvm | `--pipeline-stop-after=llvm` | arith/cf/func/memref 等 → LLVM Dialect | `custom-llvm-cleanup` |

### 自定义 Pass

| Pass 注册名 | Stage / 路径 | 源文件 | 教学重点 |
|-------------|--------------|--------|----------|
| `conv-bn-fusion` | fusion | `lib/Transforms/ConvBNFusion.cpp` | 用 rewrite pattern 将 BN 折叠进 Conv 权重和 bias |
| `custom-linalg-opt` | linalg | `lib/Transforms/CustomLinalgOpt.cpp` | 对常量输入的 elementwise `linalg.generic` 做编译期折叠 |
| `custom-buffer-opt` | bufferize | `lib/Transforms/CustomBufferOpt.cpp` | 将小的 `memref.alloc` 提升为 `alloca`，但跳过 return buffer |
| `custom-loop-tiling` | loops / `--no-vectorize` | `lib/Transforms/CustomLoopTiling.cpp` | 对 `scf.for` 做 strip-mining，跳过带 `iter_args` 的循环 |
| `custom-linalg-to-parallel-loops` | loops / 默认 | `lib/Transforms/CustomLinalgToParallelLoops.cpp` | 将 elementwise `generic` 和 2D `matmul` 自定义 lowering 到 `scf.parallel` |
| `custom-llvm-cleanup` | llvm | `lib/Transforms/CustomLLVMCleanup.cpp` | 在 LLVM dialect 上做死 store / 死 op 清理 |

这些 pass 注册在 `include/AICompiler/Passes.td` 和 `lib/Transforms/RegisterPasses.cpp`。表中的名字是 MLIR pass 注册名，会出现在 `--help` 和 `--dump-ir` 相关输出中；`pipe-demo` 运行的是固定 pipeline，不把它们作为 `--custom-linalg-opt` 这样的单独 CLI 参数开放。

```bash
# 查看本仓库注册的自定义 pass 名
./build/tools/ai-compiler-demo/pipe-demo --help 2>&1 | grep -E 'conv-bn|custom-'
```

### Loops Stage 的两条路径

`--no-vectorize` 是历史命名：它只是在本 demo 中选择顺序循环路径，并不表示本仓库实现了 Vector 方言。

| 运行方式 | loops stage 行为 | 会经过的自定义 pass |
|----------|------------------|---------------------|
| 默认，不加 `--no-vectorize` | Linalg/memref → `scf.parallel` → CF | `custom-linalg-to-parallel-loops` |
| 加 `--no-vectorize` | Linalg/memref → `scf.for` → loop tiling → CF | `custom-loop-tiling` |

```bash
# 观察顺序 loops 路径
./build/tools/ai-compiler-demo/pipe-demo \
  --input=test/matmul_add.mlir \
  --pipeline-stop-after=loops --no-vectorize

# 观察并行 loops 路径
./build/tools/ai-compiler-demo/pipe-demo \
  --input=test/matmul_add.mlir \
  --pipeline-stop-after=loops
```

### Pass 顺序

| Stage | Pass 顺序 |
|-------|-----------|
| fusion | `conv-bn-fusion` |
| linalg | `stablehlo-legalize-to-linalg` → `canonicalize` / `cse` → `linalg-fuse-elementwise-ops` → `linalg-fold-unit-extent-dims` → `custom-linalg-opt` → `canonicalize` / `cse` |
| bufferize | `one-shot-bufferize` → buffer deallocation pipeline → `custom-buffer-opt` → `canonicalize` / `cse` |
| loops / `--no-vectorize` | `convert-bufferization-to-memref` → `convert-linalg-to-loops` → `custom-loop-tiling` → `convert-scf-to-cf` |
| loops / 默认 | `convert-bufferization-to-memref` → `custom-linalg-to-parallel-loops` → `convert-scf-to-cf` |
| llvm | arith/cf/func/index/math/memref lowering → `reconcile-unrealized-casts` → `custom-llvm-cleanup` |

本仓库没有提供单独的 `mlir-opt` pass plugin target；如果要像上游 MLIR 那样手写 pass pipeline，需要另行封装插件或使用 MLIR pass pipeline 工具链。

### Lowering 路径摘要

本仓库实际实现的路径：

```text
StableHLO → Linalg → bufferize(memref) → SCF → CF → LLVM Dialect → JIT
```

本仓库没有实现 Affine / Vector 方言 IR。实际生产场景中，Affine 通常走 `linalg → affine → lower-affine → SCF → CF → LLVM`，需要经过 SCF/CF；Vector 数据算子通常由 `convert-vector-to-llvm` 直接转 LLVM dialect，不经过 CF，但控制流循环仍会走 SCF → CF → LLVM。

详见 [设计规格](docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md)。

---

## 验证与 IR 落盘

两者均调用 `pipe-demo`，**互不调用**。完整命令见 [命令速查](#命令速查)。

| | 回归 `scripts/test_pipeline_demo.sh` | Demo `scripts/run_pipeline_demo.sh` |
|--|--------------------------------------|--------------------------------------|
| 目的 | 判断对错（退出码） | 保存 IR 供阅读 |
| 输出 | 终端 | `output/pipeline-dumps/latest/` |
| 断言 | 有 | 无 |

### Shell 回归内容

`test_pipeline_demo` 等价于 `bash scripts/test_pipeline_demo.sh`。它会调用 `pipe-demo`，用 shell 断言检查 pipeline 行为；这里的 shell 断言主要指 `grep` 这类命令：例如检查输出中必须包含 `llvm.func @inference`，fusion 后不能再出现 `batch_norm_inference`，JIT 输出必须包含 `JIT result` 和 `1.5`。成功时大致输出：

```text
== matmul_add: full pipeline ==
== conv_bn_relu: full pipeline ==
== mini_model: full pipeline ==
== conv_bn_relu: fusion removes batch_norm ==
== stop-after fusion ==
== matmul_add: JIT ==
Shell regression passed.
All requested tests passed.
```

| 检查项 | 说明 |
|--------|------|
| 全 pipeline | 三份 `.mlir` 输出含 `llvm.func @inference` |
| Fusion | fusion 阶段无 `batch_norm_inference` |
| stop-after | `fusion` 阶段不到 LLVM |
| JIT | `matmul_add` 输出 `JIT result` 与 `1.5` |

### LIT（可选）

`test/lit/conv_bn_fusion.mlir`：fusion 后 FileCheck 确认无 `batch_norm_inference`。

### Demo 输出结构

```
output/pipeline-dumps/latest/
├── mini_model/after-{fusion,linalg,bufferize,loops,llvm}.mlir
├── mini_model/00-full-pipeline-with-pass-dumps.txt
├── conv_bn_relu/ …
├── matmul_add/ …
└── matmul_add-jit.txt
```

每次运行清空该目录（已 gitignore）。

---

## 与 mlir_compiler 对照

| mlir_compiler | mlir_pass stage | 说明 |
|---------------|-----------------|------|
| `6_stablehlo_passes/` | fusion | 真实 MLIR `conv-bn-fusion` |
| `7_stablehlo_opt/` | — | 图优化概念并入 stage 2 |
| `8_linalg_opt/` | linalg | + `custom-linalg-opt` |
| `9_bufferize/` | bufferize | + `custom-buffer-opt` |
| `10_scf_affine/` | loops（部分） | SCF + tiling，无 affine 方言 |
| `11_vector/` | — | 未实现 |
| `12_llvm_lowering/` | llvm | 真实 LLVM + JIT |

[`mlir_compiler` gpu](../mlir_compiler/src/mlir/gpu/) 为 header-only 教学 IR；本仓库为真实 `PassManager`。CPU `mlir-opt` 命令链见 [cpu README §2.5](../mlir_compiler/src/mlir/cpu/README.md)。

---

## 相关文档

- [设计规格](docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md)
- [实现计划](docs/superpowers/plans/2026-05-23-mlir-ai-compiler-demo.md)
