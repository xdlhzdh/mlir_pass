# MLIR AI Compiler Pipeline Demo

基于官方 MLIR C++ API 的端到端编译演示：将 StableHLO 子图（Conv + BN + ReLU + MatMul + Add）经 **Linalg → Bufferize → Loops → LLVM** 降至 LLVM Dialect，支持 JIT、分阶段 IR 导出与自定义 teaching pass。

概念阶段对齐 [`mlir_compiler`](../mlir_compiler) P5–P9；MLIR/StableHLO 环境搭建见 [`mlir_compiler/src/mlir/cpu/README.md`](../mlir_compiler/src/mlir/cpu/README.md) §1.2–§1.5。设计细节见文末 [相关文档](#相关文档)。

---

## 快速开始

```bash
# 1. 构建（需已安装 /usr/local 的 MLIR + StableHLO）
cmake -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DSTABLEHLO_LIB_DIR=/usr/local/lib
ninja -C build

# 2. 回归测试
bash test/run_tests.sh

# 3. 运行完整 pipeline
./build/tools/ai-compiler-demo/ai-compiler-demo \
  --input=test/mini_model.mlir --no-vectorize

# 4.（可选）将各阶段 IR 落盘到 output/
bash scripts/run_pipeline_demo.sh
```

---

## 环境与工具链

### 依赖

| 组件 | 要求 |
|------|------|
| LLVM / MLIR | 源码安装至 `/usr/local`（或 `CMAKE_PREFIX_PATH`），含 `MLIRConfig.cmake` 与 `libMLIR*.a` |
| StableHLO | 与 MLIR 版本匹配的 headers / `libStablehlo*.a` |
| Ninja | 推荐构建工具 |
| lit + FileCheck | 可选，仅 LIT 回归需要 |

```bash
mlir-opt --version
ls /usr/local/lib/cmake/mlir/MLIRConfig.cmake
```

### 编译与测试工具的来源

| 用途 | 路径示例 | 说明 |
|------|----------|------|
| 编译链接 | `/usr/local` | `cmake install` 的自建 MLIR（见 `mlir_compiler` §1.3） |
| LIT 工具 | `/usr/lib/llvm-21/bin` 或 LLVM build 目录 | `apt install llvm-21-tools` 提供 `lit`/`FileCheck` |

Ubuntu 的 `llvm-21-dev` 虽含 MLIR CMake 配置，通常 **没有** 可链接的 `libMLIR*.a`，**不能** 用 `/usr/lib/llvm-21` 替代 `/usr/local` 作为 `CMAKE_PREFIX_PATH`。`lit`/`FileCheck` 与 MLIR 库 **无需同版本**（仅对 `ai-compiler-demo` 的 stdout 做文本匹配）。

启用 LIT 时：

```bash
sudo apt install llvm-21-tools
export PATH="/usr/lib/llvm-21/bin:$PATH"
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/usr/local -DSTABLEHLO_LIB_DIR=/usr/local/lib
```

---

## 构建

```bash
cmake -B build -G Ninja \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DSTABLEHLO_LIB_DIR=/usr/local/lib
ninja -C build
```

- `build/compile_commands.json` 供 clangd 使用（`CMAKE_EXPORT_COMPILE_COMMANDS=ON`）。
- 使用 `-fno-rtti` 以匹配 MLIR ABI（已在 CMake 中设置）。
- CMake 目标：`test-ai-compiler`（Shell 回归）、`run-pipeline-demo`（IR 落盘）；若找到 lit/FileCheck，另有 `check-ai-compiler`、`test-ai-compiler-all`。

---

## 驱动程序 `ai-compiler-demo`

二进制：`./build/tools/ai-compiler-demo/ai-compiler-demo`

### 常用命令

```bash
# 完整 pipeline
./build/tools/ai-compiler-demo/ai-compiler-demo \
  --input=test/mini_model.mlir --no-vectorize

# 分阶段 IR 打印（阶段间 + pass 间）
./build/tools/ai-compiler-demo/ai-compiler-demo \
  --input=test/conv_bn_relu.mlir --dump-ir --no-vectorize 2>&1 | less

# 在指定 stage 停止
./build/tools/ai-compiler-demo/ai-compiler-demo \
  --input=test/mini_model.mlir --pipeline-stop-after=bufferize --no-vectorize

# JIT（仅适用于无参、常量嵌入的 @inference）
./build/tools/ai-compiler-demo/ai-compiler-demo \
  --input=test/matmul_add.mlir --jit --no-vectorize
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--input=<file>` | 输入 StableHLO `.mlir`（必填） |
| `--dump-ir` | 将 IR 打印到 stderr |
| `--pipeline-stop-after=` | `fusion` \| `linalg` \| `bufferize` \| `loops` \| `llvm` \| `all` |
| `--jit` | JIT 执行入口函数 |
| `--no-vectorize` | loops 阶段走顺序 `scf.for`（推荐，JIT 稳定） |
| `--entry-func=` | JIT 入口名（默认 `inference`） |

`--no-vectorize` 控制的是 **SCF 顺序循环 vs `scf.parallel`**，与 MLIR **Vector 方言**无关（见 [Lowering 路径](#lowering-路径)）。

### 示例输入

| 文件 | 说明 |
|------|------|
| `test/mini_model.mlir` | Conv → BN → ReLU → MatMul → Add |
| `test/conv_bn_relu.mlir` | Conv + BN fusion |
| `test/matmul_add.mlir` | MatMul + Add，适合 JIT |

---

## Pipeline 与 Pass

编排：`lib/Pipeline/Pipeline.cpp`。每个 stage 在独立 `PassManager` 中运行；`--dump-ir` 时在 stage 边界打印 IR。

### Stage 总览

| Stage | `--pipeline-stop-after` | 自定义 pass | 说明 |
|-------|-------------------------|-------------|------|
| 1 | `fusion` | `conv-bn-fusion` | StableHLO 上融合 Conv+BN |
| 2 | `linalg` | `custom-linalg-opt` | legalize + 官方 linalg 优化 |
| 3 | `bufferize` | `custom-buffer-opt` | one-shot-bufferize + deallocation |
| 4 | `loops` | `custom-loop-tiling` 或 `custom-linalg-to-parallel-loops` | 见下表 |
| 5 | `llvm` | `custom-llvm-cleanup` | 降至 LLVM Dialect |

### Stage 1 — fusion

| # | Pass | 类型 |
|---|------|------|
| 1 | `conv-bn-fusion` | 自定义 |

将 `batch_norm_inference(conv(...))` 在常量权重条件下折叠进 conv 权重与 bias；公式见 [设计规格](docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md)。

### Stage 2 — linalg

| # | Pass | 类型 |
|---|------|------|
| 1 | `stablehlo-legalize-to-linalg` | 官方 |
| 2–3 | `canonicalize`, `cse` | 官方 |
| 4–5 | `linalg-fuse-elementwise-ops`, `linalg-fold-unit-extent-dims` | 官方 |
| 6 | `custom-linalg-opt` | 自定义 |
| 7–8 | `canonicalize`, `cse` | 官方 |

### Stage 3 — bufferize

| # | Pass | 类型 |
|---|------|------|
| 1 | `one-shot-bufferize` | 官方 |
| 2 | buffer deallocation pipeline | 官方 |
| 3 | `custom-buffer-opt` | 自定义 |
| 4–5 | `canonicalize`, `cse` | 官方 |

### Stage 4 — loops

**顺序路径（`--no-vectorize`，推荐）：**

| # | Pass | 类型 |
|---|------|------|
| 1 | `convert-bufferization-to-memref` | 官方 |
| 2 | `convert-linalg-to-loops` | 官方 → `scf.for` |
| 3 | `custom-loop-tiling` | 自定义 |
| 4 | `convert-scf-to-cf` | 官方 → `cf.*` |

**并行路径（默认未加 `--no-vectorize`）：**

| # | Pass | 类型 |
|---|------|------|
| 1 | `convert-bufferization-to-memref` | 官方 |
| 2 | `custom-linalg-to-parallel-loops` | 自定义 → `scf.parallel` |
| 3 | `convert-scf-to-cf` | 官方 → `cf.*` |

### Stage 5 — llvm

| # | Pass | 类型 |
|---|------|------|
| 1–7 | `convert-arith/cf/func/index/math-to-llvm`, `finalize-memref-to-llvm`, `reconcile-unrealized-casts` | 官方 |
| 8 | `custom-llvm-cleanup` | 自定义 |

自定义 pass 注册：`include/AICompiler/Passes.td`、`lib/Transforms/RegisterPasses.cpp`。

---

## Lowering 路径

### 本仓库实现的路径

```
StableHLO (tensor)
  → Linalg
  → bufferize → memref + linalg
  → SCF (scf.for 或 scf.parallel)
  → CF (convert-scf-to-cf)
  → LLVM Dialect
  → JIT (ExecutionEngine)
```

**未使用** Affine 方言与 Vector 方言 IR。

### 生产环境参考（Affine / Vector）

**Affine**（常见 CPU 分支）：`linalg → affine loops → lower-affine → SCF → convert-scf-to-cf → CF → LLVM`。Affine **不直接** 降到 CF，须经 `lower-affine` 转为 SCF。

**Vector**（SIMD 分支）：`linalg-vectorize` 等产生 `vector.*` 算子；`convert-vector-to-llvm` 将 **数据算子** 直连 LLVM，**不经 CF**；外围循环仍走 SCF → CF → `convert-cf-to-llvm`。两条 track 在 LLVM Dialect 阶段汇合。

| IR | 典型路径 | 经 CF？ |
|----|----------|--------|
| 循环（scf / affine） | → CF → `convert-cf-to-llvm` | 是 |
| Vector 数据算子 | `convert-vector-to-llvm` | 否 |

`mlir_compiler` [cpu README §2.5](../mlir_compiler/src/mlir/cpu/README.md) 给出含 `-convert-vector-to-llvm` 的完整 `mlir-opt` 命令示例。

### `--no-vectorize` 与 Vector 方言

| CLI | loops 阶段 | 产物 |
|-----|------------|------|
| `--no-vectorize` | `convert-linalg-to-loops` + `custom-loop-tiling` | 顺序 `scf.for` |
| （默认） | `custom-linalg-to-parallel-loops` | `scf.parallel` + 内层 `scf.for` |

本仓库未实现 Vector 方言 pass：测试图规模小、与目标 ISA 强相关、上游已有 `linalg-vectorize` + `convert-vector-to-llvm`；并行教学由 `custom-linalg-to-parallel-loops` 承担。详见 [设计规格](docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md)。

---

## 验证与 IR 落盘

回归与 demo 均调用 `ai-compiler-demo`，互不依赖。

| | `test/run_tests.sh` | `scripts/run_pipeline_demo.sh` |
|--|---------------------|----------------------------------|
| 目的 | 断言正确性（pass/fail） | 导出 IR 文件供阅读 |
| 输出 | 终端 + 退出码 | `output/pipeline-dumps/latest/` |
| CI | 适用 | 不适用 |

**推荐：** 改代码后先跑回归；需要对照各 stage IR 时再跑 demo。

### 回归测试

覆盖：三份模型全 pipeline、Conv+BN fusion、`--pipeline-stop-after=fusion`、JIT 数值（`1.5`）。可选 LIT 用 `test/lit/conv_bn_fusion.mlir`（FileCheck 校验 fusion 后无 `batch_norm_inference`）。

| 范围 | Bash | Ninja |
|------|------|-------|
| Shell（默认） | `bash test/run_tests.sh` | `ninja -C build test-ai-compiler` |
| 仅 LIT | `bash test/run_tests.sh --lit` | `ninja -C build check-ai-compiler` |
| Shell + LIT | `bash test/run_tests.sh --all` | `ninja -C build test-ai-compiler-all` |

**LIT / FileCheck：** LIT 执行 `// RUN:` 中的命令；FileCheck 对 stdout 做 `// CHECK:` 文本匹配。配置入口为 `build/test/lit/lit.site.cfg.py`（由 `test/lit/lit.site.cfg.py.in` 生成）；LIT 产物目录为 `build/test/lit/Output/`（勿提交）。

典型 CI：

```bash
ninja -C build
ninja -C build test-ai-compiler
ninja -C build test-ai-compiler-all   # 需 lit + FileCheck
```

### IR 落盘 Demo

```bash
bash scripts/run_pipeline_demo.sh
# 或
ninja -C build run-pipeline-demo
```

对每个模型生成：`00-full-pipeline-with-pass-dumps.txt`（pass 级 trace）、`after-{fusion,linalg,bufferize,loops,llvm}.mlir`、`matmul_add-jit.txt`。每次运行清空 `output/pipeline-dumps/latest/`（已 gitignore）。

---

## 与 mlir_compiler 对照

[`mlir_compiler` gpu](../mlir_compiler/src/mlir/gpu/) 用 header-only 教学 IR；本仓库用真实 MLIR PassManager。

| mlir_compiler | 目录 | mlir_pass stage | 本仓库实现 |
|---------------|------|-----------------|------------|
| P5 | `6_stablehlo_passes/` | fusion | `conv-bn-fusion`（真实 MLIR） |
| P5 | `7_stablehlo_opt/` | — | 概念并入 stage 2 + fusion |
| P6 | `8_linalg_opt/` | linalg | 官方 linalg pass + `custom-linalg-opt` |
| P7 | `9_bufferize/` | bufferize | OSB + `custom-buffer-opt` |
| P8 | `10_scf_affine/` | loops（部分） | `scf.for` + tiling，无 affine |
| P8 | `11_vector/` | — | 未实现；见 Lowering 路径 |
| P9 | `12_llvm_lowering/` | llvm | 真实 LLVM lowering + JIT |

完整 `mlir-opt` 命令链见 [mlir_compiler cpu README](../mlir_compiler/src/mlir/cpu/README.md)。

---

## 相关文档

- [设计规格](docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md) — 架构、非目标、Affine/Vector 参考路径
- [实现计划](docs/superpowers/plans/2026-05-23-mlir-ai-compiler-demo.md) — 任务清单与验证命令
