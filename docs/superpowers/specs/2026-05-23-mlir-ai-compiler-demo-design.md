# MLIR AI Compiler Pipeline Demo — Design Spec

**Date:** 2026-05-23 (rev. 2026-05-23 — sync with implementation)  
**Status:** Approved  
**Project:** `mlir_pass`  
**Implementation plan:** `docs/superpowers/plans/2026-05-23-mlir-ai-compiler-demo.md`

## Goal

Build a demo AI compiler pipeline in C++ using **official MLIR APIs** plus **per-stage custom teaching passes** that lowers a small AI model subgraph (**Conv + BatchNorm + ReLU + MatMul + Add**) from StableHLO dialect down to LLVM dialect, with JIT execution. The pipeline dumps IR after each **named stage** (and optionally after each pass) for teaching and verifies behavior via tests.

**Reference:** Conceptual stages mirror `mlir_compiler` (P5–P9 / L3→L4); this repo uses real MLIR passes wired in C++, not teaching-only header printers.

## Architecture

| Layer | Responsibility |
|-------|----------------|
| `AICompilerTransforms` | Stage `build*Stage` wrappers + custom passes under `lib/Transforms/` |
| `AICompilerPipeline` | Stage orchestration (`Pipeline.cpp`) |
| `pipe-demo` | CLI driver (`tools/ai-compiler-demo/main.cpp`) |

## Project layout

```
mlir_pass/
├── CMakeLists.txt                    # CMAKE_EXPORT_COMPILE_COMMANDS=ON
├── scripts/run_pipeline_demo.sh      # One-click per-stage IR dumps
├── scripts/test_shell_regression.sh  # Required Shell regression
├── scripts/test_lit_filecheck.sh     # Optional LIT/FileCheck regression
├── scripts/test_all.sh               # Shell regression + LIT/FileCheck
├── include/AICompiler/
│   ├── Passes.td / Passes.h
│   ├── Pipeline.h
│   └── PipelineStages.h
├── lib/
│   ├── Transforms/
│   │   ├── ConvBNFusion.cpp              # Stage 1 custom
│   │   ├── StableHLOToLinalg.cpp          # Stage 2
│   │   ├── LinalgOptimization.cpp        # Stage 2 (+ custom-linalg-opt)
│   │   ├── CustomLinalgOpt.cpp           # Stage 2 custom
│   │   ├── Bufferization.cpp             # Stage 3 (+ custom-buffer-opt)
│   │   ├── CustomBufferOpt.cpp             # Stage 3 custom
│   │   ├── LowerToLoops.cpp              # Stage 4 sequential path
│   │   ├── Vectorization.cpp             # Stage 4 parallel path (scf.parallel, NOT vector dialect)
│   │   ├── CustomLoopTiling.cpp            # Stage 4 custom (sequential path)
│   │   ├── CustomLinalgToParallelLoops.cpp # Stage 4 custom (parallel path)
│   │   ├── LLVMLowering.cpp                # Stage 5 (+ custom-llvm-cleanup)
│   │   ├── CustomLLVMCleanup.cpp           # Stage 5 custom
│   │   └── RegisterPasses.cpp
│   └── Pipeline/
│       └── Pipeline.cpp
├── tools/ai-compiler-demo/
└── test/
    ├── mini_model.mlir, conv_bn_relu.mlir, matmul_add.mlir
    └── lit/                              # Optional FileCheck → test_lit_filecheck
```

## Pipeline stages (this repo)

| Stage | `--pipeline-stop-after` | Official passes (summary) | Custom passes |
|-------|-------------------------|----------------------------|---------------|
| 1 | `fusion` | — | `conv-bn-fusion` |
| 2 | `linalg` | `stablehlo-legalize-to-linalg`, canonicalize, cse, linalg fusion/fold | `custom-linalg-opt` |
| 3 | `bufferize` | one-shot-bufferize, buffer-deallocation pipeline | `custom-buffer-opt` |
| 4 | `loops` | `convert-bufferization-to-memref`, linalg→loops **or** parallel loops, `convert-scf-to-cf` | `custom-loop-tiling` (seq.) **or** `custom-linalg-to-parallel-loops` (par.) |
| 5 | `llvm` | arith/cf/func/index/math/memref → LLVM, reconcile casts | `custom-llvm-cleanup` |

**Stage 4 CLI:** `--no-vectorize` selects sequential `scf.for` path; default selects parallel `scf.parallel` path. The flag name is historical — neither path uses the **vector dialect**.

## Dialect lowering: this repo vs production paths

### This repo (implemented)

```
StableHLO → Linalg (tensor) → bufferize → Linalg (memref)
  → SCF (scf.for or scf.parallel)
  → CF  (convert-scf-to-cf)
  → LLVM Dialect (convert-*-to-llvm, finalize-memref-to-llvm)
  → JIT (ExecutionEngine)
```

**Not used:** Affine dialect, Vector dialect.

### Production reference — Affine → LLVM (e.g. polyhedral / affine loop nests)

Typical CPU pipeline when taking the **affine branch** instead of direct linalg-to-loops:

```
Linalg (memref)
  → convert-linalg-to-affine-loops     # affine.for, affine.load/store, affine.apply
  → [affine-loop-normalize, affine opts, polyhedral scheduling]
  → lower-affine                         # affine → scf.for + memref.load/store
  → convert-scf-to-cf                    # scf → cf.br / cf.cond_br
  → convert-cf-to-llvm + finalize-memref-to-llvm + …
  → LLVM Dialect
```

**Does Affine go through CF?** Yes, in the usual pipeline: **Affine → SCF (`lower-affine`) → CF (`convert-scf-to-cf`) → LLVM**. Affine is not lowered straight to CF; `lower-affine` strips affine maps into SCF loops first.

### Production reference — Vector dialect → LLVM (SIMD)

Typical CPU pipeline when vectorizing after bufferization:

```
Linalg (memref)
  → linalg-vectorize / vectorization patterns   # vector.transfer_read/write, vector.fma, …
  → [optional: vector-transfer-to-scf for masked/complex transfers]
  → convert-scf-to-cf                             # only for loop control flow around vector ops
  → convert-vector-to-llvm                        # vector ops → llvm vector types / intrinsics
  → convert-cf-to-llvm + finalize-memref-to-llvm + …
  → LLVM Dialect
```

**Does Vector go through CF?** **Vector data ops do not.** `convert-vector-to-llvm` lowers SIMD-style operations **directly** to LLVM dialect (vector types or intrinsics). **CF is only involved for loops** that may surround vector code (SCF → CF → `convert-cf-to-llvm`). At the LLVM stage, **control flow (CF) and data parallelism (Vector→LLVM) are two converging lowering tracks**, both ending in LLVM dialect before `mlir-translate`.

See [`mlir_compiler/src/mlir/cpu/README.md`](../../../../mlir_compiler/src/mlir/cpu/README.md) §2.5 for a concrete pass order including `-convert-vector-to-llvm` alongside `-convert-scf-to-cf`.

## Relation to `mlir_compiler` gpu stages (6–12)

The sibling repo [`mlir_compiler/src/mlir/gpu/`](../../../../mlir_compiler/src/mlir/gpu/) teaches the same compiler story using **header-only IR** for stages 7–12, while **this repo (`mlir_pass`)** wires **real MLIR passes** in C++. Use the table below to navigate between them.

| `mlir_compiler` | Directory | Teaching style | Concepts | `mlir_pass` stage | Implemented as |
|-----------------|-----------|----------------|----------|-------------------|----------------|
| P5 | `6_stablehlo_passes/` | Real MLIR plugin | Conv+BN `OpRewritePattern` | `fusion` | `conv-bn-fusion` |
| P5 | `7_stablehlo_opt/` | `shlo_graph.h` | Graph opts, simplified conv+BN | — | Absorbed into Stage 2 official cleanup + Stage 1 fusion |
| P6 | `8_linalg_opt/` | `linalg_ir.h` | Dep analysis, linalg fusion, tiling metadata | `linalg` | Official linalg passes + `custom-linalg-opt` |
| P7 | `9_bufferize/` | `bufferize_ir.h` | OSB: alias, in-place, copies, dealloc | `bufferize` | Official bufferize + `custom-buffer-opt` |
| P8 | `10_scf_affine/` | `scf_affine_ir.h` | Loop nest, tiling, interchange, fusion, vec prep | `loops` (partial) | `convert-linalg-to-loops`, `custom-loop-tiling`, `scf-to-cf`; **no affine dialect** |
| P8 | `11_vector/` | `vector_ir.h` | vector.transfer, fma, contract, mask, LLVM intrinsics | — (non-goal) | Documented production path; `custom-linalg-to-parallel-loops` for loop parallelism only |
| P9 | `12_llvm_lowering/` | `llvm_lowering_ir.h` | Simulated Vector→LLVM, ISel, regalloc, asm | `llvm` | Real `convert-*-to-llvm`, `custom-llvm-cleanup`, JIT |

**Reading order:** Stage 6 ↔ `mlir_pass` fusion → gpu 8–9 ↔ `mlir_pass` linalg/bufferize → gpu 10–12 concepts ↔ `mlir_pass` loops/llvm + production Affine/Vector paths in this spec §Dialect lowering.

**Style contrast:** gpu 7–12 = custom structs + printed pseudo-IR; `mlir_pass` = `PassManager` on real MLIR + optional JIT/LIT.

For a **full real `mlir-opt` command chain** (including optional vector), see [`mlir_compiler` cpu README](../../../../mlir_compiler/src/mlir/cpu/README.md) §2.5.

## Why no custom Vector dialect pass in this repo

| Reason | Detail |
|--------|--------|
| **Teaching scope** | Demo targets the **main trunk** StableHLO→Linalg→memref→SCF→CF→LLVM; Vector is an **optional L4 branch** for SIMD. |
| **Workload size** | Test graphs (e.g. 2×2 matmul) are too small for meaningful vectorization; a custom pass would not show realistic speedups. |
| **Target coupling** | Vector lowering needs vector width, alignment, and target features (x86 AVX, RISC-V RVV, etc.); JIT and tests would become fragile. |
| **Duplication** | Production path already uses upstream `linalg-vectorize` + `convert-vector-to-llvm`; a thin wrapper adds little teaching value vs. existing custom SCF passes. |
| **Parallel alternative** | `custom-linalg-to-parallel-loops` teaches **loop-level parallelism** without SIMD ABI complexity. |

**Future extension (out of scope):** optional Stage 4b with official `-linalg-vectorize` + `-convert-vector-to-llvm` behind a flag, documented against the reference path above.

## Conv+BN fusion (custom)

For `batch_norm_inference(conv(x, w), γ, β, μ, σ)` with constant tensors:

- `scaleᵢ = γᵢ / sqrt(σᵢ + ε)`
- `w' = w * scale` (per output channel)
- `b' = β - μ * scale`
- Replace BN with `add(conv(x, w'), broadcast(b'))`

## Custom pass summary

| Pass | Stage | File |
|------|-------|------|
| `conv-bn-fusion` | fusion | `ConvBNFusion.cpp` |
| `custom-linalg-opt` | linalg | `CustomLinalgOpt.cpp` |
| `custom-buffer-opt` | bufferize | `CustomBufferOpt.cpp` |
| `custom-loop-tiling` | loops (seq.) | `CustomLoopTiling.cpp` |
| `custom-linalg-to-parallel-loops` | loops (par.) | `CustomLinalgToParallelLoops.cpp` |
| `custom-llvm-cleanup` | llvm | `CustomLLVMCleanup.cpp` |

## Workloads

- `test/mini_model.mlir` — full Conv→BN→ReLU→MatMul→Add
- `test/conv_bn_relu.mlir` — fusion unit test
- `test/matmul_add.mlir` — JIT-friendly MatMul+Add

## CLI

```
--input=<file>                Required StableHLO .mlir
--dump-ir                     Print IR after each stage (and each pass when enabled)
--pipeline-stop-after=STAGE   fusion | linalg | bufferize | loops | llvm | all
--jit                         JIT nullary @inference
--no-vectorize                Sequential scf.for loops (NOT “disable vector dialect”)
--entry-func=NAME             Default: inference
```

## Testing

| Mechanism | Command | Scope |
|-----------|---------|-------|
| Shell regression | `bash scripts/test_shell_regression.sh` or `ninja -C build test_shell_regression` | Full pipeline, fusion, stop-after, JIT |
| LIT only | `bash scripts/test_lit_filecheck.sh` or `ninja -C build test_lit_filecheck` | `test/lit/*.mlir` via lit + FileCheck |
| Shell regression + LIT | `bash scripts/test_all.sh` or `ninja -C build test_all` | Above + FileCheck on fusion IR |
| Demo dumps | `bash scripts/run_pipeline_demo.sh` or `ninja -C build run_pipeline_demo` | Per-stage IR files (not a test) |

LIT/FileCheck require `lit` and `FileCheck` at cmake configure time. See `README.md` §验证与 IR 落盘.

## Non-goals

- ONNX import, GPU codegen, mlir-opt plugin, ResNet-scale models
- **Affine dialect pipeline** in this repo (documented as production reference only)
- **Vector dialect / custom vector pass** (documented as production reference only; see above)
- Polyhedral scheduling, target-specific SIMD codegen

## Document map

| Document | Role |
|----------|------|
| This spec | **What** the system is, dialect paths, custom passes, non-goals |
| `plans/2026-05-23-mlir-ai-compiler-demo.md` | **How** it was built (tasks, verify commands) |
| `README.md` | **Runbook** — build, test, visualize, toolchain |
