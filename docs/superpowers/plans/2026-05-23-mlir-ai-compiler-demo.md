# MLIR AI Compiler Pipeline Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Modular C++ demo: StableHLO (Conv+BN+ReLU+MatMul+Add) → LLVM, custom passes at each stage, per-stage IR dumps, optional JIT.

**Design spec (authoritative for architecture):** `docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md`

**Runbook:** `README.md`

---

## Task 1: CMake + TableGen scaffolding ✅

- [x] Root CMake, `Passes.td`, libs `AICompilerTransforms`, `AICompilerPipeline`, tool `pipe-demo`
- [x] `CMAKE_EXPORT_COMPILE_COMMANDS=ON`

## Task 2: Conv+BN fusion pass ✅

- [x] `lib/Transforms/ConvBNFusion.cpp` — weight scale + bias broadcast add
- [x] TableGen `conv-bn-fusion`

## Task 3: Stage modules (Plan layout) ✅

- [x] `StableHLOToLinalg.cpp`
- [x] `LinalgOptimization.cpp` — elementwise fusion + fold unit dims
- [x] `Bufferization.cpp`
- [x] `LowerToLoops.cpp` / `Vectorization.cpp`
- [x] `LLVMLowering.cpp`
- [x] `PipelineStages.h` + stage wiring in `Pipeline.cpp`

## Task 4: Pipeline orchestration ✅

- [x] `Pipeline.cpp` — run stages sequentially with optional labeled dumps
- [x] `--pipeline-stop-after=fusion|linalg|bufferize|loops|llvm|all`

## Task 5: Driver + tests ✅

- [x] `test/mini_model.mlir`, `conv_bn_relu.mlir`, `matmul_add.mlir`
- [x] `scripts/test_shell_regression.sh`
- [x] JIT on `matmul_add.mlir`

## Task 6: Documentation ✅

- [x] Design spec + README

## Task 7: LIT (optional) ✅

- [x] `test/lit/conv_bn_fusion.mlir` + `test/CMakeLists.txt` (`test_lit_filecheck` when `lit`+`FileCheck` found)

---

## Task 8: Per-stage custom teaching passes ✅

Spec §Custom pass summary; each stage has at least one custom pass beyond upstream APIs.

- [x] `custom-linalg-opt` — `CustomLinalgOpt.cpp` (constant-fold elementwise `linalg.generic`)
- [x] `custom-buffer-opt` — `CustomBufferOpt.cpp` (small `alloc` → `alloca`)
- [x] `custom-loop-tiling` — `CustomLoopTiling.cpp` (strip-mine `scf.for`, skip loops with `iter_args`)
- [x] `custom-linalg-to-parallel-loops` — `CustomLinalgToParallelLoops.cpp` (replaces official parallel-loops in parallel path)
- [x] `custom-llvm-cleanup` — `CustomLLVMCleanup.cpp` (dead store / dead op on LLVM dialect)
- [x] Integrated in respective `build*Stage` functions + `RegisterPasses.cpp`

## Task 9: Demo automation + docs sync ✅

- [x] `scripts/run_pipeline_demo.sh` + CMake target `run_pipeline_demo`
- [x] README: toolchain ( `/usr/local` vs `llvm-*-tools` ), testing scripts vs LIT/FileCheck targets
- [x] README + spec: dialect paths (SCF/CF/LLVM in-repo; Affine/Vector as production reference)
- [x] README + spec: `mlir_compiler` gpu Stage 6–12 ↔ `mlir_pass` cross-reference table
- [x] Spec/plan alignment (this revision)

---

## Verify

```bash
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/usr/local -DSTABLEHLO_LIB_DIR=/usr/local/lib
ninja -C build
ninja -C build test_shell_regression              # Shell regression (= bash scripts/test_shell_regression.sh)
ninja -C build test_lit_filecheck             # LIT only (= bash scripts/test_lit_filecheck.sh)
ninja -C build test_all                       # Shell regression + LIT (= bash scripts/test_all.sh)
ninja -C build run_pipeline_demo             # optional IR dumps(= bash scripts/run_pipeline_demo.sh)
```

---

## Explicitly not in plan (see spec non-goals)

- Polyhedral scheduling, target-specific SIMD codegen

## Task 10: Affine & Vector lowering paths ✅

Spec: `docs/superpowers/specs/2026-06-16-affine-vector-lowering-design.md`

- [x] `--loop-mode=scf-seq|scf-par|affine|vector` (replaces `--no-vectorize` as primary CLI)
- [x] `buildAffineStage` + `custom-affine-opt`
- [x] `buildVectorDialectStage` + `custom-vector-opt` + `convert-vector-to-llvm`
- [x] `--pipeline-stop-after=affine|vector`
- [x] Shell regression + LIT tests
- [x] README + spec sync
