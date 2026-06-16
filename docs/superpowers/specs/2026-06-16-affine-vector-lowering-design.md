# Affine & Vector Lowering — Design Spec

**Date:** 2026-06-16  
**Status:** Approved  
**Project:** `mlir_pass`  
**Parent spec:** `docs/superpowers/specs/2026-05-23-mlir-ai-compiler-demo-design.md`

## Goal

Extend the AI compiler demo with **Affine** and **Vector dialect** lowering paths alongside the existing SCF paths. Each new path uses upstream MLIR passes plus one custom teaching pass. All paths compile to LLVM and pass shell + LIT regression.

## CLI

| Parameter | Values | Default |
|-----------|--------|---------|
| `--loop-mode` | `scf-seq` \| `scf-par` \| `affine` \| `vector` | `scf-par` |
| `--pipeline-stop-after` | adds `affine` \| `vector` | `all` |

`--no-vectorize` remains as a deprecated alias for `--loop-mode=scf-seq`.

### stop-after validation

| loop-mode | Valid stop-after (beyond fusion/linalg/bufferize) |
|-----------|-----------------------------------------------------|
| `scf-seq`, `scf-par` | `loops`, `llvm`, `all` |
| `affine` | `affine`, `llvm`, `all` |
| `vector` | `vector`, `llvm`, `all` |

Mismatch combinations exit with an error.

## Pipeline stages

`PipelineStage` enum order: `Fusion → Linalg → Bufferize → Loops → Affine → Vector → LLVM → All`.

Only one lowering stage runs per invocation based on `--loop-mode`.

### Affine path (`--loop-mode=affine`)

```
convert-bufferization-to-memref
→ convert-linalg-to-affine-loops
→ custom-affine-opt
[→ lower-affine → convert-scf-to-cf]   (when continuing past affine stage)
```

`stop-after=affine` stops before `lower-affine`.

### Vector path (`--loop-mode=vector`)

Upstream has no standalone `linalg-vectorize` in this MLIR build. The vector path uses `affine-super-vectorize` (`createAffineVectorize`) with 1-D `vectorSizes={2}` and `vectorizeReductions=false` (matmul reductions are not vectorized).

```
convert-bufferization-to-memref
→ convert-linalg-to-affine-loops
→ affine-super-vectorize (createAffineVectorize)
→ custom-vector-opt
[→ lower-affine → convert-scf-to-cf]   (when continuing past vector stage)
```

`stop-after=vector` stops before `lower-affine`.

### LLVM stage

When `loop-mode=vector`, append `convert-vector-to-llvm` before `finalize-memref-to-llvm`.

## Custom passes

| Pass | File | Teaching focus |
|------|------|----------------|
| `custom-affine-opt` | `CustomAffineOpt.cpp` | Strip-mine outermost `affine.for` with constant bounds (tile=4) |
| `custom-vector-opt` | `CustomVectorOpt.cpp` | Lower static `vector.transfer_read/write` to `vector.load/store` |

## Testing

### Shell regression

- Full pipeline `matmul_add.mlir` with `--loop-mode=affine` and `--loop-mode=vector` → `llvm.func @inference`
- `stop-after=affine` → dump contains `affine.for`, no LLVM
- `stop-after=vector` → dump contains `vector.`, no LLVM
- Existing tests migrated to `--loop-mode=scf-seq` where applicable

### LIT

- `test/lit/affine_lowering.mlir` — CHECK `affine.for`
- `test/lit/vector_lowering.mlir` — CHECK `vector.`

## Non-goals

- Polyhedral scheduling, target-specific SIMD intrinsics selection
- JIT on vector path (IR-level verification only)
- Replacing existing SCF paths as default

## Files

New: `LowerToAffine.cpp`, `LowerToVectorDialect.cpp`, `CustomAffineOpt.cpp`, `CustomVectorOpt.cpp`, lit tests.  
Modified: `Pipeline*.h/cpp`, `LLVMLowering.cpp`, `main.cpp`, `Passes.td`, `CMakeLists.txt`, `README.md`, original spec/plan.
