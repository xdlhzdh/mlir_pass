#!/usr/bin/env bash
# One-shot demo: run full pipeline on all test models, save per-stage IR to output/.
# Matches Plan goal: per-stage IR dumps for teaching / inspection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO="${AI_COMPILER_DEMO:-${ROOT}/build/tools/ai-compiler-demo/ai-compiler-demo}"
OUT="${ROOT}/output/pipeline-dumps/latest"
VECTORIZE_FLAG=(--no-vectorize)

if [[ ! -x "${DEMO}" ]]; then
  echo "error: build first: cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=/usr/local && ninja -C build" >&2
  exit 1
fi

rm -rf "${OUT}"
mkdir -p "${OUT}"

MODELS=(
  "mini_model:${ROOT}/test/mini_model.mlir"
  "conv_bn_relu:${ROOT}/test/conv_bn_relu.mlir"
  "matmul_add:${ROOT}/test/matmul_add.mlir"
)

STAGES=(fusion linalg bufferize loops llvm)

echo "== AI compiler pipeline demo =="
echo "Output directory: ${OUT}"
echo

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  mlir="${entry#*:}"
  model_dir="${OUT}/${name}"
  mkdir -p "${model_dir}"

  echo ">> Model: ${name} (${mlir})"

  # Annotated dump: every pass inside each stage + stage banners (stderr merged).
  "${DEMO}" --input="${mlir}" "${VECTORIZE_FLAG[@]}" --dump-ir \
    >"${model_dir}/00-full-pipeline-with-pass-dumps.txt" 2>&1

  # Clean snapshot after each named stage (final IR only).
  for stage in "${STAGES[@]}"; do
    "${DEMO}" --input="${mlir}" "${VECTORIZE_FLAG[@]}" \
      --pipeline-stop-after="${stage}" \
      >"${model_dir}/after-${stage}.mlir" 2>&1
    echo "   wrote after-${stage}.mlir"
  done

  # Final LLVM IR (no dump-ir noise).
  "${DEMO}" --input="${mlir}" "${VECTORIZE_FLAG[@]}" \
    >"${model_dir}/after-llvm-final.mlir" 2>&1
  echo "   wrote after-llvm-final.mlir"
  echo
done

# JIT smoke (stdout only).
echo ">> JIT (matmul_add)"
"${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --jit "${VECTORIZE_FLAG[@]}" \
  | tee "${OUT}/matmul_add-jit.txt"
echo

echo "== Done =="
echo "Inspect IR under: ${OUT}/"
echo "  mini_model/after-fusion.mlir     — Conv+BN fused (StableHLO)"
echo "  mini_model/after-linalg.mlir     — Linalg on tensor"
echo "  mini_model/after-bufferize.mlir  — memref + buffers"
echo "  mini_model/after-loops.mlir      — scf/cf loops"
echo "  mini_model/after-llvm-final.mlir — LLVM dialect"
echo "  */00-full-pipeline-with-pass-dumps.txt — pass-by-pass trace (--dump-ir)"
echo
echo "Regression: bash test/run_tests.sh [--all]"
