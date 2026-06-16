#!/usr/bin/env bash
# Shell regression for pipe-demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO="${ROOT}/build/tools/ai-compiler-demo/pipe-demo"

if [[ ! -x "${DEMO}" ]]; then
  echo "error: build pipe-demo first: ninja -C build" >&2
  exit 1
fi

echo "== matmul_add: full pipeline (scf-seq) =="
out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=scf-seq 2>&1)"
grep -q 'llvm.func @inference' <<<"${out}"

echo "== conv_bn_relu: full pipeline (scf-seq) =="
out_conv="$("${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --loop-mode=scf-seq 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_conv}"

echo "== mini_model: full pipeline (scf-seq) =="
out_mini="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --loop-mode=scf-seq 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_mini}"

echo "== matmul_add: full pipeline (affine) =="
out_affine="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=affine 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_affine}"

echo "== matmul_add: full pipeline (vector) =="
out_vector="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=vector 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_vector}"

echo "== conv_bn_relu: fusion removes batch_norm =="
dump="$("${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --dump-ir 2>&1)"
fusion_block="$(sed -n '/IR Dump After Stage: fusion/,/IR Dump After Stage:/p' <<<"${dump}")"
if grep -q batch_norm_inference <<<"${fusion_block}"; then
  echo "error: batch_norm_inference still present after fusion stage" >&2
  exit 1
fi

echo "== stop-after fusion =="
stop_out="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.convolution' <<<"${stop_out}"
grep -qv 'llvm.func @inference' <<<"${stop_out}" || {
  echo "error: --pipeline-stop-after=fusion should not reach LLVM" >&2
  exit 1
}

echo "== stop-after affine =="
affine_dump="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=affine \
  --pipeline-stop-after=affine --dump-ir 2>&1)"
grep -q 'affine.for' <<<"${affine_dump}"
grep -qv 'llvm.func @inference' <<<"${affine_dump}" || {
  echo "error: --pipeline-stop-after=affine should not reach LLVM" >&2
  exit 1
}

echo "== stop-after vector =="
vector_dump="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=vector \
  --pipeline-stop-after=vector --dump-ir 2>&1)"
grep -q 'vector\.' <<<"${vector_dump}"
grep -qv 'llvm.func @inference' <<<"${vector_dump}" || {
  echo "error: --pipeline-stop-after=vector should not reach LLVM" >&2
  exit 1
}

echo "== matmul_add: JIT (scf-seq) =="
jit_out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --jit --loop-mode=scf-seq 2>&1)"
grep -q 'JIT result' <<<"${jit_out}"
grep -q '1.5' <<<"${jit_out}"

echo "== deprecated --no-vectorize alias =="
legacy_out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --no-vectorize 2>&1)"
grep -q 'llvm.func @inference' <<<"${legacy_out}"

echo "Shell regression passed."
echo "All requested tests passed."
