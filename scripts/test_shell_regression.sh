#!/usr/bin/env bash
# Shell regression for pipe-demo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO="${ROOT}/build/tools/ai-compiler-demo/pipe-demo"

if [[ ! -x "${DEMO}" ]]; then
  echo "error: build pipe-demo first: ninja -C build" >&2
  exit 1
fi

echo "== matmul_add: full pipeline =="
out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --no-vectorize 2>&1)"
grep -q 'llvm.func @inference' <<<"${out}"

echo "== conv_bn_relu: full pipeline =="
out_conv="$("${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --no-vectorize 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_conv}"

echo "== mini_model: full pipeline =="
out_mini="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --no-vectorize 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_mini}"

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

echo "== matmul_add: JIT =="
jit_out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --jit --no-vectorize 2>&1)"
grep -q 'JIT result' <<<"${jit_out}"
grep -q '1.5' <<<"${jit_out}"

echo "Shell regression passed."
echo "All requested tests passed."
