#!/usr/bin/env bash
# Shell regression for pipe-demo.
#
# Progress lines use: == <scenario>: <input.mlir> [<flags>] → <expectation> ==
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO="${ROOT}/build/tools/pipe-demo/pipe-demo"

if [[ ! -x "${DEMO}" ]]; then
  echo "error: build pipe-demo first: ninja -C build" >&2
  exit 1
fi

echo "== full pipeline → LLVM: matmul_add.mlir (loop-mode=scf-seq) =="
out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=scf-seq 2>&1)"
grep -q 'llvm.func @inference' <<<"${out}"

echo "== full pipeline → LLVM: matmul_add.mlir (loop-mode=scf-par, default) =="
out_par="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=scf-par 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_par}"

echo "== full pipeline → LLVM: conv_bn_relu.mlir (loop-mode=scf-seq) =="
out_conv="$("${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --loop-mode=scf-seq 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_conv}"

echo "== full pipeline → LLVM: mini_model.mlir (loop-mode=scf-seq) =="
out_mini="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --loop-mode=scf-seq 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_mini}"

echo "== full pipeline → LLVM: matmul_add.mlir (loop-mode=affine) =="
out_affine="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=affine 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_affine}"

echo "== full pipeline → LLVM: matmul_add.mlir (loop-mode=vector) =="
out_vector="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=vector 2>&1)"
grep -q 'llvm.func @inference' <<<"${out_vector}"

echo "== fusion stage: conv_bn_relu.mlir (--dump-ir) → no batch_norm_inference =="
dump="$("${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --dump-ir 2>&1)"
fusion_block="$(sed -n '/IR Dump After Stage: fusion/,/IR Dump After Stage:/p' <<<"${dump}")"
if grep -q batch_norm_inference <<<"${fusion_block}"; then
  echo "error: batch_norm_inference still present after fusion stage" >&2
  exit 1
fi

echo "== fusion stage: conv_bn_relu.mlir → ReLU canonicalized to clamp =="
grep -q 'stablehlo.clamp' <<<"${fusion_block}"
if grep -q 'stablehlo.maximum' <<<"${fusion_block}"; then
  echo "error: stablehlo.maximum should be canonicalized to stablehlo.clamp" >&2
  exit 1
fi

echo "== fusion stage: softmax_legalize.mlir → softmax divide annotated =="
softmax_out="$("${DEMO}" --input="${ROOT}/test/lit/softmax_legalize.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.softmax_canonicalized' <<<"${softmax_out}"

echo "== fusion stage: rmsnorm_legalize.mlir → RMSNorm multiply annotated =="
rmsnorm_out="$("${DEMO}" --input="${ROOT}/test/lit/rmsnorm_legalize.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.rmsnorm_canonicalized' <<<"${rmsnorm_out}"

echo "== fusion stage: attention_legalize.mlir → SDP attention annotated =="
attn_out="$("${DEMO}" --input="${ROOT}/test/lit/attention_legalize.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.scaled_dot_product_attention' <<<"${attn_out}"

echo "== fusion stage: rope_legalize.mlir → RoPE add annotated =="
rope_out="$("${DEMO}" --input="${ROOT}/test/lit/rope_legalize.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.rope_canonicalized' <<<"${rope_out}"

echo "== fusion stage: layernorm_legalize.mlir → LayerNorm add annotated =="
ln_out="$("${DEMO}" --input="${ROOT}/test/lit/layernorm_legalize.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.layernorm_canonicalized' <<<"${ln_out}"

echo "== fusion stage: constant_fold.mlir → add/mul folded to constant =="
fold_out="$("${DEMO}" --input="${ROOT}/test/lit/constant_fold.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.constant' <<<"${fold_out}"
if grep -q 'stablehlo.add' <<<"${fold_out}" || grep -q 'stablehlo.multiply' <<<"${fold_out}"; then
  echo "error: constant_fold.mlir should not retain stablehlo.add/multiply after fusion" >&2
  exit 1
fi

echo "== stop-after=fusion: mini_model.mlir → StableHLO (stablehlo.convolution), no LLVM =="
stop_out="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.convolution' <<<"${stop_out}"
grep -qv 'llvm.func @inference' <<<"${stop_out}" || {
  echo "error: --pipeline-stop-after=fusion should not reach LLVM" >&2
  exit 1
}

echo "== stop-after=affine: matmul_add.mlir (loop-mode=affine --dump-ir) → affine.for, no LLVM =="
affine_dump="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=affine \
  --pipeline-stop-after=affine --dump-ir 2>&1)"
grep -q 'affine.for' <<<"${affine_dump}"
grep -qv 'llvm.func @inference' <<<"${affine_dump}" || {
  echo "error: --pipeline-stop-after=affine should not reach LLVM" >&2
  exit 1
}

echo "== stop-after=vector: matmul_add.mlir (loop-mode=vector --dump-ir) → vector ops, no LLVM =="
vector_dump="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --loop-mode=vector \
  --pipeline-stop-after=vector --dump-ir 2>&1)"
grep -q 'vector\.' <<<"${vector_dump}"
grep -qv 'llvm.func @inference' <<<"${vector_dump}" || {
  echo "error: --pipeline-stop-after=vector should not reach LLVM" >&2
  exit 1
}

echo "== JIT smoke: matmul_add.mlir (--jit --loop-mode=scf-seq) → JIT result contains 1.5 =="
jit_out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --jit --loop-mode=scf-seq 2>&1)"
grep -q 'JIT result' <<<"${jit_out}"
grep -q '1.5' <<<"${jit_out}"

echo "== compat: matmul_add.mlir (--no-vectorize) → LLVM (alias of loop-mode=scf-seq) =="
legacy_out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --no-vectorize 2>&1)"
grep -q 'llvm.func @inference' <<<"${legacy_out}"

echo "Shell regression passed."
echo "All requested tests passed."
