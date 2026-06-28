#!/usr/bin/env bash
# Cross-repo e2e: mlir_compiler P4 attention ONNX -> StableHLO -> mlir_pass fusion.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"

L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
ONNX="$COMPILER_BUILD/src/mlir/gpu/lowering_models/lowering_attention.onnx"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE="$ROOT/test/fixtures/attention_p4.mlir"

if [[ ! -x "$L3" ]]; then
  echo "error: missing $L3 (build mlir_compiler run_lowering_l3)" >&2
  exit 1
fi
if [[ ! -f "$ONNX" ]]; then
  echo "error: missing $ONNX (cmake --build $COMPILER_BUILD --target gen_lowering_models)" >&2
  exit 1
fi
if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO (build mlir_pass pipe-demo)" >&2
  exit 1
fi

mkdir -p "$(dirname "$FIXTURE")"
echo "== Export P4 attention ONNX -> StableHLO MLIR =="
"$L3" --mlir-only "$ONNX" > "$FIXTURE"

echo "== mlir_pass fusion on exported attention graph =="
out="$("$DEMO" --input="$FIXTURE" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.dot_general' <<<"$out"
grep -q 'aicom.softmax_canonicalized' <<<"$out"
grep -q 'aicom.scaled_dot_product_attention' <<<"$out"

echo "Attention P4 -> mlir_pass fusion e2e passed."
echo "Fixture: $FIXTURE"
