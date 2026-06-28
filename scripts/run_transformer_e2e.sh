#!/usr/bin/env bash
# Cross-repo e2e: mlir_compiler P4 Transformer fixtures -> StableHLO -> mlir_pass fusion.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"

L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
MODEL_DIR="$COMPILER_BUILD/src/mlir/gpu/lowering_models"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE_DIR="$ROOT/test/fixtures"

if [[ ! -x "$L3" ]]; then
  echo "error: missing $L3 (build mlir_compiler run_lowering_l3)" >&2
  exit 1
fi
if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO (build mlir_pass pipe-demo)" >&2
  exit 1
fi

mkdir -p "$FIXTURE_DIR"

run_case() {
  local name="$1"
  local onnx="$2"
  local fixture="$FIXTURE_DIR/${name}_p4.mlir"
  shift 2
  local -a checks=("$@")

  if [[ ! -f "$onnx" ]]; then
    echo "error: missing $onnx (cmake --build $COMPILER_BUILD --target gen_lowering_models)" >&2
    exit 1
  fi

  echo "== Export P4 ${name} ONNX -> StableHLO MLIR =="
  "$L3" --mlir-only "$onnx" > "$fixture"

  echo "== mlir_pass fusion on exported ${name} graph =="
  local out
  out="$("$DEMO" --input="$fixture" --pipeline-stop-after=fusion 2>&1)"
  for pattern in "${checks[@]}"; do
    grep -q "$pattern" <<<"$out"
  done
  echo "${name} P4 -> mlir_pass fusion e2e passed."
  echo "Fixture: $fixture"
}

run_case softmax "$MODEL_DIR/lowering_softmax.onnx" \
  'aicom.softmax_canonicalized' \
  'stablehlo.divide'

run_case attention "$MODEL_DIR/lowering_attention.onnx" \
  'stablehlo.dot_general' \
  'aicom.softmax_canonicalized' \
  'aicom.scaled_dot_product_attention'

run_case rmsnorm "$MODEL_DIR/lowering_rmsnorm.onnx" \
  'aicom.rmsnorm_canonicalized' \
  'stablehlo.sqrt' \
  'stablehlo.reduce'

run_case rope "$MODEL_DIR/lowering_rope.onnx" \
  'stablehlo.slice' \
  'stablehlo.concatenate' \
  'aicom.rope_canonicalized'

run_case layernorm "$MODEL_DIR/lowering_layernorm.onnx" \
  'stablehlo.subtract' \
  'aicom.layernorm_canonicalized'

echo "Transformer (softmax/attention/rmsnorm/rope/layernorm) cross-repo e2e passed."
