#!/usr/bin/env bash
# Cross-repo dynamic shape: batch + M/N dynamic MatMul.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"
L2="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l2"
L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
MODEL_DIR="$COMPILER_BUILD/src/mlir/gpu/lowering_models"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE_DIR="$ROOT/test/fixtures"

if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO" >&2
  exit 1
fi

run_export() {
  local onnx="$1" runner="$2" name="$3"
  if [[ ! -f "$onnx" ]]; then
    echo "error: missing $onnx" >&2
    exit 1
  fi
  if [[ ! -x "$runner" ]]; then
    echo "skip: $name (missing $runner)" >&2
    return 0
  fi
  local fixture="$FIXTURE_DIR/${name}_p4.mlir"
  mkdir -p "$FIXTURE_DIR"
  "$runner" --mlir-only "$onnx" > "$fixture"
  sed -i 's/@main/@inference/' "$fixture"
  fusion_out="$("$DEMO" --input="$fixture" --pipeline-stop-after=fusion 2>&1)"
  grep -q 'tensor<?x' <<<"$fusion_out"
  grep -q 'stablehlo.dot_general' <<<"$fusion_out"
  linalg_out="$("$DEMO" --input="$fixture" --pipeline-stop-after=linalg 2>&1)"
  grep -q 'linalg.matmul' <<<"$linalg_out"
  echo "$name P4 -> mlir_pass e2e passed ($fixture)"
}

run_export "$MODEL_DIR/lowering_dynamic.onnx" "$L2" "dynamic_batch"
run_export "$MODEL_DIR/lowering_dynamic_mn.onnx" "$L3" "dynamic_mn"

echo "== decode_loop LIT (scf.while teaching) =="
loop_out="$("$DEMO" --input="${ROOT}/test/lit/decode_loop.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'scf.while' <<<"$loop_out"

echo "Dynamic shape e2e passed."
