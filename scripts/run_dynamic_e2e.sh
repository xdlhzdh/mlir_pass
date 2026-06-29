#!/usr/bin/env bash
# Cross-repo dynamic batch smoke: P4 lowering_dynamic.onnx -> mlir_pass fusion/linalg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"
L2="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l2"
MODEL_DIR="$COMPILER_BUILD/src/mlir/gpu/lowering_models"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE_DIR="$ROOT/test/fixtures"

if [[ ! -x "$L2" ]]; then
  echo "error: missing $L2 (build mlir_compiler run_lowering_l2)" >&2
  exit 1
fi
if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO (build mlir_pass pipe-demo)" >&2
  exit 1
fi

onnx="$MODEL_DIR/lowering_dynamic.onnx"
if [[ ! -f "$onnx" ]]; then
  echo "error: missing $onnx (cmake --build $COMPILER_BUILD --target gen_lowering_models)" >&2
  exit 1
fi

mkdir -p "$FIXTURE_DIR"
fixture="$FIXTURE_DIR/dynamic_batch_p4.mlir"

echo "== Export P4 dynamic ONNX -> StableHLO MLIR =="
"$L2" --mlir-only "$onnx" > "$fixture"
sed -i 's/@main/@inference/' "$fixture"

echo "== mlir_pass fusion on exported dynamic graph =="
fusion_out="$("$DEMO" --input="$fixture" --pipeline-stop-after=fusion 2>&1)"
grep -q 'tensor<?x' <<<"$fusion_out"
grep -q 'stablehlo.dot_general' <<<"$fusion_out"

echo "== mlir_pass linalg on exported dynamic graph =="
linalg_out="$("$DEMO" --input="$fixture" --pipeline-stop-after=linalg 2>&1)"
grep -q 'linalg.matmul' <<<"$linalg_out"

echo "Dynamic batch P4 -> mlir_pass fusion/linalg e2e passed."
echo "Fixture: $fixture"
