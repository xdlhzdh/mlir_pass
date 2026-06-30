#!/usr/bin/env bash
# Cross-repo quant smoke: P11 Q/DQ teaching fixture + P4 export -> mlir_pass qdq-legalize.
# PTQ 完整链（校准 demo -> quant Stage 2）见 mlir_compiler run_calib_demo + run_calib_to_quant.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE="${ROOT}/test/lit/qdq_legalize.mlir"
L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
MODEL_DIR="$COMPILER_BUILD/src/mlir/gpu/lowering_models"
FIXTURE_DIR="$ROOT/test/fixtures"

if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO (build mlir_pass pipe-demo)" >&2
  exit 1
fi

echo "== mlir_pass fusion on Q/DQ MatMul teaching fixture =="
out="$("$DEMO" --input="$FIXTURE" --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.qdq_matmul_canonicalized' <<<"$out"
grep -q 'stablehlo.dot_general' <<<"$out"
echo "Q/DQ MatMul hand-written LIT -> mlir_pass fusion e2e passed."
echo "Fixture: $FIXTURE"

if [[ ! -x "$L3" ]]; then
  echo "skip: P4 export case (missing $L3)" >&2
  exit 0
fi

mkdir -p "$FIXTURE_DIR"
onnx="$MODEL_DIR/lowering_qdq_matmul.onnx"
if [[ ! -f "$onnx" ]]; then
  echo "error: missing $onnx (cmake --build $COMPILER_BUILD --target gen_lowering_models)" >&2
  exit 1
fi

p4_fixture="$FIXTURE_DIR/qdq_matmul_p4.mlir"
echo "== Export P4 qdq_matmul ONNX -> StableHLO MLIR =="
"$L3" --mlir-only "$onnx" > "$p4_fixture"

echo "== mlir_pass fusion on exported qdq_matmul graph =="
p4_out="$("$DEMO" --input="$p4_fixture" --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.qdq_matmul_canonicalized' <<<"$p4_out"
grep -q 'stablehlo.dot_general' <<<"$p4_out"
echo "Q/DQ MatMul P4 -> mlir_pass fusion e2e passed."
echo "Fixture: $p4_fixture"

QUANT_ONNX="$COMPILER_BUILD/src/mlir/gpu/quant_models/quant_qdq_matmul.onnx"
if [[ -f "$QUANT_ONNX" ]]; then
  p11_fixture="$FIXTURE_DIR/qdq_matmul_p11.mlir"
  echo "== Export P11 quant_qdq_matmul ONNX -> StableHLO MLIR =="
  "$L3" --mlir-only "$QUANT_ONNX" > "$p11_fixture"

  echo "== mlir_pass fusion on exported P11 Q/DQ graph =="
  p11_out="$("$DEMO" --input="$p11_fixture" --pipeline-stop-after=fusion 2>&1)"
  grep -q 'aicom.qdq_matmul_canonicalized' <<<"$p11_out"
  grep -q 'stablehlo.dot_general' <<<"$p11_out"
  echo "Q/DQ MatMul P11 -> P4 -> mlir_pass fusion e2e passed."
  echo "Fixture: $p11_fixture"
fi
