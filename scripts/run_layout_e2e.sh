#!/usr/bin/env bash
# Cross-repo e2e: P4 layout conv ONNX -> StableHLO -> mlir_pass layout-bridge-legalize.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"

L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
MODEL="$COMPILER_BUILD/src/mlir/gpu/lowering_models/lowering_layout_conv.onnx"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE="$ROOT/test/fixtures/layout_conv_p4.mlir"

if [[ ! -x "$L3" ]]; then
  echo "error: missing $L3" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: missing $MODEL (gen_lowering_models)" >&2
  exit 1
fi

mkdir -p "$(dirname "$FIXTURE")"
"$L3" --mlir-only "$MODEL" > "$FIXTURE"
sed -i 's/@main/@inference/' "$FIXTURE"

out="$("$DEMO" --input="$FIXTURE" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.convolution' <<<"$out"
grep -q 'aicom.layout_folded' <<<"$out"
grep -qv 'stablehlo.transpose' <<<"$out"
grep -q 'dim_numbers = \[b, 0, 1, f\]x\[0, 1, i, o\]->\[b, 0, 1, f\]' <<<"$out" || \
  grep -q 'dim_numbers = \[b, f, 0, 1\]x\[o, i, 0, 1\]->\[b, 0, 1, f\]' <<<"$out" || \
  grep -q 'aicom.layout_folded' <<<"$out"

echo "Layout conv P4 -> mlir_pass fusion e2e passed."
echo "Fixture: $FIXTURE"
