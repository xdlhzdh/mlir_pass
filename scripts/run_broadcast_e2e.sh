#!/usr/bin/env bash
# Cross-repo e2e: P4 numpy-style broadcast ONNX -> StableHLO -> mlir_pass fusion/linalg.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"

L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
MODEL="$COMPILER_BUILD/src/mlir/gpu/lowering_models/lowering_broadcast.onnx"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE="$ROOT/test/fixtures/broadcast_p4.mlir"

if [[ ! -x "$L3" ]]; then
  echo "error: missing $L3" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: missing $MODEL (cmake --build $COMPILER_BUILD --target gen_lowering_models)" >&2
  exit 1
fi
if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO" >&2
  exit 1
fi

mkdir -p "$(dirname "$FIXTURE")"

echo "== Export P4 lowering_broadcast.onnx -> StableHLO MLIR =="
"$L3" --mlir-only "$MODEL" > "$FIXTURE"
sed -i 's/@main/@inference/' "$FIXTURE"

echo "== mlir_pass fusion: expect broadcast_in_dim from P4 tier 2 =="
fusion_out="$("$DEMO" --input="$FIXTURE" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.broadcast_in_dim' <<<"$fusion_out"

echo "== mlir_pass linalg: broadcast graph lowers without error =="
linalg_out="$("$DEMO" --input="$FIXTURE" --pipeline-stop-after=linalg 2>&1)"
grep -q 'linalg\.' <<<"$linalg_out"

echo "broadcast P4 -> mlir_pass fusion/linalg e2e passed."
echo "Fixture: $FIXTURE"
