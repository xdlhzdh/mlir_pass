#!/usr/bin/env bash
# Cross-repo KV decode: P4 lowering_decode_step -> mlir_pass fusion + P12 memplan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"
L3="$COMPILER_BUILD/src/mlir/gpu/run_lowering_l3"
MEMPLAN="$COMPILER_BUILD/src/mlir/gpu/run_memory_planning"
MODEL_DIR="$COMPILER_BUILD/src/mlir/gpu/lowering_models"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE_DIR="$ROOT/test/fixtures"

if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO" >&2
  exit 1
fi

echo "== mlir_pass kvcache_legalize LIT =="
kv_out="$("$DEMO" --input="${ROOT}/test/lit/kvcache_legalize.mlir" \
  --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.kvcache_boundary' <<<"$kv_out"

if [[ ! -x "$L3" ]]; then
  echo "skip: P4 decode_step export (missing $L3)" >&2
  exit 0
fi

onnx="$MODEL_DIR/lowering_decode_step.onnx"
if [[ ! -f "$onnx" ]]; then
  echo "error: missing $onnx" >&2
  exit 1
fi

mkdir -p "$FIXTURE_DIR"
fixture="$FIXTURE_DIR/decode_step_p4.mlir"
"$L3" --mlir-only "$onnx" > "$fixture"
sed -i 's/@main/@inference/' "$fixture"
if ! grep -q 'aicom.kv_role' "$fixture"; then
  sed -i 's/\(func.func @inference(.*\) -> \(tensor[^ ]*\) {/\1 -> \2 attributes {aicom.kv_role = "K"} {/' \
    "$fixture"
fi

out="$("$DEMO" --input="$fixture" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.dot_general' <<<"$out"
grep -q 'aicom.kvcache_boundary' <<<"$out"

if [[ -x "$MEMPLAN" ]]; then
  echo "== P12 memory planning decode KV slots =="
  "$MEMPLAN" 2>&1 | grep -q 'decode step'
fi

echo "KV decode P4 -> mlir_pass + P12 e2e passed."
echo "Fixture: $fixture"
