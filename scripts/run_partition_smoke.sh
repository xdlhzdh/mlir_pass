#!/usr/bin/env bash
# P13 graph partition demo + mlir_pass partition-boundary fusion smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_BUILD="${MLIR_COMPILER_BUILD:-$ROOT/../mlir_compiler/build}"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"

GPART="$COMPILER_BUILD/src/mlir/gpu/run_graph_partition"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
SMOKE="$ROOT/test/lit/graph_partition_smoke.mlir"

if [[ ! -x "$GPART" ]]; then
  echo "error: missing $GPART (build mlir_compiler run_graph_partition)" >&2
  exit 1
fi
if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO (build mlir_pass pipe-demo)" >&2
  exit 1
fi

echo "== P13 run_graph_partition_demo =="
"$GPART" | grep -q 'boundary tensors'

echo "== mlir_pass fusion on graph_partition_smoke.mlir =="
out="$("$DEMO" --input="$SMOKE" --pipeline-stop-after=fusion 2>&1)"
grep -q 'aicom.partition_boundary' <<<"$out"
grep -q 'stablehlo.dot_general' <<<"$out"

echo "Graph partition smoke (P13 + mlir_pass fixture) passed."
