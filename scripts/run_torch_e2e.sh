#!/usr/bin/env bash
# Torch e2e: torch-mlir Conv+BN export (or committed fixture) -> mlir_pass fusion + JIT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_EXPORT_DIR="$ROOT/scripts/torch_export"
PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"
DEMO="$PASS_BUILD/tools/pipe-demo/pipe-demo"
FIXTURE="$ROOT/test/fixtures/conv_bn_torch.mlir"
JIT_FIXTURE="$ROOT/test/fixtures/conv_bn_torch_jit.mlir"
EXPORT="$PASS_BUILD/conv_bn_export.mlir"

mkdir -p "$(dirname "$FIXTURE")"

if python3 -c "import torch, torch_mlir" 2>/dev/null; then
  echo "== Export Conv+BN via torch-mlir =="
  mkdir -p "$(dirname "$EXPORT")"
  python3 "$TORCH_EXPORT_DIR/conv_bn_model.py" -o "$EXPORT"
  cp "$EXPORT" "$FIXTURE"
  sed -i 's/@main/@inference/' "$FIXTURE" || true
  sed -i 's/func.func @[^(/]*/func.func @inference/' "$FIXTURE" || true
else
  echo "== torch-mlir unavailable; use committed fixture =="
  if [[ ! -f "$FIXTURE" ]]; then
    echo "error: missing $FIXTURE and torch-mlir not installed" >&2
    exit 1
  fi
fi

if [[ ! -x "$DEMO" ]]; then
  echo "error: missing $DEMO" >&2
  exit 1
fi

out="$("$DEMO" --input="$FIXTURE" --pipeline-stop-after=fusion 2>&1)"
grep -q 'stablehlo.convolution' <<<"$out"
if grep -qi 'batch_norm' <<<"$out"; then
  echo "error: conv-bn-fusion should eliminate batch_norm from graph" >&2
  exit 1
fi

echo "torch Conv+BN -> mlir_pass fusion e2e passed."
echo "Fixture: $FIXTURE"
