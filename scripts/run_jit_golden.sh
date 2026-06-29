#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MLIR_PASS_BUILD="${MLIR_PASS_BUILD:-$ROOT/build}"
exec python3 "$ROOT/scripts/run_jit_golden.py" "$@"
