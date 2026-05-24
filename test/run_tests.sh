#!/usr/bin/env bash
# Regression tests for ai-compiler-demo.
#
# Usage:
#   bash test/run_tests.sh           # shell regression (default)
#   bash test/run_tests.sh --lit     # LIT/FileCheck only
#   bash test/run_tests.sh --all     # shell regression + LIT/FileCheck
#   bash test/run_tests.sh --help
#
# Ninja equivalents (from repo root, after cmake -B build):
#   ninja -C build test-ai-compiler       # shell only
#   ninja -C build check-ai-compiler      # LIT only (= --lit)
#   ninja -C build test-ai-compiler-all   # shell + LIT (= --all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
DEMO="${BUILD}/tools/ai-compiler-demo/ai-compiler-demo"
MODE=shell

usage() {
  cat <<'EOF'
Usage: test/run_tests.sh [OPTIONS]

Options:
  (none)    Shell regression: full pipeline, fusion, stop-after, JIT.
  --lit     LIT/FileCheck only (ninja check-ai-compiler).
  --all     Shell regression, then LIT/FileCheck.
  --help    Show this help.

Ninja targets (equivalent commands):
  test-ai-compiler       Shell only (default)
  check-ai-compiler      LIT only (--lit)
  test-ai-compiler-all   Shell + LIT (--all)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lit) MODE=lit; shift ;;
    --all) MODE=all; shift ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${DEMO}" ]]; then
  echo "error: build ai-compiler-demo first: ninja -C build" >&2
  exit 1
fi

run_shell_regression() {
  echo "== matmul_add: full pipeline =="
  out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --no-vectorize 2>&1)"
  echo "${out}" | grep -q 'llvm.func @inference'

  echo "== conv_bn_relu: full pipeline =="
  "${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --no-vectorize >/dev/null

  echo "== mini_model: full pipeline =="
  out_mini="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --no-vectorize 2>&1)"
  echo "${out_mini}" | grep -q 'llvm.func @inference'

  echo "== conv_bn_relu: fusion removes batch_norm =="
  dump="$("${DEMO}" --input="${ROOT}/test/conv_bn_relu.mlir" --dump-ir 2>&1)"
  fusion_block="$(echo "${dump}" | sed -n '/IR Dump After Stage: fusion/,/IR Dump After Stage:/p')"
  if echo "${fusion_block}" | grep -q batch_norm_inference; then
    echo "error: batch_norm_inference still present after fusion stage" >&2
    exit 1
  fi

  echo "== stop-after fusion =="
  stop_out="$("${DEMO}" --input="${ROOT}/test/mini_model.mlir" --pipeline-stop-after=fusion 2>&1)"
  echo "${stop_out}" | grep -q 'stablehlo.convolution'
  echo "${stop_out}" | grep -qv 'llvm.func @inference' || {
    echo "error: --pipeline-stop-after=fusion should not reach LLVM" >&2
    exit 1
  }

  echo "== matmul_add: JIT =="
  jit_out="$("${DEMO}" --input="${ROOT}/test/matmul_add.mlir" --jit --no-vectorize 2>&1)"
  echo "${jit_out}" | grep -q 'JIT result'
  echo "${jit_out}" | grep -q '1.5'
}

run_lit() {
  if [[ ! -d "${BUILD}" ]]; then
    echo "error: build/ missing; run cmake -B build && ninja -C build" >&2
    exit 1
  fi
  if ! ninja -C "${BUILD}" -t targets all 2>/dev/null | grep -q '^check-ai-compiler:'; then
    echo "error: check-ai-compiler not configured." >&2
    echo "  Install lit + FileCheck, re-run cmake, then retry." >&2
    echo "  Example: sudo apt install llvm-21-tools && export PATH=/usr/lib/llvm-21/bin:\$PATH" >&2
    exit 1
  fi
  echo "== LIT / FileCheck =="
  ninja -C "${BUILD}" check-ai-compiler
}

case "${MODE}" in
  shell)
    run_shell_regression
    echo "Shell regression passed."
    ;;
  lit)
    run_lit
    echo "LIT tests passed."
    ;;
  all)
    run_shell_regression
    echo "Shell regression passed."
    run_lit
    echo "LIT tests passed."
    ;;
esac

echo "All requested tests passed."
