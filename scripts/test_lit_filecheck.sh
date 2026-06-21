#!/usr/bin/env bash
# Run LIT/FileCheck tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
LIT_DIR="${BUILD}/test/lit"
CACHE="${BUILD}/CMakeCache.txt"

if [[ ! -d "${BUILD}" ]]; then
  echo "error: build/ missing; run cmake -B build -G Ninja ..." >&2
  exit 1
fi

if [[ ! -f "${LIT_DIR}/lit.site.cfg.py" ]]; then
  echo "error: ${LIT_DIR}/lit.site.cfg.py missing." >&2
  echo "  Re-run cmake after installing lit + FileCheck." >&2
  exit 1
fi

cmake_cache_var() {
  local key="$1"
  [[ -f "${CACHE}" ]] || return 0
  sed -n "s|^${key}:[^=]*=||p" "${CACHE}" | sed -n '1p'
}

is_usable_tool() {
  local tool="$1"
  [[ -n "${tool}" && "${tool}" != *-NOTFOUND && -x "${tool}" ]]
}

lit_exe="${LIT_EXE:-}"
if [[ -z "${lit_exe}" ]]; then
  lit_exe="$(cmake_cache_var LIT_EXE)"
fi
if ! is_usable_tool "${lit_exe}"; then
  lit_exe="$(command -v lit || command -v llvm-lit || true)"
fi

filecheck_exe="${FILECHECK_EXE:-}"
if [[ -z "${filecheck_exe}" ]]; then
  filecheck_exe="$(cmake_cache_var FILECHECK_EXE)"
fi
if ! is_usable_tool "${filecheck_exe}"; then
  filecheck_exe="$(command -v FileCheck || true)"
fi

if ! is_usable_tool "${lit_exe}" || ! is_usable_tool "${filecheck_exe}"; then
  echo "error: lit or FileCheck not found." >&2
  echo "  Install lit + FileCheck, re-run cmake, then retry." >&2
  echo "  Example: sudo apt install llvm-22-tools && export PATH=/usr/lib/llvm-22/bin:\$PATH" >&2
  exit 1
fi

filecheck_dir="$(dirname "${filecheck_exe}")"
PATH="${filecheck_dir}:${PATH}" "${lit_exe}" -sv "${LIT_DIR}"
