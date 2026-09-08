#!/usr/bin/env bash
# GPU matmul codegen path:
#   Transform Dialect (tile / vectorize / tensorize)
#     → gpu / nvgpu
#     → nvvm / llvm dialect
#     → mlir-translate → LLVM IR
#     → llc -march=nvptx64 → PTX
#     → ptxas → cubin (CUDA ELF)
#
# Modes:
#   mma-sync  (default)  — nvgpu.rewrite_matmul_as_mma_sync on a warp-sized GEMM
#   tiling               — Buddy-style CTA/K/warp tiling → gpu.launch (+ optional MMA)
#
# Env overrides:
#   MLIR_OPT, MLIR_TRANSLATE, LLC, PTXAS, CUDA_SM (default sm_80)
#   MLIR_PASS_LLC_LIB_PATH  — LD_LIBRARY_PATH prefix for a fallback llc
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU_DIR="${ROOT}/gpu_codegen"
OUT_DIR="${OUT_DIR:-${ROOT}/output/gpu_codegen}"
MODE="${1:-mma-sync}"
SM="${CUDA_SM:-sm_80}"

MLIR_OPT="${MLIR_OPT:-$(command -v mlir-opt)}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-$(command -v mlir-translate)}"
PTXAS="${PTXAS:-$(command -v ptxas || true)}"
PYTHON="${PYTHON:-python3}"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "missing tool: $1"; }

need mlir-opt
need mlir-translate
[[ -n "${PTXAS}" ]] || die "missing ptxas (CUDA toolkit)"

mkdir -p "${OUT_DIR}"
STAGE="${OUT_DIR}/${MODE}"
mkdir -p "${STAGE}"

# ---------------------------------------------------------------------------
# Resolve an llc that can target nvptx64.
# Prefer MLIR_PASS_LLC / LLC; else system llc; else optional llvm-18 extract.
# Important: do NOT assign via $(find_...); LD_LIBRARY_PATH exports would be lost.
# ---------------------------------------------------------------------------
find_nvptx_llc() {
  local candidates=()
  [[ -n "${MLIR_PASS_LLC:-}" ]] && candidates+=("${MLIR_PASS_LLC}")
  [[ -n "${LLC:-}" ]] && candidates+=("${LLC}")
  candidates+=("$(command -v llc || true)")
  # Optional local extract of apt llvm-18 (see gpu_codegen/README.md).
  candidates+=(
    "${ROOT}/third_party/llvm-18/bin/llc"
    "/tmp/llvm_nvptx/extracted/usr/lib/llvm-18/bin/llc"
  )

  local c libdir env_ld
  for c in "${candidates[@]}"; do
    [[ -n "$c" && -x "$c" ]] || continue
    libdir="$(cd "$(dirname "$c")/.." && pwd)/lib"
    env_ld="${LD_LIBRARY_PATH:-}"
    [[ -n "${MLIR_PASS_LLC_LIB_PATH:-}" ]] && env_ld="${MLIR_PASS_LLC_LIB_PATH}:${env_ld}"
    [[ -d "${libdir}" ]] && env_ld="${libdir}:${env_ld}"
    [[ -d /tmp/llvm_nvptx/extracted/usr/lib/x86_64-linux-gnu ]] && \
      env_ld="/tmp/llvm_nvptx/extracted/usr/lib/x86_64-linux-gnu:${env_ld}"
    # Registered targets appear under `llc -version`, not always `--version`.
    if LD_LIBRARY_PATH="${env_ld}" "$c" -version 2>/dev/null | grep -q 'nvptx64'; then
      LLC_BIN="$c"
      export LD_LIBRARY_PATH="${env_ld}"
      return 0
    fi
  done
  return 1
}

LLC_BIN=""
find_nvptx_llc || true
[[ -n "${LLC_BIN}" ]] || die "no llc with NVPTX backend.
Rebuild MLIR/LLVM with -DLLVM_TARGETS_TO_BUILD='X86;NVPTX' (or X86;NVPTX;RISCV),
or set MLIR_PASS_LLC to an llc that registers nvptx64.
Hint: apt-get download llvm-18 libllvm18 && dpkg-deb -x ... into third_party/llvm-18"

echo "==> mode=${MODE}  sm=${SM}"
echo "    mlir-opt=${MLIR_OPT}"
echo "    llc=${LLC_BIN}"
echo "    out=${STAGE}"

# ---------------------------------------------------------------------------
run_mma_sync() {
  local src="${GPU_DIR}/matmul_mma_sync.mlir"
  local after_xf="${STAGE}/01_after_transform.mlir"
  local after_llvm="${STAGE}/02_llvm_dialect.mlir"
  local kernel_mlir="${STAGE}/03_kernel_only.mlir"
  local kernel_ll="${STAGE}/04_kernel.ll"
  local kernel_ll_s="${STAGE}/04_kernel.sanitized.ll"
  local kernel_ptx="${STAGE}/05_kernel.ptx"
  local kernel_elf="${STAGE}/06_kernel.cubin"

  echo "==> [1/5] transform: linalg.matmul → nvgpu.mma.sync"
  "${MLIR_OPT}" "${src}" \
    --pass-pipeline='builtin.module(transform-interpreter{disable-expensive-checks=true})' \
    -o "${after_xf}"
  grep -q 'nvgpu.mma.sync' "${after_xf}" \
    || die "expected nvgpu.mma.sync after transform"

  echo "==> [2/5] erase schedule + lower gpu/nvgpu → llvm/nvvm dialect"
  "${MLIR_OPT}" "${after_xf}" \
    --pass-pipeline='builtin.module(
      test-transform-dialect-erase-schedule,
      gpu-kernel-outlining,
      lower-affine,
      convert-scf-to-cf,
      convert-nvgpu-to-nvvm,
      convert-vector-to-llvm,
      convert-arith-to-llvm,
      convert-index-to-llvm,
      gpu.module(convert-gpu-to-nvvm{index-bitwidth=64},convert-nvvm-to-llvm,canonicalize,cse),
      convert-cf-to-llvm,
      convert-func-to-llvm,
      reconcile-unrealized-casts
    )' -o "${after_llvm}"

  echo "==> [3/5] extract gpu.module kernel + mlir-translate → LLVM IR"
  "${PYTHON}" - "${after_llvm}" "${kernel_mlir}" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"gpu\.module @\w+ \{([\s\S]*?)\n  \}", text)
if not m:
    sys.exit("gpu.module not found in lowered IR")
body = m.group(1)
lines = [ln[2:] if ln.startswith("  ") else ln for ln in body.splitlines()]
open(sys.argv[2], "w", encoding="utf-8").write(
    "module {\n" + "\n".join(lines) + "\n}\n"
)
PY
  "${MLIR_TRANSLATE}" --mlir-to-llvmir "${kernel_mlir}" -o "${kernel_ll}"
  grep -E -q 'llvm\.nvvm\.mma|@llvm\.nvvm\.mma' "${kernel_ll}" \
    || die "expected llvm.nvvm.mma intrinsic in LLVM IR"

  echo "==> [4/5] llc -march=nvptx64 → PTX"
  "${PYTHON}" "${ROOT}/scripts/sanitize_llvm_ir_for_nvptx.py" \
    "${kernel_ll}" -o "${kernel_ll_s}"
  "${LLC_BIN}" -march=nvptx64 -mcpu="${SM}" "${kernel_ll_s}" -o "${kernel_ptx}"
  grep -q 'mma.sync' "${kernel_ptx}" \
    || die "expected mma.sync in PTX"

  echo "==> [5/5] ptxas → cubin ELF"
  "${PTXAS}" -arch="${SM}" -o "${kernel_elf}" "${kernel_ptx}"
  file "${kernel_elf}" | grep -qi 'ELF' \
    || die "ptxas did not produce an ELF cubin"

  echo
  echo "OK: ${kernel_elf}"
  file "${kernel_elf}"
  readelf -h "${kernel_elf}" 2>/dev/null | head -12 || true
}

run_tiling() {
  local src="${GPU_DIR}/matmul_tiling.mlir"
  local after_xf="${STAGE}/01_after_transform.mlir"

  echo "==> [1/1] transform: tile / vectorize / map → gpu.launch"
  "${MLIR_OPT}" "${src}" \
    --pass-pipeline='builtin.module(transform-interpreter{disable-expensive-checks=true},test-transform-dialect-erase-schedule)' \
    -o "${after_xf}"
  grep -q 'gpu.launch' "${after_xf}" \
    || die "expected gpu.launch after tiling transform"
  grep -qE 'vector\.contract|nvgpu\.mma' "${after_xf}" \
    || die "expected vector.contract or nvgpu.mma after tiling transform"

  echo
  echo "OK: tiling/vectorize/gpu map IR → ${after_xf}"
  echo "    (Inspect gpu.launch / vector.contract / nvgpu.* )"
  echo "    For Tensor Core ELF, run: $0 mma-sync"
}

case "${MODE}" in
  mma-sync|mma) run_mma_sync ;;
  tiling|tile)  run_tiling ;;
  *) die "unknown mode '${MODE}' (use mma-sync|tiling)" ;;
esac
