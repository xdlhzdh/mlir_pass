#!/usr/bin/env python3
"""Compare pipe-demo --jit output against NumPy references (rtol/atol=1e-5)."""
from __future__ import annotations

import re
import subprocess
import sys
from math import erf
from pathlib import Path

import numpy as np

RTOL = 1e-5
ATOL = 1e-5
RTOL_GELU_P4 = 1e-3
ATOL_GELU_P4 = 1e-3
ROOT = Path(__file__).resolve().parents[1]
PASS_BUILD = Path(__import__("os").environ.get("MLIR_PASS_BUILD", ROOT / "build"))
DEMO = PASS_BUILD / "tools/pipe-demo/pipe-demo"

JIT_RE = re.compile(
    r"JIT result \((\d+) elements\):\s*([-\d.e+, ]+)", re.MULTILINE
)


def run_jit(mlir_path: Path, loop_mode: str = "scf-seq") -> np.ndarray:
    if not DEMO.is_file():
        raise FileNotFoundError(f"missing {DEMO}")
    proc = subprocess.run(
        [str(DEMO), f"--input={mlir_path}", "--jit", f"--loop-mode={loop_mode}"],
        capture_output=True,
        text=True,
        check=False,
        # A crash inside JIT code can deadlock LLVM's signal handler.
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"pipe-demo failed on {mlir_path}:\n{out}")
    m = JIT_RE.search(out)
    if not m:
        raise RuntimeError(f"no JIT result in output for {mlir_path}:\n{out}")
    values = [float(x.strip()) for x in m.group(2).split(",") if x.strip()]
    expected_n = int(m.group(1))
    if len(values) != expected_n:
        raise RuntimeError(f"parsed {len(values)} values, expected {expected_n}")
    return np.array(values, dtype=np.float32)


def ref_matmul_add() -> np.ndarray:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.eye(2, dtype=np.float32)
    bias = np.full((2, 2), 0.5, dtype=np.float32)
    return (a @ b + bias).reshape(-1)


def ref_gelu() -> np.ndarray:
    x = np.array(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=np.float32
    )
    k = np.float32(1.702)
    sig = 1.0 / (1.0 + np.exp(-k * x))
    return (x * sig).reshape(-1)


def ref_swiglu() -> np.ndarray:
    gate = np.array(
        [[1.0, 0.0, -1.0, 2.0], [0.5, 1.0, 1.5, 2.0]], dtype=np.float32
    )
    up = np.array(
        [[2.0, 1.0, 0.5, 0.25], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32
    )
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up).reshape(-1)


def ref_gelu_p4() -> np.ndarray:
    x = np.array(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=np.float32
    )
    return (0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))).reshape(-1)


def _conv_bn_relu_nhwc() -> np.ndarray:
    # NHWC input 1x2x2x1, HWIO 1x1 kernel with 2 output features.
    inp = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
    w = np.full((1, 1, 1, 2), 0.5, dtype=np.float32)
    conv = np.einsum("nhwc,co->nhwo", inp, w[0, 0])
    scale, offset, mean, var, eps = 1.0, 0.0, 0.0, 1.0, 1e-5
    bn = (conv - mean) * scale / np.sqrt(var + eps) + offset
    return np.maximum(bn, 0.0).astype(np.float32)


def ref_conv_bn_relu() -> np.ndarray:
    return _conv_bn_relu_nhwc().reshape(-1)


def ref_mini_model() -> np.ndarray:
    flat = _conv_bn_relu_nhwc().reshape(1, 8)
    fc_w = np.zeros((8, 2), dtype=np.float32)
    fc_w[0, 0] = fc_w[1, 1] = 1.0
    return (flat @ fc_w + 0.1).reshape(-1)


CASES = [
    ("matmul_add", ROOT / "test/matmul_add.mlir", ref_matmul_add, RTOL, ATOL),
    (
        "jit_scale",
        ROOT / "test/jit_scale.mlir",
        lambda: (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32) * 2.0
        ).reshape(-1),
        RTOL,
        ATOL,
    ),
    ("jit_gelu", ROOT / "test/jit_gelu.mlir", ref_gelu, RTOL, ATOL),
    ("jit_gelu_p4", ROOT / "test/fixtures/gelu_p4_jit.mlir", ref_gelu_p4,
     RTOL_GELU_P4, ATOL_GELU_P4),
    ("jit_swiglu", ROOT / "test/jit_swiglu.mlir", ref_swiglu, RTOL, ATOL),
    ("jit_swiglu_p4", ROOT / "test/fixtures/swiglu_p4_jit.mlir", ref_swiglu, RTOL, ATOL),
    ("conv_bn_relu", ROOT / "test/conv_bn_relu.mlir", ref_conv_bn_relu, RTOL, ATOL),
    ("mini_model", ROOT / "test/mini_model.mlir", ref_mini_model, RTOL, ATOL),
]


def main() -> int:
    failed = 0
    for name, mlir_path, ref_fn, rtol, atol in CASES:
        if not mlir_path.is_file():
            print(f"  SKIP {name}: missing {mlir_path}", file=sys.stderr)
            failed += 1
            continue
        try:
            got = run_jit(mlir_path)
            expected = ref_fn()
            if got.shape != expected.shape or not np.allclose(
                got, expected, rtol=rtol, atol=atol
            ):
                print(f"  FAIL {name}: max diff {np.max(np.abs(got - expected))}")
                print(f"    got:      {got}")
                print(f"    expected: {expected}")
                failed += 1
            else:
                print(f"  PASS {name} ({got.size} elements)")
        except Exception as exc:  # noqa: BLE001 — teaching script
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
            failed += 1
    if failed:
        print(f"\n{failed} JIT golden check(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(CASES)} JIT golden checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
