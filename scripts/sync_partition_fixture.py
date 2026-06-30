#!/usr/bin/env python3
"""Parse P13 graph partition stdout and verify comm-byte golden."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# P13 teaching transformer-block partition total comm bytes (fp32).
EXPECTED_COMM_BYTES = 262144


def main() -> int:
    compiler_build = Path(
        __import__("os").environ.get(
            "MLIR_COMPILER_BUILD",
            Path(__file__).resolve().parents[2] / "mlir_compiler/build",
        )
    )
    gpart = compiler_build / "src/mlir/gpu/run_graph_partition"
    if not gpart.is_file():
        print(f"skip: missing {gpart}", file=sys.stderr)
        return 0

    proc = subprocess.run([str(gpart)], capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(out, file=sys.stderr)
        return 1

    if "boundary tensors" not in out:
        print("error: P13 output missing boundary tensors", file=sys.stderr)
        return 1

    m = re.search(r"Total cross-partition comm bytes:\s*(\d+)", out)
    if not m:
        print("error: missing total comm bytes line", file=sys.stderr)
        return 1

    total = int(m.group(1))
    if total != EXPECTED_COMM_BYTES:
        print(
            f"error: comm bytes {total} != expected {EXPECTED_COMM_BYTES}",
            file=sys.stderr,
        )
        return 1

    print(f"P13 partition comm bytes: {total} (teaching graph)")
    print("PASS sync_partition_fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
