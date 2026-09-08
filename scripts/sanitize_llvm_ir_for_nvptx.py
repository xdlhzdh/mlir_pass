#!/usr/bin/env python3
"""Downgrade LLVM IR from newer mlir-translate so an older NVPTX llc can parse it.

System mlir-opt/mlir-translate may be LLVM 23 while a fallback llc (e.g. apt
llvm-18) only understands older textual IR. Strip syntax that breaks parsing.
"""
from __future__ import annotations

import argparse
import re
import sys


def sanitize(src: str) -> str:
    src = re.sub(r"\brange\([^)]+\) ", "", src)
    src = src.replace("getelementptr inbounds nuw ", "getelementptr inbounds ")
    src = src.replace("getelementptr inbounds|nuw ", "getelementptr inbounds ")
    src = re.sub(r"\b(mul|add|sub) nuw nsw ", r"\1 nsw ", src)
    src = re.sub(r"\b(mul|add|sub) nsw nuw ", r"\1 nsw ", src)
    src = src.replace("define ptx_kernel void", "define void")
    src = re.sub(r"\bnoundef ", "", src)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    text = open(args.input, encoding="utf-8").read()
    open(args.output, "w", encoding="utf-8").write(sanitize(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
