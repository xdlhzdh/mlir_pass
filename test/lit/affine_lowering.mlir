// RUN: %pipe-demo --input=%matmul_add --loop-mode=affine --pipeline-stop-after=affine 2>&1 | FileCheck %s
// CHECK: affine.for
// CHECK-NOT: llvm.func
