// RUN: %pipe-demo --input=%matmul_add --loop-mode=vector --pipeline-stop-after=vector 2>&1 | FileCheck %s
// CHECK: vector.
// CHECK-NOT: llvm.func
