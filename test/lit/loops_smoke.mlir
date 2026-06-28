// RUN: %pipe-demo --input=%matmul_add --loop-mode=scf-seq --pipeline-stop-after=loops 2>&1 | FileCheck %s
// CHECK: memref.store
// CHECK: cf.br
// CHECK-NOT: llvm.func
