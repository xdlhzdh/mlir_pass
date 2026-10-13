// RUN: %pipe-demo --input=%matmul_add --loop-mode=scf-par --pipeline-stop-after=loops --dump-ir 2>&1 | FileCheck %s
// RUN: %pipe-demo --input=%matmul_add --loop-mode=scf-par 2>&1 | FileCheck %s --check-prefix=LLVM

// custom-linalg-to-parallel-loops lowers fill + matmul + elementwise add itself.
// CHECK-LABEL: IR Dump After CustomLinalgToParallelLoops
// CHECK-NOT: linalg.
// CHECK: scf.parallel (%[[I:[a-z0-9_]+]], %[[J:[a-z0-9_]+]]) =
// CHECK-NEXT: memref.store %{{[a-z0-9_]+}}, %[[C:[a-z0-9_]+]][%[[I]], %[[J]]]
// CHECK: scf.parallel (%[[I2:[a-z0-9_]+]], %[[J2:[a-z0-9_]+]]) =
// CHECK-NEXT: %[[INIT:[0-9]+]] = memref.load %[[C]][%[[I2]], %[[J2]]]
// CHECK-NEXT: %[[ACC:[0-9]+]] = scf.for {{.*}} iter_args(%{{[a-z0-9_]+}} = %[[INIT]]) -> (f32)
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK: memref.store %[[ACC]], %[[C]][%[[I2]], %[[J2]]]
// CHECK: scf.parallel
// CHECK: arith.addf
// CHECK-NOT: linalg.
// CHECK-LABEL: IR Dump After ConvertLinalgToParallelLoops

// No linalg op or unresolved cast may survive to the LLVM dialect.
// LLVM-NOT: linalg.
// LLVM-NOT: unrealized_conversion_cast
// LLVM: llvm.func @inference
// LLVM-NOT: linalg.
// LLVM-NOT: unrealized_conversion_cast
