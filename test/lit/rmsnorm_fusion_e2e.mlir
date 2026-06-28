// RUN: %pipe-demo --input=%rmsnorm_fixture --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.sqrt
// CHECK: aicom.rmsnorm_canonicalized
