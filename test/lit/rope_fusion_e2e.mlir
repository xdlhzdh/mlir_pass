// RUN: %pipe-demo --input=%rope_fixture --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: aicom.rope_canonicalized
