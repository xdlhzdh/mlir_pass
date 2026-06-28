// RUN: %pipe-demo --input=%layernorm_fixture --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.subtract
// CHECK: aicom.layernorm_canonicalized
