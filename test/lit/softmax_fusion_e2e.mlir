// RUN: %pipe-demo --input=%softmax_fixture --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.divide
// CHECK: aicom.softmax_canonicalized
