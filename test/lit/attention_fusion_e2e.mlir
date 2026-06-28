// RUN: %pipe-demo --input=%attention_fixture --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.dot_general
// CHECK: aicom.softmax_canonicalized
