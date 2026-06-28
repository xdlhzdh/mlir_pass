// RUN: %pipe-demo --input=%conv_bn_relu --pipeline-stop-after=linalg 2>&1 | FileCheck %s
// CHECK: linalg.conv_2d
// CHECK: linalg.generic
// CHECK-NOT: stablehlo.maximum
