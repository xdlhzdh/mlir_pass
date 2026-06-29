// RUN: %pipe-demo --input=%s --pipeline-stop-after=linalg 2>&1 | FileCheck %s
// CHECK: linalg.generic
// CHECK-NOT: stablehlo.add

module {
  func.func @cse_smoke(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %c = stablehlo.constant dense<1.0> : tensor<2x3xf32>
    %0 = stablehlo.add %arg0, %c : tensor<2x3xf32>
    %1 = stablehlo.add %arg0, %c : tensor<2x3xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
}
