// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK-NOT: stablehlo.add
// CHECK-NOT: stablehlo.multiply
// CHECK: stablehlo.constant

module {
  func.func @inference() -> tensor<2x2xf32> {
    %c1 = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %c2 = stablehlo.constant dense<[[0.5, 1.5], [2.5, 3.5]]> : tensor<2x2xf32>
    %sum = stablehlo.add %c1, %c2 : tensor<2x2xf32>
    %prod = stablehlo.multiply %sum, %c1 : tensor<2x2xf32>
    return %prod : tensor<2x2xf32>
  }
}
