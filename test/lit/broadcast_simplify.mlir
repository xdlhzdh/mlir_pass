// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s

// Identity broadcast (2x4 -> 2x4 with dims [0,1]) should be removed.
// CHECK-NOT: stablehlo.broadcast_in_dim
// CHECK: stablehlo.add
module {
  func.func @inference(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
