// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.clamp
// CHECK-SAME: aicom.elementwise_chain_fused
// CHECK-NOT: stablehlo.maximum

module {
  func.func @inference() -> tensor<2x2xf32> {
    %a = stablehlo.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = stablehlo.constant dense<[[0.5, 0.5], [0.5, 0.5]]> : tensor<2x2xf32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %add = stablehlo.add %a, %b : tensor<2x2xf32>
    %zero_b = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<2x2xf32>
    %relu = stablehlo.maximum %add, %zero_b : tensor<2x2xf32>
    return %relu : tensor<2x2xf32>
  }
}
