// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.negate
// CHECK: stablehlo.exponential
// CHECK: aicom.swiglu_canonicalized

module {
  func.func @inference(%gate: tensor<2x4xf32>, %up: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %neg = stablehlo.negate %gate : tensor<2x4xf32>
    %exp = stablehlo.exponential %neg : tensor<2x4xf32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %denom = stablehlo.add %one_b, %exp : tensor<2x4xf32>
    %sigmoid = stablehlo.divide %one_b, %denom : tensor<2x4xf32>
    %silu = stablehlo.multiply %gate, %sigmoid : tensor<2x4xf32>
    %out = stablehlo.multiply %silu, %up : tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
