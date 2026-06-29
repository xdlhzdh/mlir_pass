// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.sqrt
// CHECK: chlo.erf
// CHECK: aicom.gelu_canonicalized

module {
  func.func @inference(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %half = stablehlo.constant dense<0.5> : tensor<f32>
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %two = stablehlo.constant dense<2.0> : tensor<f32>
    %sqrt2 = stablehlo.sqrt %two : tensor<f32>
    %sqrt2_b = stablehlo.broadcast_in_dim %sqrt2, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %scaled = stablehlo.divide %arg0, %sqrt2_b : tensor<2x4xf32>
    %erf = chlo.erf %scaled : tensor<2x4xf32> -> tensor<2x4xf32>
    %one_b = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %inner = stablehlo.add %one_b, %erf : tensor<2x4xf32>
    %half_b = stablehlo.broadcast_in_dim %half, dims = [] : (tensor<f32>) -> tensor<2x4xf32>
    %half_x = stablehlo.multiply %arg0, %half_b : tensor<2x4xf32>
    %gelu = stablehlo.multiply %half_x, %inner : tensor<2x4xf32>
    return %gelu : tensor<2x4xf32>
  }
}
