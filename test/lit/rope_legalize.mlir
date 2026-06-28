// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.slice
// CHECK: stablehlo.concatenate
// CHECK: aicom.rope_canonicalized

module {
  func.func @inference(%arg0: tensor<1x2x4xf32>) -> tensor<1x2x4xf32> {
    %cos = stablehlo.constant dense<1.0> : tensor<f32>
    %sin = stablehlo.constant dense<0.5> : tensor<f32>
    %neg = stablehlo.constant dense<-1.0> : tensor<f32>
    %cos_b = stablehlo.broadcast_in_dim %cos, dims = [] : (tensor<f32>) -> tensor<1x2x4xf32>
    %sin_b = stablehlo.broadcast_in_dim %sin, dims = [] : (tensor<f32>) -> tensor<1x2x4xf32>
    %neg_b = stablehlo.broadcast_in_dim %neg, dims = [] : (tensor<f32>) -> tensor<1x2x2xf32>
    %lo = stablehlo.slice %arg0 [0:1, 0:2, 0:2] : (tensor<1x2x4xf32>) -> tensor<1x2x2xf32>
    %hi = stablehlo.slice %arg0 [0:1, 0:2, 2:4] : (tensor<1x2x4xf32>) -> tensor<1x2x2xf32>
    %neg_hi = stablehlo.multiply %hi, %neg_b : tensor<1x2x2xf32>
    %rot = stablehlo.concatenate %neg_hi, %lo, dim = 2 : (tensor<1x2x2xf32>, tensor<1x2x2xf32>) -> tensor<1x2x4xf32>
    %x_cos = stablehlo.multiply %arg0, %cos_b : tensor<1x2x4xf32>
    %rot_sin = stablehlo.multiply %rot, %sin_b : tensor<1x2x4xf32>
    %out = stablehlo.add %x_cos, %rot_sin : tensor<1x2x4xf32>
    return %out : tensor<1x2x4xf32>
  }
}
