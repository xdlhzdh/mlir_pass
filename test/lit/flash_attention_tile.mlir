// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK-DAG: stablehlo.dot_general
// CHECK-DAG: aicom.scaled_dot_product_attention
// CHECK-DAG: aicom.flash_tile

module {
  func.func @inference() -> tensor<1x2x4xf32> {
    %q = stablehlo.constant dense<[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]> : tensor<1x2x4xf32>
    %k = stablehlo.constant dense<[[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]> : tensor<1x4x2xf32>
    %v = stablehlo.constant dense<[[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]]> : tensor<1x2x4xf32>
    %scale = stablehlo.constant dense<0.5> : tensor<f32>
    %neg_inf = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %scores = stablehlo.dot_general %q, %k, batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x2x4xf32>, tensor<1x4x2xf32>) -> tensor<1x2x2xf32>
    %scale_b = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f32>) -> tensor<1x2x2xf32>
    %scaled = stablehlo.multiply %scores, %scale_b : tensor<1x2x2xf32>
    %max = "stablehlo.reduce"(%scaled, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%m) : (tensor<f32>) -> ()
    }) {
      dimensions = array<i64: 2>
    } : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %max_b = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %shifted = stablehlo.subtract %scaled, %max_b : tensor<1x2x2xf32>
    %exp = stablehlo.exponential %shifted : tensor<1x2x2xf32>
    %sum = "stablehlo.reduce"(%exp, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %s = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%s) : (tensor<f32>) -> ()
    }) {
      dimensions = array<i64: 2>
    } : (tensor<1x2x2xf32>, tensor<f32>) -> tensor<1x2xf32>
    %sum_b = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<1x2x2xf32>
    %attn = stablehlo.divide %exp, %sum_b : tensor<1x2x2xf32>
    %out = stablehlo.dot_general %attn, %v, batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<1x2x2xf32>, tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %out : tensor<1x2x4xf32>
  }
}
