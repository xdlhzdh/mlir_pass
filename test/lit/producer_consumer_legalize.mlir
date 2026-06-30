// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.dot_general
// CHECK: stablehlo.multiply
// CHECK-SAME: aicom.producer_consumer_fused
// CHECK-NOT: stablehlo.subtract

module {
  func.func @inference() -> tensor<2x4xf32> {
    %a = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]> : tensor<2x4xf32>
    %b = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
    %neg_inf = stablehlo.constant dense<-3.402823e+38> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>

    %scores = stablehlo.dot_general %a, %b,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %max = "stablehlo.reduce"(%scores, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %m = stablehlo.maximum %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%m) : (tensor<f32>) -> ()
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %max_b = stablehlo.broadcast_in_dim %max, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
    %shifted = stablehlo.subtract %scores, %max_b : tensor<2x4xf32>
    %exp = stablehlo.exponential %shifted : tensor<2x4xf32>
    %sum = "stablehlo.reduce"(%exp, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %s = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%s) : (tensor<f32>) -> ()
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %sum_b = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>
    %softmax = stablehlo.divide %exp, %sum_b : tensor<2x4xf32>
    return %softmax : tensor<2x4xf32>
  }
}
