// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.subtract
// CHECK: stablehlo.dot_general
// CHECK: aicom.qdq_matmul_canonicalized

module {
  func.func @inference(%ax: tensor<2x3xf32>, %aw: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %scale_x = stablehlo.constant dense<0.1> : tensor<f32>
    %zp_x = stablehlo.constant dense<0.0> : tensor<f32>
    %sx_b = stablehlo.broadcast_in_dim %scale_x, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %zx_b = stablehlo.broadcast_in_dim %zp_x, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %sub_x = stablehlo.subtract %ax, %zx_b : tensor<2x3xf32>
    %dq_x = stablehlo.multiply %sub_x, %sx_b : tensor<2x3xf32>

    %scale_w = stablehlo.constant dense<0.2> : tensor<f32>
    %zp_w = stablehlo.constant dense<0.0> : tensor<f32>
    %sw_b = stablehlo.broadcast_in_dim %scale_w, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %zw_b = stablehlo.broadcast_in_dim %zp_w, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %sub_w = stablehlo.subtract %aw, %zw_b : tensor<3x4xf32>
    %dq_w = stablehlo.multiply %sub_w, %sw_b : tensor<3x4xf32>

    %out = stablehlo.dot_general %dq_x, %dq_w,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
