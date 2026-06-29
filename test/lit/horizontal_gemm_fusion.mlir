// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.dot_general
// CHECK: stablehlo.concatenate
// CHECK: aicom.horizontal_gemm_fused

module {
  func.func @inference(%x: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %w1 = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf32>
    %w2 = stablehlo.constant dense<[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]> : tensor<3x2xf32>
    %y1 = stablehlo.dot_general %x, %w1,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %y2 = stablehlo.dot_general %x, %w2,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    %out = stablehlo.concatenate %y1, %y2, dim = 1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
}
