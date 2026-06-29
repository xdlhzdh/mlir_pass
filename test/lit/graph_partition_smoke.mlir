// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// CHECK: stablehlo.dot_general
// CHECK: stablehlo.add
// CHECK-SAME: aicom.partition_boundary

module {
  func.func @inference() -> tensor<2x4xf32> {
    %x = stablehlo.constant dense<[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]> : tensor<2x4xf32>
    %w = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
    %attn_w = stablehlo.constant dense<[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]> : tensor<4x4xf32>

    %attn = stablehlo.dot_general %x, %attn_w,
        contracting_dims = [1] x [0]
        : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %hidden1 = stablehlo.add %x, %attn { aicom.partition_boundary } : tensor<2x4xf32>
    %ffn = stablehlo.dot_general %hidden1, %w,
        contracting_dims = [1] x [0]
        : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
    %y = stablehlo.add %hidden1, %ffn : tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }
}
