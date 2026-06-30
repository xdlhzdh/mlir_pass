// RUN: %pipe-demo --input=%s --pipeline-stop-after=linalg 2>&1 | FileCheck %s

// matmul + bias → tile then elementwise fuse on tiled nest.
// CHECK-DAG: aicom.linalg_tiled
// CHECK-DAG: scf.for

module {
  func.func @inference(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %bias = stablehlo.constant dense<0.5> : tensor<f32>
    %bias_b = stablehlo.broadcast_in_dim %bias, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
    %mm = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %out = stablehlo.add %mm, %bias_b : tensor<4x4xf32>
    return %out : tensor<4x4xf32>
  }
}
