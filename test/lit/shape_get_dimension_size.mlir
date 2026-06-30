// RUN: %pipe-demo --input=%s --pipeline-stop-after=linalg 2>&1 | FileCheck %s

// Teaching: get_dimension_size refines batch before linalg (dim op lowered away).
// CHECK: tensor.cast %arg0 : tensor<?x3xf32> to tensor<2x3xf32>
// CHECK: linalg.matmul

module {
  func.func @inference(%arg0: tensor<?x3xf32>) -> tensor<?x4xf32> {
    %dim = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x3xf32>) -> tensor<i32>
    %w = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]> : tensor<3x4xf32>
    %bias = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    %add = stablehlo.add %arg0, %bias : (tensor<?x3xf32>, tensor<2x3xf32>) -> tensor<?x3xf32>
    %mm = stablehlo.dot_general %add, %w,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
    return %mm : tensor<?x4xf32>
  }
}
