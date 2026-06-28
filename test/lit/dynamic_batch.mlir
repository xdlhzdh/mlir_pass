// RUN: %pipe-demo --input=%s --pipeline-stop-after=fusion 2>&1 | FileCheck %s
// RUN: %pipe-demo --input=%s --pipeline-stop-after=linalg 2>&1 | FileCheck %s --check-prefix=LINALG
// CHECK: tensor<?x3xf32>
// CHECK: tensor<?x4xf32>
// CHECK: stablehlo.add
// CHECK: stablehlo.dot_general
// LINALG: tensor<?x3xf32>
// LINALG: linalg.matmul

module {
  func.func @inference(%arg0: tensor<?x3xf32>, %bias: tensor<?x3xf32>) -> tensor<?x4xf32> {
    %w = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]> : tensor<3x4xf32>
    %add = stablehlo.add %arg0, %bias : tensor<?x3xf32>
    %mm = stablehlo.dot_general %add, %w,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
    return %mm : tensor<?x4xf32>
  }
}
