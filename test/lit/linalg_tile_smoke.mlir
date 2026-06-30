// RUN: %pipe-demo --input=%s --pipeline-stop-after=linalg 2>&1 | FileCheck %s

// 4x4 matmul lowered to linalg then strip-mined by custom-linalg-tile.
// CHECK-DAG: scf.for
// CHECK-DAG: aicom.linalg_tiled
// CHECK-DAG: linalg.matmul

module {
  func.func @inference(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %mm = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %mm : tensor<4x4xf32>
  }
}
