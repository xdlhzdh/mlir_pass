module {
  func.func @inference(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]> : tensor<3x4xf32>
    %1 = stablehlo.add %arg0, %arg1 : tensor<?x3xf32>
    %2 = stablehlo.dot_general %1, %0, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<?x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
    return %2 : tensor<?x4xf32>
  }
}
