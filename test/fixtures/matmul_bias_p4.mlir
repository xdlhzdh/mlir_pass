module {
  func.func @inference(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> {
    %0 = stablehlo.constant dense<[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<[0.25, 0.25, 0.25, 0.25]> : tensor<4xf32>
    %2 = stablehlo.dot_general %arg0, %0, batching_dims = [] x [], contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>
    %4 = stablehlo.add %2, %3 : tensor<2x4xf32>
    return %4 : tensor<2x4xf32>
  }
}
